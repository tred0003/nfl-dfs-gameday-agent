"""Daily refresh of the news archive -- the `news-archive` Cloud Run job.

    python news_rag/refresh.py             # collect, screen, embed, append, backfill
    python news_rag/refresh.py --dry-run   # collect and count; write nothing

index.py built dfs.news_chunks once, from the laptop, on 2026-10-14. Nothing
appended to it after that, so by the time an agent tool read it the archive
would have ended a week before the report. This job keeps it current:

  1. Add the player_keys / screen columns if the table predates them. Idempotent.
  2. Collect everything published since the archive's newest item, minus
     OVERLAP_HOURS -- ESPN's player feeds sometimes surface a note hours after
     its publish time, and a window that starts exactly at the newest item
     would miss it. Chunks already in the table are dropped by chunk_id.
  3. For the new chunks: player_keys, the injection screen, the embedding.
     Then one load job, WRITE_APPEND (no streaming buffer, so the MERGE in
     step 4 never trips over rows it cannot update yet).
  4. Fill: rows missing player_keys, or never screened, or screened
     "unchecked" (the screen call failed), get both computed and MERGEd in.
     On the first run that is the whole 19K-chunk backfill; after that, it is
     whatever a failed screen call left behind. Retrieval only ever returns
     screen = 'clean', so an unchecked chunk is invisible until this fixes it.

FAILS (exit 1, which the ML project's failed-job alert emails) when the
collection itself fails, or when the run appends nothing AND the archive's
newest item is older than STALE_HOURS. A quiet day appends little; a day that
appends nothing to an archive already two days old is a broken feed.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from news_rag.collect import CollectError, collect  # noqa: E402
from news_rag.index import (  # noqa: E402
    BQ_PROJECT, TABLE, _client, chunks, embed, player_keys, schema_fields, screen_texts,
)

OVERLAP_HOURS = 36
# The table's first article is 2026-07-15. A refresh against an empty table
# starts there, so the job can also rebuild from nothing.
FIRST_DAY = datetime(2026, 7, 15, tzinfo=timezone.utc)
STALE_HOURS = 36            # same as archive.STALE_HOURS and the canary's
# A run where more than this share of screened chunks came back "unchecked"
# means the screen itself is down: those chunks are invisible to the agent,
# so the archive is going stale while appending. Fails the job.
MAX_UNCHECKED = 0.2
FILL_LIMIT = 25_000          # rows fixed per run; the first backfill is ~19K
FILL_TABLE = f"{BQ_PROJECT}.dfs._news_chunks_fill"
SCREEN_WORKERS = 8


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc):%H:%M:%S} {msg}", flush=True)


def _bq():
    from google.cloud import bigquery
    return bigquery.Client(project=BQ_PROJECT)


def ensure_columns(bq) -> None:
    for ddl in (f"ALTER TABLE `{TABLE}` ADD COLUMN IF NOT EXISTS player_keys ARRAY<STRING>",
                f"ALTER TABLE `{TABLE}` ADD COLUMN IF NOT EXISTS screen STRING"):
        bq.query(ddl).result()


def newest_published(bq) -> datetime | None:
    rows = list(bq.query(f"SELECT MAX(published) AS m FROM `{TABLE}`").result())
    return rows[0]["m"] if rows else None


def existing_ids(bq, ids: list[str], since: datetime) -> set[str]:
    """Article ids already in the table. By ARTICLE, not chunk: an article
    re-collected with a different chunking (a column reclassified from
    player_blurb to story, 10/22) would otherwise get its extra chunks
    appended beside the old ones."""
    from google.cloud import bigquery
    if not ids:
        return set()
    cfg = bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("ids", "STRING", ids),
        bigquery.ScalarQueryParameter("since", "TIMESTAMP", since - timedelta(days=1)),
    ])
    q = (f"SELECT DISTINCT article_id FROM `{TABLE}` "
         f"WHERE published >= @since AND article_id IN UNNEST(@ids)")
    return {r["article_id"] for r in bq.query(q, job_config=cfg).result()}


def new_chunks(articles: list[dict], known: set[str]) -> pd.DataFrame:
    rows = [c for a in articles if a["id"] not in known for c in chunks(a)]
    return pd.DataFrame(rows).drop_duplicates("chunk_id") if rows else pd.DataFrame()


def append(bq, rows: pd.DataFrame) -> None:
    from google.cloud import bigquery
    schema = schema_fields()
    rows = rows.copy()
    rows["published"] = pd.to_datetime(rows["published"], utc=True)
    cfg = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_APPEND")
    bq.load_table_from_dataframe(rows[[f.name for f in schema]], TABLE, job_config=cfg).result()


def fill(bq, dry_run: bool) -> dict:
    """Compute player_keys and screen for rows that lack them; MERGE them in."""
    from google.cloud import bigquery
    q = f"""
        SELECT chunk_id, text, players FROM `{TABLE}`
        WHERE (ARRAY_LENGTH(player_keys) = 0 AND ARRAY_LENGTH(players) > 0)
           OR screen IS NULL OR screen = 'unchecked'
        LIMIT {FILL_LIMIT}
    """
    if dry_run and not {"player_keys", "screen"} <= {f.name for f in bq.get_table(TABLE).schema}:
        log("fill: columns not added yet (a real run adds them); every row will need filling")
        return {"filled": 0}
    todo = bq.query(q).to_dataframe().drop_duplicates("chunk_id")
    log(f"fill: {len(todo)} rows need player_keys and/or a screen verdict")
    if todo.empty or dry_run:
        return {"filled": 0, "fill_pending": len(todo)}
    todo["player_keys"] = todo["players"].map(lambda p: player_keys(list(p)))
    started = time.perf_counter()
    todo["screen"] = screen_texts(todo["text"].tolist(), workers=SCREEN_WORKERS)
    log(f"fill: screened {len(todo)} in {time.perf_counter() - started:.0f}s "
        f"({(todo['screen'] == 'instruction').sum()} instruction, "
        f"{(todo['screen'] == 'unchecked').sum()} unchecked)")
    cfg = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=[bigquery.SchemaField("chunk_id", "STRING"),
                bigquery.SchemaField("player_keys", "STRING", mode="REPEATED"),
                bigquery.SchemaField("screen", "STRING")])
    bq.load_table_from_dataframe(todo[["chunk_id", "player_keys", "screen"]], FILL_TABLE,
                                 job_config=cfg).result()
    bq.query(f"""
        MERGE `{TABLE}` t USING `{FILL_TABLE}` f ON t.chunk_id = f.chunk_id
        WHEN MATCHED THEN UPDATE SET player_keys = f.player_keys, screen = f.screen
    """).result()
    bq.delete_table(FILL_TABLE, not_found_ok=True)
    return {"filled": len(todo),
            "fill_flagged": int((todo["screen"] == "instruction").sum()),
            "fill_unchecked": int((todo["screen"] == "unchecked").sum())}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true", help="collect and count; write nothing")
    a = p.parse_args()

    bq = _bq()
    if not a.dry_run:
        ensure_columns(bq)
    newest = newest_published(bq)
    since = (newest - timedelta(hours=OVERLAP_HOURS)) if newest else FIRST_DAY
    log(f"archive newest item: {newest}; collecting since {since:%Y-%m-%d %H:%M} UTC")

    try:
        articles = collect(since, log=log)
    except (CollectError, OSError) as e:
        log(f"ARCHIVE REFRESH FAILED: collection failed ({type(e).__name__}: {e})")
        return 1

    rows = new_chunks(articles, existing_ids(bq, [art["id"] for art in articles], since))
    log(f"{len(articles)} articles since the window start -> {len(rows)} new chunks")

    stats = {"new_chunks": len(rows)}
    if len(rows) and not a.dry_run:
        rows["player_keys"] = rows["players"].map(player_keys)
        rows["screen"] = screen_texts(rows["text"].tolist(), workers=SCREEN_WORKERS)
        vecs = embed(rows["text"].tolist(), client=_client())
        rows["embedding"] = [[float(x) for x in v] for v in vecs]
        append(bq, rows)
        stats |= {"flagged": int((rows["screen"] == "instruction").sum()),
                  "unchecked": int((rows["screen"] == "unchecked").sum())}
        log(f"appended {len(rows)} chunks to {TABLE}")

    stats |= fill(bq, a.dry_run)
    newest = newest_published(bq)
    age_h = (datetime.now(timezone.utc) - newest).total_seconds() / 3600 if newest else None
    stats["newest_hours_old"] = round(age_h, 1) if age_h is not None else None
    # One line, fixed prefix: the log-based metric for monitoring reads it.
    log("ARCHIVE REFRESH " + " ".join(f"{k}={v}" for k, v in stats.items()))

    if not len(rows) and (age_h is None or age_h > STALE_HOURS):
        log(f"ARCHIVE REFRESH FAILED: nothing new and the newest item is {stats['newest_hours_old']} h old")
        return 1
    screened = len(rows) + stats.get("filled", 0)
    unchecked = stats.get("unchecked", 0) + stats.get("fill_unchecked", 0)
    if screened and unchecked / screened > MAX_UNCHECKED:
        log(f"ARCHIVE REFRESH FAILED: {unchecked} of {screened} chunks could not be screened "
            "-- the injection screen is failing, so they are hidden from the agent")
        return 1
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
