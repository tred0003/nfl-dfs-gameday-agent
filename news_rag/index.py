"""Chunk the news archive, embed it, and load it into BigQuery for VECTOR_SEARCH.

    python news_rag/index.py

Reads data/news/articles.jsonl (collect.py) and replaces the table
`fantasy-football-ml-503718.dfs.news_chunks`.

Why BigQuery and not Vertex AI RAG Engine: the question this archive exists
to answer is "what could the agent have known before kickoff", and that needs
a hard filter on publish time applied BEFORE the nearest-neighbour search.
In BigQuery that is a WHERE clause inside VECTOR_SEARCH. RAG Engine's
retrieval filters are much thinner, and a time filter applied after top-k
returns fewer results instead of the right ones.

Chunking:
  player blurbs   one chunk each. Median 437 characters; splitting a blurb
                  would cut the injury from the player it is about.
  stories         ~1,200-character chunks on sentence boundaries, each one
                  prefixed with the headline so it still says what it is
                  about when retrieved alone.

Embeddings are cached in data/news/embeddings.parquet keyed on a hash of the
chunk text, so a re-run after collecting new articles only pays for the new
chunks.

Two columns exist for the agent, not for search quality:
  player_keys   the tagged players' names through predict_slate.normalize_name,
                the same rule the lineup sheet is matched with, so a DK name
                and an ESPN name meet on one spelling ("Kenneth Walker III"
                and "Kenneth Walker" are both "kenneth walker").
  screen        the injection screen's verdict on the chunk: clean,
                instruction or unchecked. Retrieval returns only "clean", so
                an unchecked chunk is withheld until a later run checks it --
                fail closed, like everywhere else third-party text reaches
                the agent. Screened per CHUNK, not per article: the screen
                reads at most 1,800 characters of an item, and a story is
                longer than that.

This file rebuilds the whole table from the laptop's articles.jsonl. The
daily cloud refresh that appends to it is refresh.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ARTICLES = ROOT / "data" / "news" / "articles.jsonl"
CACHE = ROOT / "data" / "news" / "embeddings.parquet"

BQ_PROJECT = os.environ.get("BQ_PROJECT", "fantasy-football-ml-503718")
TABLE = f"{BQ_PROJECT}.dfs.news_chunks"
EMBED_PROJECT = os.environ.get("EMBED_PROJECT", "advisor-agent-2026")
EMBED_MODEL = "text-embedding-005"
CHUNK_CHARS = 1200
BATCH = 50
# text-embedding-005 caps a request at 20,000 tokens. A fixed 50 chunks went
# 121 tokens over on the first run, so batches are cut by size as well.
BATCH_CHARS = 40_000


def _sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?\"”])\s+(?=[A-Z\"“])", text)


def chunks(article: dict) -> list[dict]:
    base = {
        "article_id": article["id"], "kind": article["kind"], "source": article["source"],
        "published": article["published"], "headline": article["headline"],
        "players": [p["name"] for p in article["players"]],
        "player_ids": [str(p["espn_id"]) for p in article["players"]],
        "teams": article["teams"], "url": article.get("url"),
    }
    if article["kind"] == "player_blurb":
        parts = [f"{article['headline']} {article['text']}".strip()]
    else:
        parts, cur = [], ""
        for s in _sentences(article["text"]):
            if cur and len(cur) + len(s) > CHUNK_CHARS:
                parts.append(cur)
                cur = ""
            cur = f"{cur} {s}".strip()
        if cur:
            parts.append(cur)
        parts = [f"{article['headline']}. {p}" for p in parts]
    return [base | {"chunk_id": f"{article['id']}#{i}", "chunk_index": i, "text": t,
                    "text_hash": hashlib.sha1(t.encode()).hexdigest()}
            for i, t in enumerate(parts)]


def player_keys(names: list[str]) -> list[str]:
    """Normalized names for the player_keys column. Same function the agent
    uses on DK names (gameday_agent/archive.py), or the two never meet."""
    from gameday_agent.archive import name_keys
    return name_keys(names)


from gameday_agent.injection import CONFIRM_MODEL  # noqa: E402,F401  (re-exported for tests)


def screen_texts(texts: list[str], workers: int = 4) -> list[str]:
    """One injection verdict per text: "clean", "instruction" or "unchecked".

    The agent's own two-step screen (injection.screen_confirmed: Flash-Lite,
    then CONFIRM_MODEL on whatever it flags), batched the way it batches,
    with batches run in parallel -- a backfill is ~1,000 of them.
    """
    from gameday_agent.injection import BATCH, screen_confirmed
    groups = [texts[i:i + BATCH] for i in range(0, len(texts), BATCH)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return [v["verdict"] for g in ex.map(screen_confirmed, groups) for v in g]


def _client():
    from google import genai
    from google.genai import types
    retry = types.HttpRetryOptions(attempts=6, initial_delay=2.0, max_delay=60.0,
                                   exp_base=2.0, http_status_codes=[429, 500, 503, 504])
    return genai.Client(vertexai=True, project=EMBED_PROJECT, location="us-central1",
                        http_options=types.HttpOptions(retry_options=retry))


def embed(texts: list[str], task: str = "RETRIEVAL_DOCUMENT", client=None) -> list[list[float]]:
    from google.genai import types
    client = client or _client()
    batches, cur, size = [], [], 0
    for t in texts:
        if cur and (len(cur) == BATCH or size + len(t) > BATCH_CHARS):
            batches.append(cur)
            cur, size = [], 0
        cur.append(t)
        size += len(t)
    if cur:
        batches.append(cur)
    out: list[list[float]] = []
    for b in batches:
        resp = client.models.embed_content(
            model=EMBED_MODEL, contents=b,
            config=types.EmbedContentConfig(task_type=task))
        out += [e.values for e in resp.embeddings]
    return out


def schema_fields():
    from google.cloud import bigquery
    return [
        bigquery.SchemaField("chunk_id", "STRING"), bigquery.SchemaField("article_id", "STRING"),
        bigquery.SchemaField("chunk_index", "INT64"), bigquery.SchemaField("kind", "STRING"),
        bigquery.SchemaField("source", "STRING"), bigquery.SchemaField("published", "TIMESTAMP"),
        bigquery.SchemaField("headline", "STRING"), bigquery.SchemaField("text", "STRING"),
        bigquery.SchemaField("players", "STRING", mode="REPEATED"),
        bigquery.SchemaField("player_ids", "STRING", mode="REPEATED"),
        bigquery.SchemaField("teams", "STRING", mode="REPEATED"),
        bigquery.SchemaField("url", "STRING"), bigquery.SchemaField("text_hash", "STRING"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
        bigquery.SchemaField("player_keys", "STRING", mode="REPEATED"),
        bigquery.SchemaField("screen", "STRING"),
    ]


def main() -> None:
    articles = [json.loads(line) for line in ARTICLES.open(encoding="utf-8")]
    rows = pd.DataFrame([c for a in articles for c in chunks(a)])
    print(f"{len(articles)} articles -> {len(rows)} chunks "
          f"({(rows['kind'] == 'player_blurb').sum()} blurbs, "
          f"{(rows['kind'] != 'player_blurb').sum()} story chunks)")

    cache = pd.read_parquet(CACHE) if CACHE.exists() else pd.DataFrame(columns=["text_hash", "embedding"])
    known = dict(zip(cache["text_hash"], cache["embedding"]))
    todo = rows.loc[~rows["text_hash"].isin(known), ["text_hash", "text"]].drop_duplicates("text_hash")
    print(f"embedding {len(todo)} new chunks ({len(rows) - len(todo)} cached)")

    if len(todo):
        client = _client()
        groups = [todo.iloc[i:i + BATCH * 4] for i in range(0, len(todo), BATCH * 4)]
        with ThreadPoolExecutor(max_workers=4) as ex:
            for g, vecs in zip(groups, ex.map(lambda g: embed(g["text"].tolist(), client=client), groups)):
                known.update(zip(g["text_hash"], vecs))
        pd.DataFrame({"text_hash": list(known), "embedding": [list(map(float, v)) for v in known.values()]}
                     ).to_parquet(CACHE, index=False)

    rows["embedding"] = rows["text_hash"].map(lambda h: [float(x) for x in known[h]])
    rows["published"] = pd.to_datetime(rows["published"], utc=True)
    rows["player_keys"] = rows["players"].map(player_keys)
    print(f"screening {len(rows)} chunks for injection")
    rows["screen"] = screen_texts(rows["text"].tolist())

    from google.cloud import bigquery
    bq = bigquery.Client(project=BQ_PROJECT)
    schema = schema_fields()
    cfg = bigquery.LoadJobConfig(schema=schema, write_disposition="WRITE_TRUNCATE",
                                 time_partitioning=bigquery.TimePartitioning(field="published"))
    bq.load_table_from_dataframe(rows[[f.name for f in schema]], TABLE, job_config=cfg).result()
    print(f"loaded {len(rows)} chunks into {TABLE}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
