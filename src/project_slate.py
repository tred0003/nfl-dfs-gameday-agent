"""Cloud job: project this week's DK main slate and write it to BigQuery.

This is what the project-slate Cloud Run job runs (Friday evening and Sunday
11:00 ET, project fantasy-football-ml-503718). The gameday agent's
get_value_picks tool reads its output; the agent never runs the model itself.

    python src/project_slate.py                 # the cloud run
    python src/project_slate.py --dry-run       # everything except the writes
    python src/project_slate.py --local-data    # use data/raw as-is, no GCS sync

WHAT ONE RUN DOES

  1. Pull raw history from gs://BUCKET/raw/ into data/raw/ (the container
     starts empty; the history is ~285 MB and changes only for this season).
  2. Refresh THIS season from nflverse -- stats, injuries, snaps, schedules,
     play-by-play -- and push the refreshed files back, so the next run starts
     current and the laptop can pull the same files.
  3. Fetch the DK main slate and save it to gs://BUCKET/slates/SEASON/wkNN/.
     That archive is the salary history for later weeks: salary features shift
     by ROW, not by week, so a missing week does not show up as a gap -- it
     silently turns "salary change since last week" into a longer change.
     Weeks 1-5 of 2026 were never saved and had to be downloaded by hand.
  4. Score with whatever the Model Registry calls @champion RIGHT NOW. Moving
     the alias is the whole deploy: no code change, no image rebuild.
  5. predict_week.build_pool() -- the same function the laptop CLI runs, so
     the cloud and local projections cannot drift apart.
  6. Append the rows to BigQuery, one run_id per run. Append, not replace: a
     history of what was projected before kickoff is what lets the projections
     be graded against actual points later.

DST HAS NO MODEL. Its `projection` is DK's FPPG scaled by the Vegas implied
total (predict_slate.project_dk_vegas), and `method` says so on every row.

The job FAILS -- exit 1, which the project's alerting sees -- on a preflight
FATAL, on stats that do not reach last week, or on an empty slate. A missing
table is better than a table that looks right and is not.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import dk_client  # noqa: E402
import fetch_nflverse  # noqa: E402
import nfl_week  # noqa: E402
import predict_week  # noqa: E402
import suggest_lineup  # noqa: E402

PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "fantasy-football-ml-503718")
REGION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
BUCKET = os.environ.get("DATA_BUCKET", "fantasy-football-ml-503718-nfl-dfs")
MODEL_ID = os.environ.get("MODEL_ID", "7035124545010794496")  # nfl-dfs-projection
MODEL_ALIAS = os.environ.get("MODEL_ALIAS", "champion")
BQ_TABLE = os.environ.get("BQ_TABLE", f"{PROJECT}.dfs.projections")

RAW = ROOT / "data" / "raw"
# nflverse tables refreshed in place. Depth charts are left out on purpose:
# nflverse switched 2026 to a timestamped format with no `week` column, which
# crashes update() (and leaves depth_rank empty for 2026 -- a known gap).
REFRESH = ["injuries", "schedules", "snaps", "weekly"]
# The raw files those refreshes rewrite, plus this season's play-by-play.
PUSHED = ["injuries.parquet", "schedules.parquet", "snaps.parquet", "weekly.parquet"]


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {msg}", flush=True)


# ------------------------------------------------------------------ storage
def _bucket():
    from google.cloud import storage
    return storage.Client(project=PROJECT).bucket(BUCKET)


def sync_down() -> None:
    n = 0
    for blob in _bucket().list_blobs(prefix="raw/"):
        if blob.name.endswith("/"):
            continue
        dest = RAW / blob.name[len("raw/"):]
        dest.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(dest)
        n += 1
    log(f"pulled {n} raw files from gs://{BUCKET}/raw/")


def push(paths: list[Path]) -> None:
    b = _bucket()
    for p in paths:
        if p.exists():
            b.blob(f"raw/{p.relative_to(RAW).as_posix()}").upload_from_filename(p)
    log(f"pushed {len(paths)} refreshed files to gs://{BUCKET}/raw/")


def refresh(season: int) -> list[Path]:
    fetch_nflverse.update(season, only=REFRESH)
    fetch_nflverse._pull_release("pbp", "play_by_play_{year}.parquet", [season],
                                 "pbp", "play-by-play", refresh=True)
    return [RAW / f for f in PUSHED] + [RAW / "pbp" / f"play_by_play_{season}.parquet"]


# ------------------------------------------------------------------ slate
def fetch_slate(season: int) -> tuple[int, int, pd.DataFrame, Path]:
    """(draft_group_id, week, slate, local players parquet) for the main slate."""
    slates = dk_client.list_slates()
    dg = dk_client.pick_main_slate(slates)
    kickoff = slates.loc[slates["draft_group_id"].eq(dg), "start_time_utc"].iloc[0]
    week = nfl_week.current_week(season, kickoff_utc=kickoff)
    out = dk_client.snapshot(dg)
    players = sorted(out.glob("players_*.parquet"))[-1]
    slate = predict_week.load_slate(str(players), season, week)
    if slate.empty:
        raise SystemExit(f"draft group {dg} came back with no players")
    log(f"main slate dg{dg}: {len(slate)} players, {slate['game'].nunique()} games, week {week}")
    return dg, week, slate, players


def archive_slate(players: Path, season: int, week: int) -> None:
    _bucket().blob(f"slates/{season}/wk{week:02d}/{players.name}").upload_from_filename(players)
    log(f"archived slate -> gs://{BUCKET}/slates/{season}/wk{week:02d}/{players.name}")


def slate_history(season: int, week: int) -> dict[int, pd.DataFrame]:
    """The latest archived slate for every earlier week of this season."""
    latest: dict[int, object] = {}
    for blob in _bucket().list_blobs(prefix=f"slates/{season}/"):
        parts = blob.name.split("/")  # slates/2026/wk06/players_<ts>.parquet
        if len(parts) != 4 or not parts[2].startswith("wk"):
            continue
        wk = int(parts[2][2:])
        if wk < week and (wk not in latest or blob.name > latest[wk].name):
            latest[wk] = blob
    hist = {}
    for wk, blob in sorted(latest.items()):
        local = ROOT / "data" / "slates" / "history" / f"{season}_wk{wk:02d}.parquet"
        local.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local)
        hist[wk] = predict_week.load_slate(str(local), season, wk)
    log(f"salary history from archived slates: weeks {sorted(hist) or 'none'}"
        " (FantasyData CSVs in raw/ cover the rest)")
    return hist


# ------------------------------------------------------------------ model
def resolve_model() -> tuple[str, str]:
    """(artifact URI, version id) of the model the registry calls @ALIAS now."""
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    url = (f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT}"
           f"/locations/{REGION}/models/{MODEL_ID}@{MODEL_ALIAS}")
    r = AuthorizedSession(creds).get(url, timeout=30)
    r.raise_for_status()
    m = r.json()
    log(f"@{MODEL_ALIAS} = version {m['versionId']} -> {m['artifactUri']}")
    return m["artifactUri"], m["versionId"]


def check_stats_current(season: int, week: int) -> None:
    """Stats must reach last week, or every rolling feature is a week stale --
    and nothing downstream can tell that from a quiet week."""
    w = pd.read_parquet(RAW / "weekly.parquet", columns=["season", "week"])
    have = pd.to_numeric(w.loc[w["season"].eq(season), "week"]).max()
    if week > 1 and (pd.isna(have) or have < week - 1):
        raise SystemExit(f"weekly stats reach {season} week {have}, need week {week - 1} "
                         f"before projecting week {week}. nflverse has not posted yet.")
    log(f"weekly stats through {season} week {int(have) if pd.notna(have) else '-'}")


# ------------------------------------------------------------------ output
def to_rows(pool: pd.DataFrame, *, run_id: str, run_ts: datetime, season: int,
            week: int, dg: int, version: str, uri: str) -> pd.DataFrame:
    is_dst = pool["position"].eq("DST")
    out = pd.DataFrame({
        "run_id": run_id,
        "run_ts": run_ts,
        "season": season,
        "week": week,
        "draft_group_id": dg,
        "name": pool["name"],
        "position": pool["position"],
        "team": pool["team"],
        "opp_team": pool["opp_team"],
        "game": pool["game"],
        "salary": pool["salary"].astype("int64"),
        "status": pool["status"].astype("string"),
        "projection": pool["projection"].astype("float64"),
        "ceiling": pool["ceiling"].astype("float64"),
        "salary_implied_pts": pool["salary_implied_pts"].astype("float64"),
        "value": pool["value"].astype("float64"),
        "depth_rank": pd.to_numeric(pool["depth_rank"], errors="coerce").astype("float64"),
        "method": is_dst.map({True: "dst_rule_dk_fppg_x_vegas",
                              False: f"xgb_v3@{MODEL_ALIAS}_v{version}"}),
        "model_version": version,
        "model_uri": uri,
    })
    return out


def write_bq(rows: pd.DataFrame) -> None:
    from google.cloud import bigquery
    client = bigquery.Client(project=PROJECT)
    cfg = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND",
                                 time_partitioning=bigquery.TimePartitioning(field="run_ts"))
    client.load_table_from_dataframe(rows, BQ_TABLE, job_config=cfg).result()
    log(f"appended {len(rows)} rows to {BQ_TABLE} (run_id {rows['run_id'].iloc[0]})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Project the DK main slate -> BigQuery")
    ap.add_argument("--season", type=int, default=int(os.environ.get("SEASON", "2026")))
    ap.add_argument("--dry-run", action="store_true", help="no GCS pushes, no BigQuery write")
    ap.add_argument("--local-data", action="store_true",
                    help="use data/raw as it is: no GCS pull, no nflverse refresh")
    ap.add_argument("--post-lineups", action="store_true",
                    help="after writing, post suggested lineups to Discord "
                         "(Thursday and Sunday schedules; not Friday)")
    a = ap.parse_args()
    run_ts = datetime.now(timezone.utc)
    run_id = f"{run_ts:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"

    if not a.local_data:
        sync_down()
        touched = refresh(a.season)
        if not a.dry_run:
            push(touched)

    dg, week, slate, players = fetch_slate(a.season)
    check_stats_current(a.season, week)
    if not a.dry_run:
        archive_slate(players, a.season, week)
    history = {} if a.local_data else slate_history(a.season, week)

    uri, version = resolve_model()
    pool, unmatched = predict_week.build_pool(a.season, week, slate, history=history,
                                              model_dir=uri, strict=True)
    rows = to_rows(pool, run_id=run_id, run_ts=run_ts, season=a.season, week=week,
                   dg=dg, version=version, uri=uri)
    log(f"{int(rows['projection'].notna().sum())} projected, "
        f"{len(unmatched)} DK names not matched to an NFL id")

    top = (rows.dropna(subset=["projection"])
           .assign(rank_by=lambda d: d["value"].fillna(d["projection"]))
           .sort_values("rank_by", ascending=False).groupby("position").head(3))
    print(top[["name", "position", "team", "salary", "projection", "value", "method"]]
          .round(1).to_string(index=False), flush=True)

    if a.dry_run:
        out = ROOT / "data" / "lineups" / f"{a.season}_wk{week:02d}_cloud_dryrun.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(out, index=False)
        log(f"dry run -- not written to BigQuery. rows -> {out}")
        if a.post_lineups:
            suggest_lineup.post(rows, dry_run=True)
        return 0
    if not suggest_lineup.gate_write(rows, notify=a.post_lineups):
        log("SANITY CHECK FAILED: not writing this run to BigQuery")
        return 1
    write_bq(rows)
    if a.post_lineups:
        # After the write, so a failed post never costs the projections. A
        # failed post still exits 1 so the job alert fires.
        try:
            sent = suggest_lineup.post(rows)
        except suggest_lineup.ProjectionCheckFailed:
            log("SANITY CHECK FAILED: lineups withheld (details above)")
            return 1
        if not sent:
            log("lineups NOT posted: no DISCORD_WEBHOOK_URL")
            return 1
        log("suggested lineups posted to Discord")
    return 0


if __name__ == "__main__":
    sys.exit(main())
