"""Write the training-distribution baseline that Model Monitoring drifts against.

WHY THIS FILE EXISTS AND ISN'T JUST model_table_v3.parquet

Vertex AI Model Monitoring's GCS source only reads csv, jsonl, or tf-record --
there is no parquet option (checked against
ModelMonitoringGcsSource.DataFormat directly: CSV, TF_RECORD, JSONL, nothing
else). The champion artifact's own training table can't be pointed at as-is;
it has to be re-exported in one of those three shapes.

WHY THE WHOLE TABLE, NOT A SAMPLE OR A DATE RANGE

model_card.json records train_rows: 51356 -- the same row count as
model_table_v3.parquet in full. The champion was trained on every row in the
table, not a slice of it, so the baseline for drift has to be every row too.
Sampling here would compare this week's slate against a baseline the model
never actually saw in full, which measures the sample, not the model.

SAME SHAPE AS THE WEEKLY BATCH INPUT, ON PURPOSE

export_batch_input.py sends `position` plus the 79 real (non pos_*) card
features -- the four pos_QB/RB/TE/WR dummies are rebuilt from `position` at
serve time and were never in the table to begin with. The monitoring schema
declared in deploy/create_model_monitor.py has to name columns that actually
exist in both files, so this script reuses the identical selection logic.
Divergence here is exactly the kind of thing that fails silently: the job
runs, reports success, and "monitors" a column that's absent from one side.

Run:
    python src/export_monitoring_baseline.py
    python src/export_monitoring_baseline.py --upload
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import model_io  # noqa: E402

TABLES = ROOT / "data" / "processed"
BATCH = ROOT / "data" / "batch"

# Identical to export_batch_input.py's PASSTHROUGH. `position` rides along
# here specifically because it's the categorical field the monitoring schema
# needs -- everything else in this list is for the join, not the drift check.
PASSTHROUGH = ["player_id", "player_display_name", "position", "recent_team",
               "opponent_team", "season", "week"]


def build(table: str, card: dict) -> pd.DataFrame:
    t = pd.read_parquet(TABLES / f"{table}.parquet")

    feats = card["features"]
    derived = {f for f in feats if f.startswith("pos_")}
    absent = [f for f in feats if f not in t.columns and f not in derived]
    if absent:
        raise SystemExit(
            f"{table}.parquet is missing {len(absent)} card feature(s), can't "
            f"build a baseline that matches the serving contract: {absent}"
        )

    send = [c for c in PASSTHROUGH if c in t.columns]
    send += [f for f in feats if f in t.columns and f not in send]
    return t[send]


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    # Same reason as export_batch_input.py: to_json writes NaN as JSON null;
    # json.dumps would write a bare `NaN` token that isn't valid JSON at all.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for line in df.to_json(orient="records", lines=True, date_format="iso").splitlines():
            fh.write(line + "\n")


def upload(local: Path, gcs_uri: str) -> None:
    from google.cloud import storage

    bucket_name, _, blob_name = gcs_uri[len("gs://"):].partition("/")
    storage.Client().bucket(bucket_name).blob(blob_name).upload_from_filename(str(local))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="model_table_v3")
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "current"))
    ap.add_argument("--upload", action="store_true", help="also push to GCS")
    ap.add_argument("--gcs-uri", default=None,
                     help="defaults to gs://<bucket>/monitoring/baseline/<table>_baseline.jsonl, "
                          "reading the bucket name from .env.example's convention if not given")
    ap.add_argument("--bucket", default="fantasy-football-ml-503718-nfl-dfs")
    a = ap.parse_args()

    _, card = model_io.load(a.model_dir)
    print(f"card: {card['n_features']} features, git_sha {card.get('git_sha')}, "
          f"trained on {card.get('train_rows'):,} rows")

    rows = build(a.table, card)
    out = BATCH / f"{a.table}_baseline.jsonl"
    write_jsonl(rows, out)

    feats_sent = [c for c in rows.columns if c in card["features"]]
    nulls = rows[feats_sent].isna().sum().sum()
    print(f"{len(rows):,} rows -> {out}")
    print(f"  {len(feats_sent)} of {card['n_features']} features present "
          f"('position' carries the 4 pos_ dummies as one categorical column)")
    print(f"  {nulls:,} null cells across {len(rows) * len(feats_sent):,} "
          f"({nulls / (len(rows) * len(feats_sent)):.1%}) -- this IS the baseline's "
          f"null rate; drift on a feature's null-ness is a real thing to catch")

    if a.upload:
        gcs_uri = a.gcs_uri or f"gs://{a.bucket}/monitoring/baseline/{a.table}_baseline.jsonl"
        upload(out, gcs_uri)
        print(f"  uploaded -> {gcs_uri}")


if __name__ == "__main__":
    main()
