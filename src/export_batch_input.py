"""Write a batch-prediction input file: one JSON object per line, one per player.

WHY A SEPARATE SCRIPT AND NOT A FLAG ON predict_week.py

predict_week.py refits the model in memory and predicts inside the same process.
That is exactly the thing 4.4 replaces. This script deliberately stops at the
INPUT: it produces the rows and hands them off. What scores them is a registered
model version running in a container the platform starts -- not an xgboost object
that exists for ninety seconds inside this interpreter.

THE FILE FORMAT, AND THE TWO WAYS IT SILENTLY BREAKS

Agent Platform batch prediction with instancesFormat=jsonl reads ONE JSON object
per line and packs lines into {"instances": [...]} requests against the model's
predict route. Two failure modes, both quiet:

  1. NaN IS NOT JSON. json.dumps happily writes a bare NaN token, which is not
     valid JSON, and the job fails per-line with an error you read in Cloud
     Logging twenty minutes later. pandas' to_json writes null instead, which is
     what the platform expects and what serve.py reads back as a missing value.
     That is the whole reason this uses to_json rather than json.dumps.

  2. THE POSITION DUMMIES ARE NOT IN THE TABLE. The card pins 83 features; the
     model table carries 79. pos_QB/pos_RB/pos_TE/pos_WR are built at predict
     time from `position`, and serve.py rebuilds them from the card on every
     request. So the input carries `position` and NOT the dummies -- sending
     pre-built dummies would work too, but it would test a path no real caller
     uses.

THE LABEL DOES NOT GO IN THE REQUEST

dk_points is written to a separate local file keyed by player_id. align() would
drop it harmlessly, but an inference payload carrying its own target is a habit
that eventually ships a leak somewhere it is not harmless. Join it back after.

PASSTHROUGH COLUMNS ARE FREE AND WORTH IT

align() drops any column not in the card before the model sees anything, and the
jsonl output echoes each instance back beside its prediction. So player_id and
name ride along at zero cost to the model and come back as the join key. Without
them the output is 318 anonymous numbers in whatever order the replicas finished.

Run:
    python src/export_batch_input.py --season 2025 --week 17
    python src/export_batch_input.py --season 2025 --week 17 --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import model_io  # noqa: E402

TABLES = ROOT / "data" / "processed"
BATCH = ROOT / "data" / "batch"
LABEL = "dk_points"

# Ride along for the join; dropped by align() before the model sees them.
PASSTHROUGH = ["player_id", "player_display_name", "position", "recent_team",
               "opponent_team", "season", "week"]


def build(table: str, season: int, week: int,
          card: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    t = pd.read_parquet(TABLES / f"{table}.parquet")
    rows = t[t["season"].eq(season) & t["week"].eq(week)].copy()
    if rows.empty:
        raise SystemExit(f"no rows for {season} week {week} in {table}.parquet")

    feats = card["features"]
    # The dummies are built downstream from `position`; everything else must
    # already be here. Anything in neither bucket is a genuine gap and the
    # count is printed, because a request full of nulls returns a confident
    # near-mean number and nothing about it looks wrong.
    derived = {f for f in feats if f.startswith("pos_")}
    absent = [f for f in feats if f not in rows.columns and f not in derived]

    send = [c for c in PASSTHROUGH if c in rows.columns]
    send += [f for f in feats if f in rows.columns and f not in send]
    out = rows[send]

    labels = rows[["player_id", LABEL]] if LABEL in rows.columns else pd.DataFrame()
    return out, labels, absent


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    # to_json per row: NaN -> null, numpy int64/float64 -> JSON numbers. Both
    # are things json.dumps gets wrong.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for line in df.to_json(orient="records", lines=True, date_format="iso").splitlines():
            fh.write(line + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--table", default="model_table_v3")
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "current"),
                    help="artifact whose card defines the feature contract")
    ap.add_argument("--check", action="store_true",
                    help="also predict locally, for the batch-vs-local skew comparison")
    a = ap.parse_args()

    models, card = model_io.load(a.model_dir)
    print(f"card: {card['n_features']} features, git_sha {card.get('git_sha')}, "
          f"calibration {'yes' if card.get('calibration') else 'NONE'}")

    rows, labels, absent = build(a.table, a.season, a.week, card)
    stem = f"{a.season}_wk{a.week:02d}"
    jsonl = BATCH / f"{stem}_input.jsonl"
    write_jsonl(rows, jsonl)

    sent = [c for c in rows.columns if c in card["features"]]
    print(f"{len(rows)} rows -> {jsonl}")
    print(f"  {len(sent)} of {card['n_features']} features sent; "
          f"{len([f for f in card['features'] if f.startswith('pos_')])} rebuilt from `position` at serve time")
    if absent:
        print(f"  WARNING: {len(absent)} feature(s) in neither place, will arrive null: {absent}")
    nulls = rows[sent].isna().sum().sum()
    print(f"  {nulls:,} null cells across {len(rows) * len(sent):,} "
          f"({nulls / (len(rows) * len(sent)):.1%}) -- nulls are legitimate to XGBoost, "
          f"but a jump here means the table changed, not the players")

    if not labels.empty:
        labels.to_parquet(BATCH / f"{stem}_actuals.parquet", index=False)
        print(f"  actuals held back -> {BATCH / f'{stem}_actuals.parquet'}")

    if a.check:
        # The comparison that means something: this, versus what the container
        # returns. Not versus actuals -- the artifact was refit on these rows.
        #
        # The dummy rebuild has to be repeated HERE, by hand, because it lives
        # in serve.py and nowhere else. model_io.predict() does not do it, so a
        # naive local baseline sends all four pos_ columns as NaN while the
        # container sends real ones, and the two disagree for a reason that has
        # nothing to do with the container. Caught on the first run of this
        # script. The durable fix is to move this loop into model_io next to
        # align(), so the local path and the serving path cannot drift; until
        # then these six lines and serve.py's must be kept identical.
        scored = rows.copy()
        for col in (f for f in card["features"] if f.startswith("pos_")):
            scored[col] = (scored["position"].astype(str) == col[len("pos_"):]).astype(float)
        local = model_io.predict(scored, models, card, label="batch-check")
        local.insert(0, "player_id", rows["player_id"].values)
        out = BATCH / f"{stem}_local.parquet"
        local.to_parquet(out, index=False)
        print(f"\nlocal predictions -> {out}")
        print(f"  pred_mean: {local['pred_mean'].mean():.2f} avg, "
              f"[{local['pred_mean'].min():.2f}, {local['pred_mean'].max():.2f}]")
        print("  these are RAW. serve.py applies calibration; compare mean_raw, not mean.")


if __name__ == "__main__":
    main()
