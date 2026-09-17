"""Read a batch prediction job's output back, and check it against the laptop.

WHAT THE OUTPUT ACTUALLY LOOKS LIKE, WHICH IS NOT WHAT YOU ASKED FOR

You gave the job an outputUriPrefix. It does not write there. It creates a
TIMESTAMPED SUBDIRECTORY under it and shards the results across files named
`prediction.results-00000-of-0000N`. So the path in the job body is a parent, the
path in the job's outputInfo.gcsOutputDirectory is the real one, and neither is a
file. This script takes either and searches downward.

Two file kinds live in that directory and only one is the answer:

    prediction.results-*        one line per instance that SUCCEEDED
    prediction.errors_stats-*   the ones that did not

A batch job reports JOB_STATE_SUCCEEDED when the JOB ran, not when every row
scored. Rows can fail individually and land in the errors file while the job
reports success and the results file comes back short. Reading only the results
file and counting the lines you got -- rather than comparing them to the lines
you sent -- is how a quietly truncated slate becomes a lineup.

THE SHAPE OF A RESULT LINE

    {"instance": {...everything you sent...}, "prediction": {...what came back...}}

The instance is echoed in full, which is why export_batch_input.py sends
player_id and player_display_name along for the ride: align() drops them before
the model sees anything, and they come home as the join key. Without them these
are anonymous numbers in whatever order the replicas finished -- and the order is
NOT guaranteed to match the input.

WHAT IS BEING CHECKED, AND WHAT IS NOT

Against the local baseline: mean_raw should match pred_mean to floating point.
That is a serving-skew test -- same artifact, same rows, different machine. It is
the serving half of the 0.6419-vs-0.6423 check from 4.1.

NOT against actuals. The artifact was refit on all 51,356 rows, this week
included, so scoring it against what happened would measure memorisation. The
actuals are joined for eyeballing only and the script says so.

Run:
    python src/read_batch_output.py --output-dir gs://BUCKET/batch/output/RUN
    python src/read_batch_output.py --output-dir gs://... --season 2025 --week 17
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import model_io  # noqa: E402

BATCH = ROOT / "data" / "batch"


def read_shards(uri: str) -> tuple[pd.DataFrame, list[str]]:
    """Pull every results shard under `uri`. Returns (frame, error_blob_names)."""
    from google.cloud import storage

    bucket_name, prefix = model_io._split_uri(uri)
    client = storage.Client()
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    if not blobs:
        raise SystemExit(
            f"nothing under {uri}\n"
            "  The job body's outputUriPrefix is a PARENT. The real directory is in\n"
            "  the job's outputInfo.gcsOutputDirectory -- poll the job and use that."
        )

    results = [b for b in blobs if "prediction.results" in b.name]

    # The errors file is written whether or not anything failed, so its EXISTENCE
    # says nothing -- only its contents do. Warning on existence would fire on
    # every clean run, and a warning that always fires is one you stop reading.
    errors = []
    for b in blobs:
        if "errors" in b.name:
            body = b.download_as_text().strip()
            if body:
                errors.append((b.name, body))
    if not results:
        raise SystemExit(f"found {len(blobs)} object(s) under {uri} but no prediction.results-* shard:\n"
                         + "\n".join("    " + b.name for b in blobs[:20]))

    rows = []
    for b in sorted(results, key=lambda b: b.name):
        for line in b.download_as_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    print(f"{len(rows)} result line(s) from {len(results)} shard(s) under {uri}")
    return pd.DataFrame(rows), errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True, help="gs:// prefix or the job's gcsOutputDirectory")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--week", type=int, default=17)
    a = ap.parse_args()

    raw, errors = read_shards(a.output_dir)
    if errors:
        print("")
        print(f"  !! {len(errors)} error file(s) with CONTENT -- rows failed individually")
        print( "     while the JOB still reported success. That is the quiet way a slate")
        print( "     truncates: the job is green and the results file is short.")
        for name, body in errors:
            print(f"       {name}")
            for line in body.splitlines()[:5]:
                print(f"         {line}")
    else:
        print("  errors file empty -- nothing failed per-row. It is written either way,")
        print("  so this line is the check; its absence would prove nothing.")

    # Flatten {"instance": {...}, "prediction": {...}} into one row per player.
    inst = pd.json_normalize(raw["instance"])
    pred = pd.json_normalize(raw["prediction"])
    df = pd.concat([inst[["player_id", "player_display_name", "position"]], pred], axis=1)

    stem = f"{a.season}_wk{a.week:02d}"
    sent = sum(1 for _ in open(BATCH / f"{stem}_input.jsonl", encoding="utf-8"))
    print(f"\nsent {sent} instances, got {len(df)} predictions back  "
          f"-- {'MATCH' if sent == len(df) else 'SHORT BY ' + str(sent - len(df))}")

    print("\nfeature coverage as the container saw it:")
    print(f"  features_present  min {int(df['features_present'].min())} / "
          f"{int(df['features_expected'].iloc[0])}, "
          f"median {int(df['features_present'].median())}, "
          f"{int((df['features_present'] == df['features_expected']).sum())} row(s) complete")
    print(f"  calibrated on {int(df['calibrated'].sum())} of {len(df)} row(s)")
    if not bool(df["calibrated"].all()):
        print("    ^ a row missing salary or position. calibrate.apply needs both.")

    # THE COMPARISON THAT MEANS SOMETHING. Join on player_id, never on row order
    # -- the shards come back in whatever order the replicas finished.
    local_path = BATCH / f"{stem}_local.parquet"
    if local_path.exists():
        local = pd.read_parquet(local_path)
        m = df.merge(local, on="player_id", how="inner", validate="one_to_one")
        d = np.abs(m["mean_raw"].to_numpy(float) - m["pred_mean"].to_numpy(float))
        dc = np.abs(m["ceiling"].to_numpy(float) - m["pred_ceiling"].to_numpy(float))
        print(f"\ncloud container vs local model_io.predict(), {len(m)} rows joined on player_id:")
        print(f"  mean_raw  max abs diff {d.max():.3e}")
        print(f"  ceiling   max abs diff {dc.max():.3e}")
        print("  Same artifact, same rows, different machine. This is the serving")
        print("  half of the 0.6419-vs-0.6423 check. Anything but ~0 is a real finding.")
        if d.max() > 1e-5:
            worst = m.loc[np.argsort(-d)[:5], ["player_display_name", "position",
                                               "mean_raw", "pred_mean", "features_present"]]
            print("\n  worst disagreements:")
            print(worst.to_string(index=False))
    else:
        print(f"\nno local baseline at {local_path} -- rerun export_batch_input.py --check")

    actuals = BATCH / f"{stem}_actuals.parquet"
    if actuals.exists():
        df = df.merge(pd.read_parquet(actuals), on="player_id", how="left")
        print("\ntop 10 by calibrated projection (actuals shown for EYEBALLING ONLY --")
        print("the artifact was refit on these rows, so this measures memory, not skill):")
        cols = ["player_display_name", "position", "mean", "ceiling", "dk_points", "features_present"]
        print(df.sort_values("mean", ascending=False).head(10)[cols].round(2).to_string(index=False))

    out = BATCH / f"{stem}_batch_predictions.parquet"
    df.to_parquet(out, index=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
