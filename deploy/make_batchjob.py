"""Fill batchjob.predict.json.template into a submittable BatchPredictionJob body.

WHY THIS EXISTS RATHER THAN A POWERSHELL -replace

The training job uses `(Get-Content template) -replace ... | Set-Content`, and
that works because customjob.train.yaml is YAML -- YAML has comments, so the
explanations live in the file and the file is still submittable.

JSON has no comments. The explanations in the template are therefore _comment_*
KEYS, and a Google REST API rejects unknown fields outright:

    Invalid JSON payload received. Unknown name "_comment_1"

So something has to strip them, and that something cannot be PowerShell 5.1's
ConvertFrom-Json | ConvertTo-Json round trip: ConvertTo-Json defaults to a depth
of 2 and would silently flatten machineSpec and gcsSource into the string
"System.Object[]". The job would submit and fail for a reason with no connection
to what you changed.

It also validates before you spend anything. A batch job costs node-minutes to
discover a typo, and the platform reports the failure asynchronously.

Run:
    python deploy/make_batchjob.py ^
        --model "projects/P/locations/us-central1/models/ID@candidate" ^
        --input "gs://BUCKET/batch/input/2025_wk17_input.jsonl" ^
        --output-prefix "gs://BUCKET/batch/output/RUN" ^
        --display-name "nfl-dfs-batch-RUN"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "batchjob.predict.json.template"
OUT = HERE / "batchjob.predict.json"


def strip_comments(obj):
    """Drop _comment_* keys at every level. They are documentation, not payload."""
    if isinstance(obj, dict):
        return {k: strip_comments(v) for k, v in obj.items() if not k.startswith("_comment")}
    if isinstance(obj, list):
        return [strip_comments(v) for v in obj]
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="full resource path, @alias or @version. Bare IDs are rejected.")
    ap.add_argument("--input", required=True, help="gs:// path to the .jsonl")
    ap.add_argument("--output-prefix", required=True, help="gs:// PREFIX; a subdir is created under it")
    ap.add_argument("--display-name", required=True)
    ap.add_argument("--machine-type", default=None, help="override the template's machineSpec")
    a = ap.parse_args()

    raw = TEMPLATE.read_text(encoding="utf-8")
    for token, value in (("MODEL_REF", a.model), ("INPUT_URI", a.input),
                         ("OUTPUT_PREFIX", a.output_prefix), ("DISPLAY_NAME", a.display_name)):
        if token not in raw:
            raise SystemExit(f"template has no {token} placeholder -- did it already get filled?")
        raw = raw.replace(token, value)

    body = strip_comments(json.loads(raw))
    if a.machine_type:
        body["dedicatedResources"]["machineSpec"]["machineType"] = a.machine_type

    # Cheap checks that each map to a real way this fails asynchronously, twenty
    # minutes and some node-minutes after you stop watching.
    problems = []
    if not a.model.startswith("projects/"):
        problems.append("--model must be the FULL resource path. A bare model ID is rejected, "
                        "the same way --parent-model is on `gcloud ai models upload`.")
    if not a.input.startswith("gs://") or not a.input.endswith(".jsonl"):
        problems.append("--input must be a gs:// path ending in .jsonl to match instancesFormat.")
    if not a.output_prefix.startswith("gs://"):
        problems.append("--output-prefix must be a gs:// path.")
    if a.input.rsplit("/", 1)[0] == a.output_prefix.rstrip("/"):
        problems.append("input and output share a prefix. The job writes a subdirectory under the "
                        "output prefix; pointing both at one place makes the next run's input ambiguous.")
    if "dedicatedResources" not in body:
        problems.append("a custom container needs dedicatedResources -- the platform will not "
                        "choose a machine type for you.")
    if problems:
        raise SystemExit("\n".join("  - " + p for p in problems))

    OUT.write_text(json.dumps(body, indent=2), encoding="utf-8")
    print(f"wrote {OUT}\n")
    print(json.dumps(body, indent=2))
    print(f"\nmodel        {body['model']}")
    print(f"reads        {body['inputConfig']['gcsSource']['uris'][0]}")
    print(f"writes under {body['outputConfig']['gcsDestination']['outputUriPrefix']}")
    print(f"machine      {body['dedicatedResources']['machineSpec']['machineType']} "
          f"x{body['dedicatedResources']['startingReplicaCount']}, no accelerator")


if __name__ == "__main__":
    main()
