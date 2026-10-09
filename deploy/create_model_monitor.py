"""One-time setup: register the top-level ModelMonitor resource for 4.5.

WHY THIS API AND NOT `gcloud ai model-monitoring-jobs`

That surface exists in gcloud, but `create` requires `--endpoint` -- it
monitors traffic hitting a deployed endpoint. 4.3 ended with the endpoint
undeployed on purpose (batch beats an always-on node for ~300 requests a
week), so an endpoint-shaped monitor would mean standing up the exact thing
4.3 tore down just to watch it. `aiplatform_v1beta1.ModelMonitoringService`
is the newer, dataset-shaped API: its `ModelMonitoringInput` has a
`batch_prediction_output` variant that points straight at a Vertex batch
prediction job's resource name. No endpoint, no traffic -- it reads the same
job the weekly run already produces. It has no gcloud surface either, same
situation as batch prediction itself: REST or the Python SDK, this uses the
SDK because the request shape is deeply nested and worth getting from types,
not from hand-written JSON.

TWO RESOURCES, NOT ONE

ModelMonitor is the durable, top-level thing: which model version is being
watched, what the schema looks like, where alerts go. It is created once.
ModelMonitoringJob is one run of the comparison -- created fresh every week
by deploy/run_monitoring_job.py, pointed at that week's batch prediction job.
This script only creates the first one.

THE SCHEMA HAS TO NAME COLUMNS THAT EXIST ON BOTH SIDES

feature_fields here has to match, by name, both the baseline JSONL
(src/export_monitoring_baseline.py) and the batch prediction job's echoed
instances (src/export_batch_input.py) -- same PASSTHROUGH-minus-IDs logic:
`position` is the one categorical column, everything else the model actually
reads is float once align() gets to it (model_io.align() ends in
`.astype("float64")` regardless of the table's storage dtype), and the four
pos_QB/RB/TE/WR dummies are never in either file -- they're rebuilt from
`position` at serve time, so they aren't features Model Monitoring can see.

Run:
    python deploy/create_model_monitor.py --version-id 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from google.cloud.aiplatform_v1beta1 import ModelMonitoringServiceClient
from google.cloud.aiplatform_v1beta1 import types as t

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import model_io  # noqa: E402


def build_schema(card: dict) -> t.ModelMonitoringSchema:
    feats = [f for f in card["features"] if not f.startswith("pos_")]
    fields = [t.ModelMonitoringSchema.FieldSchema(name="position", data_type="categorical")]
    fields += [t.ModelMonitoringSchema.FieldSchema(name=f, data_type="float") for f in feats]
    return t.ModelMonitoringSchema(feature_fields=fields)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="fantasy-football-ml-503718")
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--model-id", default="7035124545010794496")
    ap.add_argument("--version-id", required=True,
                     help="numeric version id, not an alias -- VertexModelSource takes "
                          "model_version_id, which the API resolves from a version number, "
                          "not a movable label. Look it up with "
                          "`gcloud ai models list-version MODEL_ID --region=REGION`.")
    ap.add_argument("--display-name", default="nfl-dfs-champion-monitor")
    ap.add_argument("--baseline-uri",
                     default="gs://fantasy-football-ml-503718-nfl-dfs/monitoring/baseline/model_table_v3_baseline_noroutes.jsonl")
    ap.add_argument("--output-prefix",
                     default="gs://fantasy-football-ml-503718-nfl-dfs/monitoring/results")
    ap.add_argument("--email", default="you@example.com")
    # models/current is the WITH-routes card (83 features) -- champion is the
    # no-routes model (77 features, models/no-routes), chosen deliberately
    # because route-share participation data doesn't exist for the 2026
    # season at all (100% null), so a backtest edge earned on years where it
    # existed doesn't carry over. The schema here has to match whichever
    # artifact --version-id actually points at, not "current".
    ap.add_argument("--model-dir", default=str(ROOT / "models" / "no-routes"))
    a = ap.parse_args()

    _, card = model_io.load(a.model_dir)
    schema = build_schema(card)
    print(f"schema: {len(schema.feature_fields)} feature fields "
          f"(1 categorical + {len(schema.feature_fields) - 1} float)")

    client = ModelMonitoringServiceClient(
        client_options={"api_endpoint": f"{a.region}-aiplatform.googleapis.com"}
    )
    parent = f"projects/{a.project}/locations/{a.region}"

    existing = list(client.list_model_monitors(parent=parent))
    hit = next((m for m in existing if m.display_name == a.display_name), None)
    if hit:
        print(f"already exists: {hit.name}")
        print("delete it first (client.delete_model_monitor) to recreate, or pick a "
              "different --display-name.")
        return

    monitor = t.ModelMonitor(
        display_name=a.display_name,
        model_monitoring_target=t.ModelMonitor.ModelMonitoringTarget(
            vertex_model=t.ModelMonitor.ModelMonitoringTarget.VertexModelSource(
                model=f"projects/{a.project}/locations/{a.region}/models/{a.model_id}",
                model_version_id=a.version_id,
            )
        ),
        training_dataset=t.ModelMonitoringInput(
            columnized_dataset=t.ModelMonitoringInput.ModelMonitoringDataset(
                gcs_source=t.ModelMonitoringInput.ModelMonitoringDataset.ModelMonitoringGcsSource(
                    gcs_uri=a.baseline_uri,
                    format_=t.ModelMonitoringInput.ModelMonitoringDataset.ModelMonitoringGcsSource.DataFormat.JSONL,
                )
            )
        ),
        model_monitoring_schema=schema,
        notification_spec=t.ModelMonitoringNotificationSpec(
            email_config=t.ModelMonitoringNotificationSpec.EmailConfig(user_emails=[a.email]),
            enable_cloud_logging=True,
        ),
        output_spec=t.ModelMonitoringOutputSpec(
            gcs_base_directory=t.GcsDestination(output_uri_prefix=a.output_prefix)
        ),
    )

    print(f"creating ModelMonitor watching model {a.model_id}@{a.version_id} ...")
    # A custom --model-monitor-id looks supported (CreateModelMonitorRequest
    # takes it) and even gets accepted at create time, but the LRO's own
    # polling then breaks: "Since the idType 'modelMonitors' is not a
    # documented LRO Parent ID, the associated value must be a Long, but was
    # instead: nfl-dfs-champion". The operation-status endpoint assumes a
    # numeric resource id and can't look up a named one -- a v1beta1
    # rough edge, not something this project controls. Leaving the id off
    # gets an auto-generated numeric one and the polling works.
    op = client.create_model_monitor(parent=parent, model_monitor=monitor)
    result = op.result()
    print(f"created: {result.name}")
    print(f"\npass this to run_monitoring_job.py:\n  --model-monitor {result.name}")


if __name__ == "__main__":
    main()
