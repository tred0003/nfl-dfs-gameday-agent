"""Per-run step: check one week's batch prediction job for feature drift.

WHAT THIS ACTUALLY COMPARES

baseline_dataset = the whole training table, exported once by
src/export_monitoring_baseline.py. target_dataset = THIS week's batch
prediction job, read straight off its resource name via
`batch_prediction_output` -- Model Monitoring pulls the instances back out of
the job's own output itself, so there's no separate export step for the
target side the way there is for the baseline.

WHY A NEW JOB EVERY WEEK RATHER THAN ONE LONG-RUNNING THING

ModelMonitor (deploy/create_model_monitor.py) is the durable resource: which
model version, what schema, where alerts go. ModelMonitoringJob is one
comparison, same relationship as a training run to a registered model --
immutable once it's run, and next week's drift check is a new job under the
same monitor, not a mutation of this one.

WHAT "FEATURE DRIFT" MEANS HERE

feature_drift_spec with jensen_shannon_divergence (numeric) and l_infinity
(categorical) measures distance between this week's feature distributions and
the baseline's, per feature, and fires the configured alert when a feature
crosses the threshold. It says nothing about whether the PREDICTIONS are
right -- that would be prediction_output_drift_spec or a feature_attribution
spec, deliberately out of scope for this first pass.

Run:
    python deploy/run_monitoring_job.py \\
        --model-monitor projects/.../locations/us-central1/modelMonitors/ID \\
        --batch-job projects/.../locations/us-central1/batchPredictionJobs/ID
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

from google.cloud.aiplatform_v1beta1 import ModelMonitoringServiceClient
from google.cloud.aiplatform_v1beta1 import types as t

# Thresholds are a starting point, not a validated one: JS-divergence is
# bounded [0, 1] (natural log) and 0.3 is a common "worth a look" cutoff in
# the drift-monitoring literature, not a number tuned against this model's
# actual week-to-week noise. The honest fix is watching a few real weeks and
# adjusting -- there's no backtest for "what does normal week-to-week
# variance look like" the way there is for slate_spearman.
DEFAULT_THRESHOLD = 0.3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-monitor", required=True,
                     help="full resource name from create_model_monitor.py's output")
    ap.add_argument("--batch-job", required=True,
                     help="full resource name of a COMPLETED BatchPredictionJob, e.g. "
                          "projects/P/locations/R/batchPredictionJobs/ID")
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--display-name", default=None)
    a = ap.parse_args()

    client = ModelMonitoringServiceClient(
        client_options={"api_endpoint": f"{a.region}-aiplatform.googleapis.com"}
    )

    alert = t.ModelMonitoringAlertCondition(threshold=a.threshold)
    objective_spec = t.ModelMonitoringObjectiveSpec(
        tabular_objective=t.ModelMonitoringObjectiveSpec.TabularObjective(
            feature_drift_spec=t.ModelMonitoringObjectiveSpec.DataDriftSpec(
                categorical_metric_type="l_infinity",
                numeric_metric_type="jensen_shannon_divergence",
                default_categorical_alert_condition=alert,
                default_numeric_alert_condition=alert,
            )
        ),
        # baseline_dataset is left unset here -- it inherits the ModelMonitor's
        # training_dataset (the exported model_table_v3 baseline). Only the
        # target changes week to week.
        target_dataset=t.ModelMonitoringInput(
            batch_prediction_output=t.ModelMonitoringInput.BatchPredictionOutput(
                batch_prediction_job=a.batch_job
            )
        ),
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    job = t.ModelMonitoringJob(
        display_name=a.display_name or f"nfl-dfs-drift-{stamp}",
        model_monitoring_spec=t.ModelMonitoringSpec(objective_spec=objective_spec),
    )

    print(f"checking {a.batch_job}\n  against baseline on {a.model_monitor} "
          f"(threshold {a.threshold} on both metrics) ...")
    result = client.create_model_monitoring_job(parent=a.model_monitor, model_monitoring_job=job)
    print(f"created: {result.name}")
    print(f"state: {result.state.name}")
    print("\npoll with:\n  python deploy/read_monitoring_result.py --job", result.name)


if __name__ == "__main__":
    main()
