"""Poll a ModelMonitoringJob to completion, then print per-feature drift stats.

A ModelMonitoringJob answers "done yet" the same way a batch prediction job
does -- JOB_STATE_SUCCEEDED means the comparison RAN, not that nothing
drifted. The actual verdict is in SearchModelMonitoringStats (every feature's
distance from baseline, whether it crossed the threshold) and
SearchModelMonitoringAlerts (only the features that did). This script prints
both, sorted worst-first, so a healthy week and a drifting one look visibly
different rather than requiring you to read every stats_name by hand.

Run:
    python deploy/read_monitoring_result.py --job projects/.../modelMonitoringJobs/ID
"""

from __future__ import annotations

import argparse
import time

from google.cloud.aiplatform_v1beta1 import ModelMonitoringServiceClient
from google.cloud.aiplatform_v1beta1 import types as t


def model_monitor_of(job_name: str) -> str:
    # projects/P/locations/R/modelMonitors/M/modelMonitoringJobs/J -> the /modelMonitors/M prefix
    head, _, _ = job_name.partition("/modelMonitoringJobs/")
    return head


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="full ModelMonitoringJob resource name")
    ap.add_argument("--region", default="us-central1")
    ap.add_argument("--wait", type=int, default=300, help="seconds to poll before giving up")
    ap.add_argument("--poll-every", type=int, default=15)
    a = ap.parse_args()

    client = ModelMonitoringServiceClient(
        client_options={"api_endpoint": f"{a.region}-aiplatform.googleapis.com"}
    )
    monitor = model_monitor_of(a.job)

    deadline = time.time() + a.wait
    job = client.get_model_monitoring_job(name=a.job)
    while job.state in (t.JobState.JOB_STATE_QUEUED, t.JobState.JOB_STATE_PENDING,
                        t.JobState.JOB_STATE_RUNNING):
        if time.time() > deadline:
            print(f"still {job.state.name} after {a.wait}s, giving up on this poll -- "
                  f"re-run the same command later, the job keeps running server-side")
            return
        print(f"  {job.state.name}, waiting {a.poll_every}s ...")
        time.sleep(a.poll_every)
        job = client.get_model_monitoring_job(name=a.job)

    print(f"final state: {job.state.name}")
    if job.state != t.JobState.JOB_STATE_SUCCEEDED:
        print("not SUCCEEDED -- check job_execution_detail below for why:")
        print(job.job_execution_detail)
        return

    # Both search_* methods' flattened kwargs only expose model_monitor --
    # same gap as create_model_monitor's missing model_monitor_id. The filter
    # and the job-scoping have to go through the full request message.
    stats_resp = client.search_model_monitoring_stats(
        request=t.SearchModelMonitoringStatsRequest(
            model_monitor=monitor,
            stats_filter=t.SearchModelMonitoringStatsFilter(
                tabular_stats_filter=t.SearchModelMonitoringStatsFilter.TabularStatsFilter(
                    objective_type="raw-feature-drift",
                    model_monitoring_job=a.job,
                )
            ),
        )
    )

    # current_stats is a TypedValue oneof documented as double_value OR
    # distribution_value. In practice, on a real run, every single feature
    # comes back distribution_value -- a full TFDV histogram/struct, no
    # scalar anywhere in it. Reading .double_value on one of those doesn't
    # raise; protobuf silently returns the field's zero-value, 0.0,
    # indistinguishable from genuine no-drift. That is exactly what the first
    # version of this script showed: three features flagged ANOMALY while
    # printing 0.0000 right next to sixty-nine features that were actually
    # 0.0000 -- both looked identical. There is no per-feature distance this
    # API will hand back; has_anomaly is the only signal that exists, decided
    # server-side, and that's what this prints.
    rows = []
    for ms in stats_resp.monitoring_stats:
        series = ms.tabular_stats
        for pt in series.data_points:
            which = pt.current_stats._pb.WhichOneof("value") if pt.current_stats else None
            val = pt.current_stats.double_value if which == "double_value" else None
            rows.append((series.stats_name, val, pt.threshold_value, pt.has_anomaly))
    rows.sort(key=lambda r: (not r[3], r[1] is None, -(r[1] or 0)))

    n_anomaly = sum(1 for r in rows if r[3])
    print(f"\n{len(rows)} feature(s) scored, {n_anomaly} anomaly(ies) (threshold {rows[0][2]:.2f} on both metrics)")
    for name, val, threshold, anomaly in rows:
        if not anomaly and val is None:
            continue  # no scalar to show and nothing flagged -- the alert list below covers it
        flag = " <-- ANOMALY" if anomaly else ""
        val_s = f"{val:.4f}" if val is not None else "no scalar exposed by this API"
        print(f"  {name:30s} {val_s:>28s}{flag}")
    if n_anomaly == 0:
        print("  (no per-feature scalar returned for the rest -- has_anomaly=False on all of them)")

    alerts = list(client.search_model_monitoring_alerts(
        request=t.SearchModelMonitoringAlertsRequest(
            model_monitor=monitor, model_monitoring_job=a.job
        )
    ))
    if alerts:
        print(f"\n{len(alerts)} alert(s):")
        for al in alerts:
            print(f"  {al.stats_name} [{al.objective_type}]")
    else:
        print("\nno alerts -- nothing crossed threshold this run")


if __name__ == "__main__":
    main()
