# Scheduled agent reports — off the laptop (built 2026-10-09)

Both agents (gameday, Sleeper dynasty) were always deployed on Agent Engine.
What lived on the laptop was the **alarm clock** (Windows Task Scheduler) and
the **thin client** that calls the agent and posts to Discord. That is where
both real failures happened:

| Date | What the log showed | Root cause |
|---|---|---|
| Tue 9/29 | Dynasty memo ran at 15:55, not 12:30 | Laptop asleep; task ran on wake |
| Tue 10/6 | `ImportError: DLL load failed ... An Application Control policy has blocked this file` | Windows Smart App Control blocked a pandas DLL. The job died before it ever called the agent. Intermittent: pandas imports fine now, and the Code Integrity log had already rolled past 10/6 |

Neither failure had anything to do with the agent. Both were laptop problems.

## What runs now

```
Cloud Scheduler ──► Cloud Run job ──► deployed agent (Agent Engine) ──► Discord
 (report-scheduler)   (agent-reports)                                    (webhook from
                                                                          Secret Manager)
        any failed execution ──► Cloud Monitoring alert ──► email
```

All in project `advisor-agent-2026`, region `us-central1`.

| Piece | Name | Notes |
|---|---|---|
| Image | `us-central1-docker.pkg.dev/advisor-agent-2026/agent-reports/reports` | `deploy/Dockerfile.reports`; one image, both jobs; no ADK inside |
| Build | `deploy/cloudbuild.reports.yaml` | runs as `report-builder@` (logWriter, artifactregistry.writer, read on the cloudbuild bucket) |
| Jobs | `gameday-report`, `sleeper-report` | 1 retry each; 600s / 900s timeout |
| Schedules | `gameday-report-schedule` `30 12 * * *`, `sleeper-report-schedule` `30 12 * * 2` | America/New_York, same times the laptop used |
| Secret | `discord-webhook` | mounted as `DISCORD_WEBHOOK_URL` |
| Alert | "Scheduled agent report FAILED" | `run.googleapis.com/job/completed_execution_count`, `result=failed` > 0 |

## Three identities, three jobs

- `report-builder@`: may build and push the image. Nothing else.
- `report-scheduler@`: may **start** the two jobs (`run.invoker` on each job, not project-wide).
- `agent-reports@`: what the job runs as. May call the agents (`aiplatform.user`) and read **one** secret.

Splitting them is the point. If the job's identity leaks, it cannot rebuild
the image or schedule anything; if the scheduler's leaks, it can only start
two jobs that already exist.

## Gotchas hit while building it

1. **Cloud Build's default identity on a new project has no permissions.**
   `gcloud builds submit` failed with `compute@developer... does not have
   storage.objects.get`. Fixed with a dedicated build account
   (`--service-account`), not by widening the default one.
2. **`gcloud builds submit` honours `.gitignore`.** All three `.env` files were
   skipped (confirmed in the gcloud upload log), so no secrets went to the build.
3. **A freshly enabled Cloud Scheduler was slow.** The first two manual
   `scheduler jobs run` calls returned success and did nothing for minutes;
   the third went through. Not a permissions problem.
4. **A missing webhook used to exit 0.** `send_discord()` prints and returns
   False when no URL is set, which on a schedule is a silent non-delivery.
   Both clients now exit 1 in that case, so the alert sees it.

## Rebuild / redeploy the client

```bash
P=advisor-agent-2026; R=us-central1
IMG=$R-docker.pkg.dev/$P/agent-reports/reports:$(git rev-parse --short HEAD)
gcloud builds submit . --project $P --region $R --config deploy/cloudbuild.reports.yaml \
  --service-account=projects/$P/serviceAccounts/report-builder@$P.iam.gserviceaccount.com \
  --substitutions=_IMAGE=$IMG
gcloud run jobs update gameday-report --image $IMG --project $P --region $R
gcloud run jobs update sleeper-report --image $IMG --project $P --region $R
```

Test without posting: `gcloud run jobs execute sleeper-report --args=sleeper_agent/run_report.py,--dry-run --wait`

## Still on the laptop (Phase 2)

The gameday agent's `WEEK` and `MY_LINEUP` are **baked into its deployment**.
Changing either one means a redeploy from the laptop, and they had not been
changed since 9/20. The deployed agent still says `WEEK = 2`, so every daily
report since then checked a stale lineup and quoted Week 2 Vegas totals, and
every run exited 0. Plan: work out the week from the DK slate date, read the
lineup from a Google Sheet in the client, and send it with the request.

Windows tasks "NFL DFS Gameday Report" and "Sleeper Dynasty Memo" are
**disabled, not deleted**, so rolling back is a single click.
