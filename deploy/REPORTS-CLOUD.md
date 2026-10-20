# Scheduled agent reports — off the laptop (built 2026-10-09)

> **2026-10-15: the Sleeper dynasty agent was removed** to keep the project on
> the DFS system. Its job, schedule and Agent Engine deployment are deleted;
> the code is at git tag `archive/sleeper-agent`. Sleeper mentions below that
> describe 9/29–10/9 are history, kept as written.

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
| Image | `us-central1-docker.pkg.dev/advisor-agent-2026/agent-reports/reports` | `deploy/Dockerfile.reports`; one image, every job; no ADK inside |
| Build | `deploy/cloudbuild.reports.yaml` | runs as `report-builder@` (logWriter, artifactregistry.writer, read on the cloudbuild bucket) |
| Jobs | `gameday-report`, `gameday-canary` | 1 retry each; 600s / 600s timeout |
| Schedules | `gameday-report-schedule` `30 12 * * *`, `gameday-canary-schedule` `0 10 * * 4` | America/New_York; the report keeps the laptop's time |
| Secret | `discord-webhook` | mounted as `DISCORD_WEBHOOK_URL` |
| Alert | "Scheduled agent report FAILED" | `run.googleapis.com/job/completed_execution_count`, `result=failed` > 0 |

## Three identities, three jobs

- `report-builder@`: may build and push the image. Nothing else.
- `report-scheduler@`: may **start** the jobs (`run.invoker` on each job, not project-wide).
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
gcloud run jobs update gameday-canary --image $IMG --project $P --region $R
```

Test without posting: `gcloud run jobs execute gameday-report --args=gameday_agent/run_report.py,--dry-run --wait`

## Phase 2: nothing about the week or the lineup lives in the deployment

**The incident.** The gameday agent's `WEEK` and `MY_LINEUP` were baked into
its deployment: `adk deploy agent_engine` forwards every line of
`gameday_agent/.env` as a deployment env var. Changing either meant a
redeploy from the laptop, and neither had changed since 9/20. On 10/9 the
deployed agent still said `WEEK = 2`, so every daily report for three weeks
checked a stale lineup and quoted Week 2 Vegas totals, and every run exited 0.
Logs, metrics and traces all looked healthy. It was confidently wrong.

**The design now.**

```
Google Sheet "DFS Lineup" ──read (Viewer)──► Cloud Run client ──create_session(state={"my_lineup": [...]})──► deployed agent
                                                     │                                                         │
                                       unreadable/empty → exit 1 → alert              lineup tools read state first
                                                                                       week derived from the slate date
```

| Piece | Where | Behaviour |
|---|---|---|
| Week | `src/nfl_week.py` | The nflverse schedule week whose games span the slate's first kickoff, in **ET**. Between weeks (Tue/Wed) it's the *next* week. `WEEK` is an override only, used for replays and the eval harness |
| Lineup source | `src/lineup_sheet.py` | Sheet `<SHEET_ID>`, column B, through ADC (the job's SA is a Viewer on the sheet) |
| Lineup delivery | `gameday_agent/run_report.py` | A new session per run with `state={"my_lineup": [...]}`, deleted afterwards |
| Lineup read | `gameday_agent/slate.py` `my_lineup_names(tool_context)` | Session state first, then `MY_LINEUP` env (only for `adk web`) |
| Fallback report | `src/gameday_report.py` `build_report(names=...)` | Checks the same sheet lineup, and derives the week itself |

**Tradeoff, chosen on purpose:** if the sheet can't be read or is empty, the
client sends **no report** and exits 1, so the alert fires. The rule-based
fallback covers "the agent broke" but still checks the right players. Nothing
can cover "we don't know the players", and a stale roster reads exactly like
a real one ("all 9 active"). A missing report gets noticed; a wrong one went
unnoticed for three weeks.

**Also fixed in passing:** `vegas_for_slate()` now treats a local schedule
whose rows have no lines as missing, and falls back to live data. The 9/17
parquet had every Week 5 game and no Week 5 totals.

### Verified (2026-10-09, before deploy)

- `nfl_week`: 10/9 gives 5, Mon 10/12 gives 5, Tue 10/13 gives 6, a Sunday-night
  kickoff stored as Monday UTC gives 5, `WEEK=3` overrides, and it works without
  the parquet through the nflverse CSV.
- Sheet read as `agent-reports@` (impersonated) returns the 9 players. An empty
  ID or a bad ID raises a readable `LineupError`. The laptop's own ADC gets 403
  "insufficient scopes", which is expected (it has no Sheets scope).
- `run_report.py --local --dry-run` with `MY_LINEUP` and `WEEK` blanked: the
  lineup came only through session state, and the agent reported on the sheet's
  players.
- Eval harness `--no-judge`: 6/6 scenarios PASS, now passing the lineup through
  session state with `MY_LINEUP` removed. Run one scenario at a time because of
  Gemini 429s.
- Deployed agent: `create_session(state=...)` returns a dict with `id`, the state
  survives `get_session`, and `delete_session` works.

### Deploy steps still to run (in order)

1. **Remove `MY_LINEUP` and `WEEK` from `gameday_agent/.env`.** Delete or
   comment them out. Don't leave them blank: an empty value is a 400 at deploy.
   While they're present they get baked in again, and a request without state
   would silently fall back to the stale roster.
2. **Redeploy the agent in place**, keeping the same resource ID so
   `AGENT_ENGINE_ID` doesn't change:
   ```
   adk deploy agent_engine --project advisor-agent-2026 --region us-central1 `
     --agent_engine_id 3925664979730038784 --otel_to_cloud --extra_packages src gameday_agent
   ```
   `--otel_to_cloud` (added 2026-10-09) keeps telemetry on. Leave it off and the
   redeploy silently drops it: the request/response content in the agent's logs
   goes away. The content-capture settings live in `gameday_agent/.env`.
3. **Rebuild the client image** (it now carries `nfl_week.py` and `lineup_sheet.py`):
   ```bash
   P=advisor-agent-2026; R=us-central1
   IMG=$R-docker.pkg.dev/$P/agent-reports/reports:$(git rev-parse --short HEAD)
   gcloud builds submit . --project $P --region $R --config deploy/cloudbuild.reports.yaml \
     --service-account=projects/$P/serviceAccounts/report-builder@$P.iam.gserviceaccount.com \
     --substitutions=_IMAGE=$IMG
   ```
4. **Point the job at the new image and the sheet, and drop the baked values:**
   ```bash
   gcloud run jobs update gameday-report --project $P --region $R --image $IMG \
     --update-env-vars LINEUP_SHEET_ID=<SHEET_ID> \
     --remove-env-vars WEEK,MY_LINEUP
   gcloud run jobs update sleeper-report --project $P --region $R --image $IMG
   ```
5. **Dry-run in the cloud** with `--args=gameday_agent/run_report.py,--dry-run,--no-fallback`,
   and check the log shows `lineup from sheet: 9 players` and Week 5 Vegas totals.
6. Optional cleanup: remove the `iam.serviceAccountTokenCreator` grant your
   user has on `agent-reports@`. It was added only for testing.

Windows tasks "NFL DFS Gameday Report" and "Sleeper Dynasty Memo" are
**disabled, not deleted**, so rolling back is a single click.

## Phase 2 — DEPLOYED 2026-10-09 (commits ec77d3d, 31f8db8)

Live: agent redeployed in place (same resource ID), job image `reports:31f8db8`,
gameday job has `LINEUP_SHEET_ID` and no `WEEK`/`MY_LINEUP`. Cloud dry run:
`lineup from sheet: 9 players`, derived week 5, and all six implied totals in the
report match the Week 5 lines exactly (SF 21.25 — the old baked Week 2 value was 29.0).

The first Phase 2 dry run exposed two older bugs, both "wrong report, exit 0":
1. **Last text wins.** The model writes sections between tool calls; the client
   kept only the final one and sent a 397-character Vegas-only report. Clients now
   join every non-partial text part, and the agent is told to write once, at the end.
2. **A model 429 is an event, not an exception.** `error_code: _ResourceExhaustedError`
   arrived mid-run with no content and was skipped, so a half-written report would
   have gone out. Clients now raise on error events (gameday falls back, dynasty fails
   visibly and alerts).

## Thursday canary (added 2026-10-09)

`gameday-canary` runs `gameday_agent/canary.py` every Thursday 10:00 ET, three
days before the Sunday report matters. Same image, identity and secret as
`gameday-report`. It asks the deployed agent to call all six tools and fails
unless each returned `status: "ok"` with rows where rows are expected -- no
fallback, because the daily report's fallback is exactly what would hide a
broken agent. It posts a ✅/❌ verdict to Discord; a failed execution trips the
existing alert (its filter covers every Cloud Run job in the project).

First scheduled run (manual trigger, 10/9 18:54 UTC): attempt 1 hit a Gemini
**429**, posted ❌ and exited 1; the automatic retry passed 6/6 two minutes
later, so the execution succeeded and no alert fired. Same 429 as the 16:34
daily run -- the canary's first finding is that quota, not tools, is the
flakiest dependency.

Test without posting: `gcloud run jobs execute gameday-canary --args=gameday_agent/canary.py,--dry-run --wait`

## Phase 3: one report per user, from Firestore (multi-user step 1, 2026-10-16)

The single lineup Sheet is replaced by Firestore (`(default)` database,
Native mode, `us-central1`, created 10/16). Four users, all the same person
through Gmail plus-addressing, so the fan-out has more than one user:

| uid | Lineup | Discord |
|---|---|---|
| `u1` | the real one, copied once from the old Sheet | yes |
| `u2`-`u4` | the optimizer's top 3 lineups for Week 6 (`deploy/seed_users.py`) | no |

```
Cloud Scheduler ──► gameday-report ──► for each active user, one at a time:
                                         users/{uid}.lineup ──► agent session (user_id = uid)
                                         report ──► users/{uid}/reports/{YYYY-MM-DD}
                                                └─► Discord, only if discord: true
```

- `src/user_store.py`: `FirestoreStore` (real) and `MemoryStore` (tests), one
  interface. Every call takes a uid and builds the path from it.
- `gameday_agent/run_report.py`: `--user u2` runs one user, `--no-discord`
  stores but posts nothing. One user failing doesn't stop the others; the job
  exits 1 at the end, so the alert still fires.
- `gameday_agent/canary.py`: reads `users/{CANARY_UID}` (default `u1`).
- `tests/test_fanout.py`: 12 offline tests of that contract, in CI.
- IAM: `agent-reports@` got `roles/datastore.user`.

Verified 10/16: `run_report.py --user u2 --no-discord` against the deployed
agent called all 7 tools, passed the report check, and saved
`users/u2/reports/2026-10-16`. All 9 of u2's players are in it; none of u1's.
One user took 107s, so **four users need more than the job's 600s timeout**.

### Still to run (not done yet)

```bash
P=advisor-agent-2026; R=us-central1
IMG=$R-docker.pkg.dev/$P/agent-reports/reports:$(git rev-parse --short HEAD)
gcloud builds submit . --project $P --region $R --config deploy/cloudbuild.reports.yaml \
  --service-account=projects/$P/serviceAccounts/report-builder@$P.iam.gserviceaccount.com \
  --substitutions=_IMAGE=$IMG
gcloud run jobs update gameday-report --project $P --region $R --image $IMG \
  --task-timeout 1800s --remove-env-vars LINEUP_SHEET_ID
gcloud run jobs update gameday-canary --project $P --region $R --image $IMG \
  --remove-env-vars LINEUP_SHEET_ID
gcloud run jobs execute gameday-report --project $P --region $R --wait \
  --args=gameday_agent/run_report.py,--dry-run
```

`src/lineup_sheet.py` stays in the repo only because `seed_users.py` reads
u1 from it; nothing in the image uses it.

## Weekly review (added 2026-10-20) -- NOT DEPLOYED YET

`gameday-weekly-review` runs `gameday_agent/weekly_review.py` Wednesdays 08:00
ET, on the week that ended Monday night. It is the only thing that goes back to
the reports actually sent: the LLM judge grades each distinct agent report of
the week against its stored tool results, nflverse snap counts score the Sunday
report's OUT/Q/fine calls against who played, and the week's numbers go to
`eval_weekly/{season}-w{NN}` and are compared with up to four weeks before. A
regression exits 1, so the existing "Scheduled agent report FAILED" alert
fires; the summary posts to Discord either way. Rules and thresholds are in the
module docstring.

Wednesday, not Tuesday: nflverse snap counts for Monday night are not reliably
published by Tuesday morning. Until every team's data is in, a player without a
snap row is "not known yet", never "sat".

Own image (`deploy/Dockerfile.review`), because the judge imports ADK and the
reports image deliberately has none. Same identity (`agent-reports@`: Vertex
for the judge, `datastore.user` for Firestore) and the same webhook secret. **No
retries**: a retry re-runs the judge (~$0.03 per report) and posts twice.

Cost: one week of 4 users x 4-7 days was 16 judge calls, ~$0.55, 7 minutes.
Reports reused across users with the same lineup are judged once.

First run measured week 6 (dry, not saved): quality 52.9. Do not use week 6 as a
baseline -- its reports predate "My take" (10/19), so every one scores 1/5 on
`recommendations`. The first saved week should be week 7 (10/20-10/26).

```bash
P=advisor-agent-2026; R=us-central1
IMG=$R-docker.pkg.dev/$P/agent-reports/review:$(git rev-parse --short HEAD)
gcloud builds submit . --project $P --region $R --config deploy/cloudbuild.review.yaml \
  --service-account=projects/$P/serviceAccounts/report-builder@$P.iam.gserviceaccount.com \
  --substitutions=_IMAGE=$IMG
gcloud run jobs create gameday-weekly-review --project $P --region $R --image $IMG \
  --service-account agent-reports@$P.iam.gserviceaccount.com \
  --set-secrets DISCORD_WEBHOOK_URL=discord-webhook:latest \
  --set-env-vars GOOGLE_CLOUD_PROJECT=$P,GOOGLE_CLOUD_LOCATION=$R,GOOGLE_GENAI_USE_VERTEXAI=TRUE \
  --max-retries 0 --task-timeout 1800s --memory 2Gi
gcloud run jobs add-iam-policy-binding gameday-weekly-review --project $P --region $R \
  --member serviceAccount:report-scheduler@$P.iam.gserviceaccount.com --role roles/run.invoker
gcloud scheduler jobs create http gameday-weekly-review-schedule --project $P --location $R \
  --schedule "0 8 * * 3" --time-zone America/New_York --http-method POST \
  --uri https://run.googleapis.com/v2/projects/$P/locations/$R/jobs/gameday-weekly-review:run \
  --oauth-service-account-email report-scheduler@$P.iam.gserviceaccount.com
# test without saving or posting
gcloud run jobs execute gameday-weekly-review --project $P --region $R --wait \
  --args=gameday_agent/weekly_review.py,--dry-run,--no-judge
```
