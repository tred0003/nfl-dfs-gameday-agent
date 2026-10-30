# Production monitoring — what pages, and why

Built 2026-10-22. Config lives in `deploy/monitoring/` (one JSON file per log
metric and per alert policy), so it is reviewed and recreated from git, not
clicked together in the console.

## The split

**Cloud Monitoring detects and pages.** It runs whether or not anyone is
looking, so every alert lives here. **The eval dashboard explains** — trends
and the per-report drill-down (Control Tower view, next step) read the same
signals. A dashboard tab cannot email anyone when the laptop is shut.

## Service level: SLI, SLO, no SLA

- **SLI** (what is measured): the share of user-days that got the **agent's**
  report. A fallback is delivered, but it is the failure the job's exit code
  hides, so it counts as a miss, as does no report at all.
- **SLO** (the target): **95% over a rolling 28 days**
  (`dashboard.SLO_TARGET`, `SLO_WINDOW_DAYS`). The error budget is the misses
  that still allows: (1 − 0.95) × user-days in the window.
- **No SLA.** An SLA is an agreement with a paying customer, with consequences
  for missing it. Nobody pays for this.

With 4 users, 28 days is ~112 user-days, so the budget is ~5 misses: one bad
Sunday for everyone spends most of it. Small numbers make an SLO jumpy; at real
scale it would be set per tier and paired with a burn-rate alert. Shown on the
dashboard's Tower tab; not yet an alert policy.

## The signal: one structured line per report

`run_report.metric_event()` prints one JSON object per user per run. Cloud Run
parses a stdout line that is a JSON object into `jsonPayload`, so every metric
and alert filters on field names, not on English sentences that change when
someone rewords a log message.

| field | meaning |
|---|---|
| `source` | `agent`, `fallback`, `reused`, `no_lineup`, `failed` |
| `check_failed`, `problems` | report_check found missed facts |
| `missing_core`, `missing_core_n` | core tools the agent report did not call |
| `optional_tools`, `optional_n` | which of the 3 optional tools a fresh agent report called, as one string (`none`, `find_replacements`, `search_news+find_replacements`); null for fallback/reused/failed (added 10/30) |
| `total_cost_usd` | agent + in-tool Gemini calls (null when reused) |
| `elapsed_s` | seconds for this user's report (null when reused) |
| `dry_run` | true for manual dry runs; every alert excludes them |

Why it exists: **a fallback exits 0.** When the agent fails, the client sends
the rule-based report and the job succeeds, so the job-failure alert never
fired for the 10/9 fallback. `source = "fallback"` now pages.

## Alerts

| Policy | Project | Fires on | Normal |
|---|---|---|---|
| Gameday report FELL BACK or FAILED | agent | `source="fallback"` or `"failed"` (widened 10/30: a failed attempt the retry then delivered paged nothing on 10/29) | never |
| Weekly review: agent report quality BELOW FLOOR | agent | `WEEKLY REVIEW BELOW FLOOR` from `gameday-weekly-review` (Mondays) | never |
| Gameday report FAILED its fact-check | agent | `check_failed=true` | never (report posts with a correction) |
| Gameday report SKIPPED core tools | agent | `missing_core_n>0` | never (prefetch) |
| Gameday report cost over $0.04 | agent | `total_cost_usd>0.04` | $0.005-0.015 before 10/22; ~$0.020 after the history tool |
| Gameday report took over 240 s | agent | `elapsed_s>240` | 20-130 s; the client gives up at 300 |
| Cloud Scheduler trigger FAILED | both | scheduler log severity ≥ ERROR | never |
| Scheduled agent report FAILED | agent | any failed Cloud Run job execution (existing) | never |
| ML project job FAILED (project-slate, news-archive) | ML | any failed Cloud Run job execution (existing, renamed) | never |

Log-match alerts notify at most once an hour per policy and auto-close after a
day. All go to the existing email channels.

The scheduler alerts close a gap the job alerts cannot see: if Cloud Scheduler
cannot start a job (the 10/12 `PERMISSION_DENIED`, status 7), there is no
execution, so there is no failed execution to alert on.

## Log-based metrics (for charts, not alerts)

- `gameday/reports` — counter, labels `source`, `check_failed`: share of users
  served by the agent vs fallback, fact-check pass rate.
- `gameday/report_cost_usd` — distribution of cost per agent report.
- `gameday/report_elapsed_s` — distribution of time per report.
- `gameday/optional_tools` — counter of fresh agent reports, label
  `optional_tools`. Charted on the Cloud Monitoring dashboard **Gameday agent:
  optional tool use** (`dashboard-agent-tools.json`).

## Optional tools: a rate, not an alert (added 2026-10-30)

8 of the agent's 11 tools are core: prefetch calls them, and skipping one
pages. The other 3 (`find_replacements`, `search_news`,
`search_news_archive`) are the agent's call -- the prompt says "may call" and
only under conditions. So there is no right number and no alert. Zero is the
correct answer on a quiet lineup.

Before this was logged, Firestore showed 59 fresh agent reports (10/16-10/30):
`find_replacements` 2 (3%), `search_news` 0, `search_news_archive` 0.

Two places show it:
- **Daily:** the dashboard chart above, from the report log line.
- **Weekly:** the Monday review prints `Optional tools: find_replacements 1/7 ·
  ...` and up to three **Maybe should have called** lines: a judged report that
  scored 2 or under on `recommendations` without `find_replacements`, or on
  `grounded` without either search. That pairing is the useful signal -- a
  rate alone cannot tell "never needed" from "never bothers."

## Verified 2026-10-22

- Dry run of `gameday-report` on image `reports:23fcf57` for u2: the line
  arrived as `jsonPayload` (agent, check passed, 0 missing core, $0.020377,
  44.6 s, dry_run true).
- Alert path: one synthetic fallback event (uid `alert-test`) written to the
  report job's log to prove a page reaches email. It is the one `alert-test`
  fallback in `gameday/reports`; ignore it in charts.

## Not covered (known)

- **"The report never ran at all."** If the scheduler is deleted or paused,
  nothing errors. A metric-absence alert cannot express "one run a day" (the
  absence window tops out below 24 h). The Thursday canary is the backstop.
- **Report correctness beyond the five rules.** That is the LLM judge's job in
  the weekly review, not a real-time alert.
- **Outcome checks** (did the warnings come true) — weekly review, not paging.

## Recreate

```bash
cd deploy/monitoring
gcloud logging metrics create gameday/reports --project advisor-agent-2026 --config-from-file metric-gameday-reports.json
gcloud logging metrics create gameday/report_cost_usd --project advisor-agent-2026 --config-from-file metric-gameday-report-cost.json
gcloud logging metrics create gameday/report_elapsed_s --project advisor-agent-2026 --config-from-file metric-gameday-report-elapsed.json
gcloud logging metrics create gameday/optional_tools --project advisor-agent-2026 --config-from-file metric-gameday-optional-tools.json
gcloud monitoring dashboards create --project advisor-agent-2026 --config-from-file dashboard-agent-tools.json
for p in report-fallback report-check-failed report-missing-core report-cost report-slow scheduler-error-agent; do
  gcloud alpha monitoring policies create --project advisor-agent-2026 --policy-from-file $p.json; done
gcloud alpha monitoring policies create --project fantasy-football-ml-503718 --policy-from-file scheduler-error-ml.json
```
