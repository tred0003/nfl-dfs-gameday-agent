# Value picks — model projections in the cloud (built 2026-10-10)

The gameday agent's `get_value_picks` tool lists the top 5 QB/RB/WR/TE by
value and the top 3 DST. **The agent never runs the model.** A scheduled job
in the ML project projects the whole slate and writes a table; the agent reads
it.

```
ML project  fantasy-football-ml-503718                  Agent project  advisor-agent-2026
──────────────────────────────────────                  ───────────────────────────────────
Cloud Scheduler  project-slate-fri  Fri 19:00 ET
                 project-slate-sun  Sun 11:00 ET
      │  (project-slate-scheduler@, run.invoker on the job only)
      ▼
Cloud Run job  project-slate   (runs as project-slate@)
  1. pull gs://…-nfl-dfs/raw/ → refresh this season from nflverse → push back
  2. DK main slate → archive to gs://…/slates/SEASON/wkNN/   (salary history)
  3. resolve @champion in the Model Registry at run time
  4. predict_week.build_pool()   ← same function as the laptop CLI
  5. append rows to BigQuery  dfs.projections  ───────────►  get_value_picks()
                                                  READER on dfs + jobUser,
                                                  granted to the Agent Engine
                                                  service agent — the ONE
                                                  permission at the boundary
any failed execution ──► alert "Projection job FAILED" ──► email
```

| Piece | Name |
|---|---|
| Code | `src/project_slate.py`, `deploy/Dockerfile.project`, `deploy/cloudbuild.project.yaml`, `deploy/requirements-project.txt` |
| Image | `us-central1-docker.pkg.dev/fantasy-football-ml-503718/nfl-dfs/project-slate:<sha>` |
| Job | `project-slate`: 2 CPU, 4 GiB, 1800 s, 1 retry. A run takes ~8 min (feature build ~7) |
| Table | `fantasy-football-ml-503718.dfs.projections`, append-only, partitioned on `run_ts`, one `run_id` per run |
| Identities | `project-slate@` (objectAdmin on the bucket, dataEditor on `dfs`, jobUser, aiplatform.viewer) · `project-slate-scheduler@` (invoker on the job) |

## Design choices worth defending

- **The model is a file, not a fit.** The job scores the registered artifact
  that `@champion` points at; promoting a model is moving the alias. Nothing
  retrains on a schedule.
- **Parity by construction, then verified.** `build_pool()` is shared with
  `predict_week.py`. A dry run on the same slate snapshot matched the CLI on
  all 587 rows to 0.00e+00.
- **Append, never replace.** What was projected before kickoff is kept, so the
  projections can be graded against actual points later.
- **DST has no model.** DK FPPG × Vegas implied total. Every DST row says
  `method = dst_rule_dk_fppg_x_vegas`, and the tool ranks them by points per $1k.
- **Three tool states.** `ok`, `not_ready` (no run for this slate yet — normal
  before Friday; the Thursday canary accepts it), `error` (canary fails).

## Things that bit while building it — read before touching it

1. **Missing salary weeks do not look missing.** Salary features shift by ROW,
   not by week: with 2026 weeks 3–5 absent, `salary_chg_1w` silently became a
   week-2→now change. `salary_chg_3w` going 100% NaN is what preflight caught.
   Weeks 1/3/4/5 were downloaded by hand from FantasyData; from week 5 on the
   job archives each slate to `gs://…/slates/` and feeds it back as history.
2. **Derive the week; never type it.** A hand-passed `--week 6` scored this
   week's slate as next week's. The job derives the week from the slate kickoff.
3. **Push data BEFORE running.** The first cloud runs failed preflight because
   the salary CSVs reached the laptop after the GCS upload.
   `gcloud storage rsync data/raw/fantasydata gs://fantasy-football-ml-503718-nfl-dfs/raw/fantasydata --recursive`
4. **`depth_rank` is a percentile where 1.0 means LAST.** The tool shows the
   model "top of depth chart" / "behind N teammate(s)" instead. The first live
   run's #1 QB value was MIN's backup.
5. **nflverse injury lag.** If nflverse has not posted the week, preflight
   accepts the DK slate's OUT/IR/D designations (which `build_rows` already
   uses as the week's report). Zero ruled out still fails.
6. **The depth-chart refresh is excluded** from the job's nflverse update:
   2026 depth charts are a daily format with no `week` column, which crashes
   `fetch_nflverse.update`. The features read the daily format fine; the
   `depth` table in raw/ is just not refreshed in-season. Known gap.

## Run it by hand

```bash
P=fantasy-football-ml-503718
gcloud run jobs execute project-slate --project $P --region us-central1 --wait
gcloud run jobs execute project-slate --project $P --region us-central1 --args=--dry-run --wait   # no writes
```
