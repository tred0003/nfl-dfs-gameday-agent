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

## Suggested lineups to Discord (live 2026-10-12)

`project_slate.py --post-lineups` posts right after the BigQuery write, so the
post can never race the data it reads. Schedule:

| Scheduler | When (ET) | Posts lineups? |
|---|---|---|
| `project-slate-thu` | Thu 09:00 (before the 10:00 canary) | yes |
| `project-slate-fri` | Fri 19:00 | no |
| `project-slate-sun` | Sun 11:00 ("could change") | yes |

The flag rides in the scheduler body as a Cloud Run v2 override:
`{"overrides":{"containerOverrides":[{"args":["--post-lineups"]}]}}`.

Three things had to be true, and each failed once:

1. **Cross-project secret.** The webhook lives in `advisor-agent-2026`; the job
   runs in the ML project. The SA `project-slate@` got `secretAccessor` on it,
   and the job mounts it as `DISCORD_WEBHOOK_URL`. gcloud rejected the
   `projects/<name>/secrets/...` form ("not a valid secret name"); the
   **project number** form worked: `projects/389876399573/secrets/discord-webhook:latest`.
2. **Overrides need their own role.** `roles/run.invoker` lets a scheduler run a
   job but NOT pass args. First trigger returned PERMISSION_DENIED (scheduler
   status code 7) and the job never started. Fix: grant the scheduler SA
   `roles/run.jobsExecutorWithOverrides` on the job only. Note this broke the
   Sunday schedule too until the grant landed, since both now send a body.
3. **Verified by a live trigger**, not by reading config: execution
   `project-slate-k5x8b` ran with the arg, appended 513 rows, logged
   "suggested lineups posted to Discord", exit 0.

Known gap seen in that run: Monday's slate had no injury report yet ("0
Out/Doubtful -- NOT PUBLISHED YET"), so the Thursday post for a fresh week
leans on DK's own OUT/IR designations. Expected, but worth a caveat line in the
post later.

## Sanity gate (2026-10-12, commits 69e1dab + write gate)

`src/projection_checks.py` sits between the model and everything downstream.
Two gates, same checks:

1. **Before the BigQuery write** (`suggest_lineup.gate_write`, called from
   `project_slate.main`). A run that fails is NOT appended: `dfs.projections`
   also feeds the agent's `get_value_picks`, so a bad run in the table is worse
   than a missing one. Thursday/Sunday (`--post-lineups`) also post a "withheld"
   notice to Discord; Friday is silent and relies on the exit-1 job alert.
2. **Before the Discord post** (`suggest_lineup.post`): pool checks again, then
   per-lineup checks on the solved entries (9 players, legal positions, under the
   cap, no duplicates, nobody ruled out, 60-200 projected pts).

Blocks: <150 projected rows, projection outside 0-50, salary-vs-projection
Spearman < 0.35, mixed week/model_version. Warns (rides under the lineups): a
healthy player at $6,000+ with no projection (the Hollywood Brown name-match miss);
unprojected OUT players are ignored. Calibrated on the real 10/12 run: 368
projected, range 1.7-23.5, Spearman 0.72, lineups ~137 pts, zero findings.

- **Tests:** `python -m unittest discover -s tests` -> 22 offline cases. Each breaks
  one thing in a synthetic healthy slate and asserts which check fires; the real
  optimizer solves the lineups. Run with `-W ignore` to hide pulp deprecation noise.
- **Verified in the cloud:** image `project-slate:69e1dab`, dry run
  `project-slate-vfwpm` (no false alarm on the live Week 6 pool, exit 0).
  The BLOCK path has only been exercised by tests, not by a cloud run;
  Thursday 09:00 is its first real use. The write gate is committed after 69e1dab
  and needs one more image build to go live.
- **Laptop gotcha:** Smart App Control blocks sklearn's `_target_encoder_fast` DLL, so
  `import project_slate` fails locally (same class as the 10/6 pandas DLL). That is
  why `gate_write` lives in `suggest_lineup.py`: it imports without the ML stack.
