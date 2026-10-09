# gcp_lab — one problem, four models, every run tracked

Goal: get first-hand, command-line practice with BigQuery, BigQuery ML, Vertex AI
Workbench, Vertex AI Experiments (+ TensorBoard) and AutoML. Born from a missed PMLE
practice question (Q19): per-step loss -> `log_time_series_metrics` + TensorBoard,
final numbers -> `log_metrics`, settings -> `log_params`, preprocessed data ->
an *artifact* attached with `assign_input_artifacts` (never a metric).

**Problem:** predict `dk_points` from `model_table_v3` (51,356 rows, 2016-2026).
**Scoreboard:** within-week Spearman (same as `src/train.py`), dev = 2021-24, final = 2025.
**Baselines to beat:** production XGBoost 0.644 dev / 0.644 final; FantasyData 0.671 / 0.676.
Split rule (never random, by season): train 2016-2023, validate 2024, final exam 2025.
2026 stays out entirely (live season). Note this differs from production's walk-forward,
so lab scores are not directly comparable to 0.644 until `06_compare.py` re-scores them.

| Stage | Tool | Files | Status |
|---|---|---|---|
| 0 | Local: bucket, dataset, upload | `00_setup.sh` | todo |
| 1 | Parquet -> BigQuery (`bq load`, in Workbench) | `01_bq_load.md` | todo |
| 2 | SQL features + split view, BQML baseline | `sql/*.sql` | todo |
| 3 | Workbench instance (CLI-created, local) | `03_workbench.sh` | todo |
| 4 | Experiments: XGBoost + Keras runs, tracked | `04_experiments.py` | todo |
| 5 | AutoML tabular | `05_automl.py` | todo |
| 6 | Compare all in one table, write up | `06_compare.py` | todo |

Stages 0 and 3 run locally; stages 1, 2, 4, 5, 6 run inside Workbench. Notebooks live in `notebooks/`.

Log every command you run, with what it printed and what surprised you, in `COMMANDS.md`.

## Known snags (checked 2026-09-25)
- `bq` on this machine is broken: it looks for `python3.14`. `00_setup.sh` works around it with CLOUDSDK_PYTHON.
- APIs already enabled on `fantasy-football-ml-503718`: aiplatform, bigquery, notebooks, storage.
- No Workbench instance and no BigQuery datasets exist yet.
- Cost: Workbench bills while running. Stop it after each session (`03_workbench.sh` prints the command).
