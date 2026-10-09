# Command log
Format: command / what it does in plain English / what it printed / what surprised me.

## 2026-09-25
- `bq load --source_format=PARQUET --replace ...` (Workbench terminal): first try failed with `bigquery.jobs.create` denied — the instance's service account had no BigQuery roles. Fixed by granting jobUser, dataEditor, aiplatform.user (project) and objectAdmin (bucket only). Loaded 51,356 rows, 2016-2026.
- Notebook 04, section 3: `pd.read_gbq` gone in new pandas -> use bigquery.Client. Then `to_dataframe()` failed with bigquery.readsessions.create denied (Storage Read API needs its own role) -> `create_bqstorage_client=False`, fine for ~45k rows.
- Notebook 04 XGBoost: "sklearn needs to be installed" — `!pip` installed to a different Python than the kernel and xgboost checks sklearn at import time. Fix: `%pip install` + restart kernel. Also make TensorBoard create idempotent (list first).
