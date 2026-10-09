# Stage 1 (run in the Workbench terminal): parquet -> BigQuery

    bq load --source_format=PARQUET --replace \
      nfl_lab.model_table_v3 gs://fantasy-football-ml-503718-gcp-lab/data/model_table_v3.parquet
    bq show --format=prettyjson nfl_lab.model_table_v3 | head -40
    bq query --use_legacy_sql=false 'SELECT season, COUNT(*) n FROM nfl_lab.model_table_v3 GROUP BY season ORDER BY season'

`--replace` overwrites instead of appending (safe to re-run). The upload includes 2026 rows;
`sql/01_split_view.sql` filters to season <= 2025, so the live season stays out of every model.
Parquet keeps the column types, so no schema file is needed.
