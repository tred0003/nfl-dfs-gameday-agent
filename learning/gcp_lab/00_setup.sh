#!/usr/bin/env bash
# Stage 0 (run LOCALLY, once): project, bucket, dataset, upload the parquet.
# Everything after this runs inside the Workbench instance.
set -euo pipefail
# If bq picks the wrong Python, point it at yours:  export CLOUDSDK_PYTHON=/path/to/python
PROJECT=fantasy-football-ml-503718 REGION=us-central1
BUCKET=gs://${PROJECT}-gcp-lab
PARQUET="$(dirname "$0")/../../data/processed/model_table_v3.parquet"

gcloud config set project $PROJECT
gcloud services list --enabled | grep -Ei "aiplatform|bigquery|notebooks|storage"   # confirm APIs
gcloud storage buckets create $BUCKET --location=$REGION --uniform-bucket-level-access
bq --location=$REGION mk --dataset --description "NFL DFS lab" ${PROJECT}:nfl_lab
gcloud storage cp "$PARQUET" $BUCKET/data/model_table_v3.parquet
gcloud storage ls -l $BUCKET/data/
