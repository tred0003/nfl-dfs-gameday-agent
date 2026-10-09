#!/usr/bin/env bash
# Stage 3: create a Workbench instance from the command line, not the console.
set -euo pipefail
gcloud workbench instances create nfl-lab-wb \
  --project=fantasy-football-ml-503718 --location=us-central1-a \
  --machine-type=e2-standard-4
gcloud workbench instances list --location=us-central1-a
# When done for the day (stops the billing):
#   gcloud workbench instances stop nfl-lab-wb --location=us-central1-a
