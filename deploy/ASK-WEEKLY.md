# Ask's weekly exam — deploy

`webapp/ask_weekly.py`: 50 start/sit questions every Sunday before kickoff,
graded every Tuesday against DraftKings points from nflverse box scores.
The job runs on the **gameday-web image**, as the **web-app service account**,
so the exam asks exactly the Ask code and identity the site serves (same
news-mcp, same model card, same Gemini project).

| Piece | Where |
|---|---|
| Answers (Sunday) | `fantasy-football-ml-503718.dfs.ask_weekly` |
| Grades (Tuesday) | `fantasy-football-ml-503718.dfs.ask_weekly_grades` |
| Jobs | Cloud Run jobs `ask-weekly-ask`, `ask-weekly-grade` (advisor-agent-2026) |
| Schedules | `ask-weekly-ask-schedule` Sun 11:15 ET; `ask-weekly-grade-schedule` Tue + Wed 10:00 ET |
| Read it | eval dashboard → **Ask exam**, or `python webapp/ask_weekly.py report` |

Grading runs Tuesday AND Wednesday: nflverse's box scores usually land
Tuesday, and a week with no box scores yet is skipped and picked up the next
run. A week already graded is never graded twice; a week already asked is
never asked twice.

## Setup (one time)

```bash
P=advisor-agent-2026
SA=web-app@$P.iam.gserviceaccount.com
IMG=us-central1-docker.pkg.dev/$P/agent-reports/web:$(git rev-parse --short HEAD)

# Tables (created from the code's own schema)
python -c "import sys; sys.path[:0]=['webapp']; import ask_weekly as aw; from google.cloud import bigquery as b; c=aw._bq(); [c.create_table(b.Table(t, schema=aw._schema(k)), exists_ok=True) for t,k in [(aw.ANSWERS,'answers'),(aw.GRADES,'grades')]]"

# The web-app SA already reads projections, calls Gemini and invokes news-mcp.
# It needs write on these two tables only -- not on the dataset.
for T in ask_weekly ask_weekly_grades; do
  bq add-iam-policy-binding --member serviceAccount:$SA \
    --role roles/bigquery.dataEditor fantasy-football-ml-503718:dfs.$T
done

# Jobs: the web image with a different command
for CMD in ask grade; do
  gcloud run jobs create ask-weekly-$CMD --project $P --region us-central1 \
    --image $IMG --service-account $SA --cpu 1 --memory 1Gi \
    --task-timeout 1800 --max-retries 1 \
    --command python --args webapp/ask_weekly.py,$CMD \
    --set-env-vars GOOGLE_CLOUD_PROJECT=$P
done

# Schedules (same scheduler SA as the gameday report)
for CMD in ask grade; do
  gcloud run jobs add-iam-policy-binding ask-weekly-$CMD --project $P --region us-central1 \
    --member serviceAccount:report-scheduler@$P.iam.gserviceaccount.com --role roles/run.invoker
done
gcloud scheduler jobs create http ask-weekly-ask-schedule --project $P --location us-central1 \
  --schedule "15 11 * * 0" --time-zone America/New_York --http-method POST \
  --uri https://run.googleapis.com/v2/projects/$P/locations/us-central1/jobs/ask-weekly-ask:run \
  --oauth-service-account-email report-scheduler@$P.iam.gserviceaccount.com
gcloud scheduler jobs create http ask-weekly-grade-schedule --project $P --location us-central1 \
  --schedule "0 10 * * 2,3" --time-zone America/New_York --http-method POST \
  --uri https://run.googleapis.com/v2/projects/$P/locations/us-central1/jobs/ask-weekly-grade:run \
  --oauth-service-account-email report-scheduler@$P.iam.gserviceaccount.com
```

## Every web deploy

The jobs pin an image tag. After a new gameday-web image, point both jobs at
it too, or the exam keeps grading the old Ask:

```bash
for CMD in ask grade; do
  gcloud run jobs update ask-weekly-$CMD --project $P --region us-central1 --image $IMG
done
```

## By hand

```bash
gcloud run jobs execute ask-weekly-ask --project $P --region us-central1 --args=webapp/ask_weekly.py,ask,--dry-run --wait
python webapp/ask_weekly.py report        # season to date, from the laptop
```
