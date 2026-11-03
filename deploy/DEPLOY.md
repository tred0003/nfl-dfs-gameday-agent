# Deploying the gameday report to Google Cloud

> **Superseded 2026-10-09** by `REPORTS-CLOUD.md`. This plan (rule-based report as a Cloud Run job) was never built; the agents replaced it.

Goal: at **12:00 PM ET every Sunday**, a container wakes up, pulls the live
DraftKings slate, and posts an availability / weather / news report to Discord.
Your laptop is closed.

Run these yourself — the point is the GCP reps. Each block says what it does and
why, not just what to type.

**What this touches:** Artifact Registry, Cloud Build, Secret Manager, Cloud Run
jobs, Cloud Scheduler, and service-account IAM. That is most of the Google Cloud
deployment surface, on something that actually needs to run.

**Cost:** effectively zero. One tiny build a week, one container execution a week
measured in seconds, a few MB of image storage. All comfortably inside free tiers.

---

## 0. Set your variables

```bash
export PROJECT=advisor-agent-2026
export REGION=us-central1
export REPO=nfl-dfs
export IMAGE=$REGION-docker.pkg.dev/$PROJECT/$REPO/gameday-report:v1

gcloud config set project $PROJECT
```

## 1. Enable the APIs

Nothing works until these are on. Each is a separate service and each has to be
enabled per project — a first-time deploy fails here more often than anywhere else.

```bash
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  cloudscheduler.googleapis.com \
  secretmanager.googleapis.com
```

## 2. Create the image repository

Artifact Registry is where built images live. (Container Registry is the older
one — check what the console calls it today before following any older tutorial.)

```bash
gcloud artifacts repositories create $REPO \
  --repository-format=docker \
  --location=$REGION \
  --description="NFL DFS gameday report"
```

## 3. Put the Discord webhook in Secret Manager

The webhook URL is a credential — anyone holding it can post to your server. It
does not belong in the image, in an env var on disk, or in git.

Create the webhook first: Discord → Server Settings → Integrations → Webhooks →
New Webhook → pick a channel → Copy Webhook URL.

```bash
printf '%s' 'PASTE_YOUR_WEBHOOK_URL_HERE' | \
  gcloud secrets create discord-webhook --data-file=-
```

`printf` rather than `echo` on purpose: `echo` appends a newline, which becomes
part of the secret and produces a confusing 404 from Discord later.

## 4. Build the image with Cloud Build

You have no local Docker, and you do not need it. Cloud Build uploads the source,
builds in the cloud, and pushes to Artifact Registry.

```bash
gcloud builds submit \
  --config deploy/cloudbuild.yaml \
  --substitutions=_IMAGE=$IMAGE \
  .
```

Watch the log stream. First build takes a few minutes (installing pandas); later
ones are faster.

## 5. Create the Cloud Run job

A **job**, not a service. Services listen on a port and wait for traffic; jobs run
to completion and exit. A scheduled report is a job.

```bash
gcloud run jobs create gameday-report \
  --image=$IMAGE \
  --region=$REGION \
  --max-retries=2 \
  --task-timeout=5m \
  --set-secrets="DISCORD_WEBHOOK_URL=discord-webhook:latest" \
  --set-env-vars="^@^WEEK=1@DRAFT_GROUP=151307@MY_LINEUP=Jalen Hurts,Jahmyr Gibbs,De'Von Achane,DK Metcalf,Makai Lemon,Calvin Ridley,Michael Mayer,Justin Jefferson,Ravens"
```

**The `^@^` prefix matters.** It tells gcloud to split on `@` instead of `,` —
without it, your comma-separated lineup gets parsed as nine separate env vars and
the job silently reports on nobody.

## 6. Test it before trusting it to a schedule

```bash
gcloud run jobs execute gameday-report --region=$REGION --wait
```

Check Discord. If nothing arrives:

```bash
gcloud run jobs executions list --job=gameday-report --region=$REGION
gcloud logging read \
  'resource.type=cloud_run_job AND resource.labels.job_name=gameday-report' \
  --limit=50 --format='value(textPayload)'
```

## 7. Let Cloud Scheduler invoke the job

Scheduler calls the Cloud Run Admin API as a service account, and that account
needs permission to run the job.

```bash
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT --format='value(projectNumber)')
export SA=$PROJECT_NUMBER-compute@developer.gserviceaccount.com

gcloud run jobs add-iam-policy-binding gameday-report \
  --region=$REGION \
  --member="serviceAccount:$SA" \
  --role="roles/run.invoker"
```

## 8. Schedule it

```bash
gcloud scheduler jobs create http gameday-report-sunday \
  --location=$REGION \
  --schedule="0 12 * * 0" \
  --time-zone="America/New_York" \
  --uri="https://run.googleapis.com/v2/projects/$PROJECT/locations/$REGION/jobs/gameday-report:run" \
  --http-method=POST \
  --oauth-service-account-email=$SA
```

`0 12 * * 0` is noon on Sunday. **`--time-zone` is not optional** — leave it off
and Cloud Scheduler uses UTC, which would fire at 8:00 AM ET. Using
`America/New_York` rather than a fixed offset also means it stays correct when
daylight saving ends in November, mid-season.

Force one run to prove the wiring:

```bash
gcloud scheduler jobs run gameday-report-sunday --location=$REGION
```

---

## Every week after this

Only the lineup and week change, and neither needs a rebuild:

```bash
gcloud run jobs update gameday-report --region=$REGION \
  --set-env-vars="^@^WEEK=2@DRAFT_GROUP=<new dg>@MY_LINEUP=<nine names>"
```

Get the new draft group id with `python src/dk_client.py --list`.

Rebuild only when the code changes — bump the tag (`:v2`) rather than overwriting
`:v1`, so a bad deploy can be rolled back by pointing the job at the previous tag.

## Timing note

Official inactives drop **90 minutes before kickoff** — 11:30 AM ET for a 1:00 PM
slate. Noon puts the report after that deadline with an hour to spare. If you want
a second look closer to lock, add another scheduler job at `40 12 * * 0`.
