# Phase 4.1 — Training the model on Agent Platform

Goal: the model stops being trained on your laptop. A container wakes up in
us-central1, reads the model table out of Cloud Storage, fits both models, and
writes a versioned artifact to a bucket. Your laptop is closed.

**Run these yourself — the point is the GCP reps.** Each block says what it does
and why, not just what to type.

**What this touches:** Cloud Storage, Artifact Registry, Cloud Build, Agent
Platform custom training jobs, and service-account IAM. That is most of the
training half of the PMLE exam, on something you actually want to exist.

**Cost:** a few cents. One build, one `n1-standard-4` running for under two
minutes, and about 3 MB of storage.

---

## Naming, before anything else

Google renamed this surface in April 2026, and the rename is **console-deep,
not API-deep**. This is the thing that cost you the exam, so keep both columns
in your head:

| what you type | what the console calls it now |
|---|---|
| `gcloud ai custom-jobs` | Agent Platform → **Models** → Training |
| `gcloud ai models upload` | → **Model Registry** |
| `gcloud ai endpoints` | → **Online Inference** |
| `aiplatform.googleapis.com` | **"Agent Platform API"** (same ID) |

The CLI still says "Vertex AI" in its own help text. That is not out of date —
the API surface genuinely did not rename. Only the console did.

---

## 0. Set your variables

```bash
export PROJECT=fantasy-football-ml-503718
export REGION=us-central1
export BUCKET=$PROJECT-nfl-dfs
export REPO=nfl-dfs
export IMAGE=$REGION-docker.pkg.dev/$PROJECT/$REPO/train:v1

gcloud config set project $PROJECT
```

Bucket names are **globally unique across all of Google Cloud**, not just your
project — prefixing with the project ID is the standard way to guarantee you
get one.

## 1. Enable the APIs

```bash
gcloud services enable \
  aiplatform.googleapis.com \
  storage.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com
```

`aiplatform.googleapis.com` is already on in this project — it shows up as
**"Agent Platform API"** in the console. Same ID it always had.

## 2. Create the bucket

```bash
gcloud storage buckets create gs://$BUCKET --location=$REGION
```

**Put the bucket in the same region as the training job.** A job in
`us-central1` reading from a multi-region or a different-region bucket works,
but you pay egress and add latency for no reason. Region matching is a routine
exam question.

## 3. Upload the model table

The image ships **code, not data**. That is deliberate — retraining on new data
should never require rebuilding a container.

```bash
gcloud storage cp data/processed/model_table_v3.parquet gs://$BUCKET/processed/

# NOTE: the tables were rebuilt on 2026-09-16 (scoring fix + leak fixes), so
# re-upload even if you copied them earlier. 51,356 rows now, not 51,046.
gcloud storage ls gs://$BUCKET/processed/
```

About 12 MB. Re-run this line each week after you rebuild features, and the
next training run picks it up with no image rebuild.

## 4. Create the image repository

```bash
gcloud artifacts repositories create $REPO \
  --repository-format=docker \
  --location=$REGION \
  --description="NFL DFS containers"
```

Artifact Registry is where built images live. **Container Registry (`gcr.io`)
is retired** — if a tutorial tells you to push to `gcr.io`, it predates 2026.

## 5a. Check the image will actually run (one second, saves ten minutes)

```bash
python deploy/check_image_deps.py
```

A Dockerfile's `COPY` line is a dependency list that nothing keeps in sync. When
`train.py` gained `import calibrate`, the COPY line did not follow — and that
build would have succeeded, pushed, been submitted as a job, and died on its
first line with `ModuleNotFoundError`, with the error buried in Cloud Logging
five minutes later.

This parses the imports transitively and compares them against the Dockerfile.
Run it before every build.

## 5. Build the image

You have no local Docker and do not need it. Cloud Build uploads the source,
builds in the cloud, and pushes to Artifact Registry.

```bash
gcloud builds submit \
  --config deploy/cloudbuild.train.yaml \
  --substitutions=_IMAGE=$IMAGE \
  .
```

First build takes 3–5 minutes (xgboost and scipy wheels). Later builds reuse the
dependency layer and are much faster, **as long as you don't touch
`requirements-train.txt`** — changing that file invalidates the cache.

## 6. Run the training job

```bash
export RUN=$(date +%Y%m%d-%H%M%S)

gcloud ai custom-jobs create \
  --region=$REGION \
  --display-name=nfl-dfs-train-$RUN \
  --base-output-directory=gs://$BUCKET/runs/$RUN \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=$IMAGE \
  --args=^@^--table=model_table_v3@--export@--no-save@--data-dir=gs://$BUCKET/processed
```

Three things here are worth understanding rather than copying.

**`--base-output-directory` is how the job learns where to write.** Agent
Platform sets the environment variable **`AIP_MODEL_DIR`** inside your container
to `<base-output-directory>/model`. `train.py` reads that as its default
`--model-dir`, so the same script that writes to `models/current` on your laptop
writes to Cloud Storage here **with no code change**. That is the whole trick.

**`^@^` switches the argument separator to `@`.** Without it gcloud splits
`--args` on commas, and it would also choke on the `//` in the `gs://` URI.
Same pattern as the `MY_LINEUP` variable in `DEPLOY.md`.

**A timestamped run directory, not a fixed one.** Every run gets its own folder,
so a bad model never overwrites a good one and you can always point Model
Registry back at last week's artifact.

**No GPU, on purpose.** Gradient-boosted trees on 51k rows are CPU work —
`tree_method="hist"` is already the fast path. Attaching an accelerator here
would cost more and run no faster, and knowing *when not to* is as testable as
knowing how.

## 7. Watch it

```bash
gcloud ai custom-jobs list --region=$REGION --limit=5

# streams until the job finishes
gcloud ai custom-jobs stream-logs JOB_ID --region=$REGION
```

`JOB_ID` is the numeric id from the list output. You should see the familiar
walk-forward lines, the held-out table, then:

```
final fit on all 51,046 rows (83 features)
exported artifact -> gs://.../runs/<RUN>/model
```

## 8. Confirm the artifact landed

```bash
gcloud storage ls -l gs://$BUCKET/runs/$RUN/model/
gcloud storage cat gs://$BUCKET/runs/$RUN/model/model_card.json
```

Three files: `model_mean.ubj`, `model_ceiling.ubj`, `model_card.json`.

**Read the card.** `git_sha` ties the artifact to the code that built it,
`features` pins the exact column order that prevents training/serving skew, and
`metrics` carries the walk-forward scores — which are the model's *estimated*
skill, not this artifact's measured score. The shipped model was refit on
everything and has never been graded on data it did not see.

---

## If it fails

**`PermissionDenied` on the bucket.** The job runs as the Compute Engine
default service account, `PROJECT_NUMBER-compute@developer.gserviceaccount.com`.
It needs object read/write:

```bash
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT --format='value(projectNumber)')
gcloud storage buckets add-iam-policy-binding gs://$BUCKET \
  --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"
```

**`ModuleNotFoundError: model_io`.** `PYTHONPATH=/app/src` is missing from the
image, or `model_io.py` was not copied in.

**The job succeeds but writes nothing.** You left `--export` off. Without it
`train.py` validates and prints and saves no artifact — which is the correct
default for a research run and the wrong one for a training job.

**`FileNotFoundError` on the parquet.** Step 3 was skipped, or `--data-dir`
points at the bucket root instead of `gs://$BUCKET/processed`.

---

## What this bought you

The model is now produced by a **reproducible, auditable process** rather than
by whatever was on your laptop at the time. Every run leaves a timestamped
artifact, tied by commit hash to the code that made it, with its feature
contract pinned.

That is the prerequisite for everything in 4.2 through 4.5 — you cannot
register, serve, or monitor a model that only exists as a local variable.

**Next: 4.2 — upload this artifact to Model Registry, where it gets a version
number and an alias.**
