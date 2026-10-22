# News archive in production — the agent reads weeks of news, not 48 hours

Built 2026-10-22. Before this, `dfs.news_chunks` (the RAG archive, `news_rag/`)
was loaded once from the laptop on 10/14 and nothing used it. Now a daily job
keeps it current and the gameday agent reads it on every report.

```
Cloud Scheduler  news-archive-daily  11:00 ET every day (before the 12:30 report)
      │  (project-slate-scheduler@, run.invoker on the job)
      ▼
Cloud Run job  news-archive   (runs as news-archive@)   news_rag/refresh.py
  1. add player_keys / screen columns if missing (one-time, idempotent)
  2. collect ESPN player notes + NFL stories since the newest item - 36 h
  3. new chunks only: player_keys, injection screen (Flash-Lite), embedding
  4. append (load job) to dfs.news_chunks
  5. fill: any row missing keys or a clean/instruction verdict → MERGE
      │
      ▼
BigQuery  fantasy-football-ml-503718.dfs.news_chunks
      │   READER on dfs + jobUser, already granted to the Agent Engine
      │   service agent for get_value_picks — no new permission
      ▼
gameday agent
  get_player_history    CORE, prefetched every report: newest notes per
                        rostered player, 28 days, by name + date (SQL)
  search_news_archive   OPTIONAL, the model's call: question → embedding →
                        VECTOR_SEARCH with player + date + screen filter,
                        plus the 2 newest items
any failed execution ──► existing alert "Projection job FAILED" (it matches
                          every Cloud Run job in the project) ──► email
canary (Thu) ──► fails if the archive's newest item is > 36 h old
```

## Design choices worth defending

- **History is a lookup, search is retrieval.** "Everything recent about my
  player" has no question for an embedding to rank against, and for news the
  ranking that matters is time. So the core tool is a filter (name, date,
  screen) ordered newest-first. Vector search earns its place when there IS
  a question ("how long has his hamstring been an issue?"), and there the
  newest items still get guaranteed slots because embeddings encode meaning,
  not time.
- **Core, not optional.** Prefetched like the other core tools, so it costs no
  extra model turn and the model cannot skip it. The optional search is the
  part the model decides.
- **Screened at index time, filtered at query time.** Every chunk is screened
  once by the same `injection.screen()` the agent uses, and retrieval returns
  only `screen = 'clean'`. A failed screen leaves `unchecked`, which is
  invisible until the next run re-checks it. Fail closed.
- **No model between the source and the agent.** Notes are passed verbatim,
  fenced, with date and source. Nothing is summarised on the way in, so
  there is no extracted claim that can drift from its quote.
- **Staleness is a failure, not a footnote.** A refresh job that stops makes
  the tool answer "ok" from a history that ends days ago. The tool reports
  the archive's age, the report says so past 36 h, and the canary fails.
- **ESPN's player feed carries columns too.** "Week 5 winners and losers"
  sits in a player's feed at 12K characters naming him once. Real notes have
  no link and columns link to a /story/ page, so columns are classified as
  stories at collection and excluded from the per-player notes.

## First measurements (2026-10-22)

- Dry run: 794 player feeds, 0 failed; 1,273 articles since 10/13 → 2,272 new
  chunks (~280/day).
- Eval `history_pattern` 3/3: the take used the archive-only pattern
  ("recurring hamstring issue ... his snaps dipped last week") and never
  revived a 24-day-old ankle note contradicted by a newer one. Full suite
  10/10, gate PASS.
- Expected cost: ~$1/month all-in (see the 10/22 estimate); the one-time
  backfill screen of ~21K chunks is under $1 with Flash-Lite.

## Setup (one time)

```bash
P=fantasy-football-ml-503718
SA=news-archive@$P.iam.gserviceaccount.com
IMG=us-central1-docker.pkg.dev/$P/nfl-dfs/news-archive:$(git rev-parse --short HEAD)

# Image
gcloud builds submit --project $P --config deploy/cloudbuild.news.yaml \
  --substitutions _IMAGE=$IMG .

# Identity: write the table, run queries, call Gemini + embeddings in this project
gcloud iam service-accounts create news-archive --project $P \
  --display-name "news archive refresh job"
gcloud projects add-iam-policy-binding $P --member serviceAccount:$SA \
  --role roles/bigquery.jobUser --condition None
gcloud projects add-iam-policy-binding $P --member serviceAccount:$SA \
  --role roles/aiplatform.user --condition None
# dataEditor on the dfs dataset only (Cloud Shell's bq; the laptop bq is broken)
bq add-iam-policy-binding --member serviceAccount:$SA \
  --role roles/bigquery.dataEditor $P:dfs

# Job
gcloud run jobs create news-archive --project $P --region us-central1 \
  --image $IMG --service-account $SA --cpu 1 --memory 2Gi \
  --task-timeout 3600 --max-retries 1 \
  --set-env-vars GOOGLE_CLOUD_PROJECT=$P,GOOGLE_GENAI_USE_VERTEXAI=TRUE,EMBED_PROJECT=$P,BQ_PROJECT=$P

# First run does the backfill (~21K chunks screened); watch it
gcloud run jobs execute news-archive --project $P --region us-central1 --wait

# Schedule, reusing the projection job's scheduler identity
gcloud run jobs add-iam-policy-binding news-archive --project $P --region us-central1 \
  --member serviceAccount:project-slate-scheduler@$P.iam.gserviceaccount.com \
  --role roles/run.invoker
gcloud scheduler jobs create http news-archive-daily --project $P --location us-central1 \
  --schedule "0 11 * * *" --time-zone America/New_York --http-method POST \
  --uri https://run.googleapis.com/v2/projects/$P/locations/us-central1/jobs/news-archive:run \
  --oauth-service-account-email project-slate-scheduler@$P.iam.gserviceaccount.com
```

Then, in this order (the agent's SQL needs the columns the first run adds):
1. Redeploy the agent: `nfl-dfs-deploy` trigger (tests → eval gate → deploy → verify).
2. Rebuild the report-client image and point `gameday-canary` at it — the
   canary's checks for the two new tools live in that image.
3. Run the canary dry run.

```bash
gcloud run jobs execute news-archive --project $P --region us-central1 --args=--dry-run --wait  # no writes
```
