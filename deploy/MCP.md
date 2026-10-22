# News-archive MCP server

`mcp_server/server.py` on Cloud Run (service `news-mcp`, fantasy-football-ml-503718,
us-central1). It publishes the agent's two archive tools over MCP so any MCP
client — Claude Code, Gemini CLI, another agent — can ask the archive questions
directly, e.g. while building a lineup by hand.

**It is not a new tool.** `player_history` and `search_news_archive` call
`gameday_agent/archive.py`: the same SQL, the same "screen = clean" filter, the
same fencing and freshness check the deployed agent runs. Only the tool
descriptions are new, because the reader is different (see the module docstring).

| | Agent tool | MCP tool |
|---|---|---|
| Who picks the players | the lineup in session state, resolved against DK | the caller passes names |
| Search limit | 3 per run (session state) | none — no run to count against; IAM-only |
| Descriptions | written for writing a lineup report | written for a person asking questions |

## Who calls it

| Client | How | Auth |
|---|---|---|
| Claude Code | `claude mcp add --transport http` through the gcloud proxy | your gcloud identity |
| gameday-web, "My players" list | `webapp/news.py`, one direct tool call | ID token as `web-app@` (`run.invoker`) |
| gameday-web, Ask box | `webapp/ask.py`: Gemini 2.5 Flash with tools built from this server's `tools/list`, up to 4 calls, then an answer with dates and sources | same |

Add a tool here and the Ask agent sees it on the next question; nothing in
the web app lists tool names.

## Shape

- **Transport:** streamable HTTP at `/mcp`, stateless, JSON responses. Stateless
  means no session lives on one instance, so Cloud Run can scale to zero and any
  instance can answer any request.
- **Auth:** Cloud Run IAM (`--no-allow-unauthenticated`). No token code in the
  server; a caller needs `roles/run.invoker` on the service.
- **Identity:** its own service account `news-mcp`, read-only: BigQuery
  `jobUser` + `dataViewer` on `dfs`, and `aiplatform.user` for the query embedding.
  Not the refresh job's account, which can write the table.
- **Cost:** scales to zero. A history call is one BigQuery query; a search is two
  queries plus one embedding call — fractions of a cent.

## Setup (one time)

```bash
P=fantasy-football-ml-503718
R=us-central1
SA=news-mcp@$P.iam.gserviceaccount.com
IMG=$R-docker.pkg.dev/$P/nfl-dfs/news-mcp:$(git rev-parse --short HEAD)

# Image (the build fails if the tool list is not exactly the two tools)
gcloud builds submit --project $P --config deploy/cloudbuild.mcp.yaml \
  --substitutions _IMAGE=$IMG .

# Read-only identity
gcloud iam service-accounts create news-mcp --project $P \
  --display-name "news archive MCP server (read-only)"
gcloud projects add-iam-policy-binding $P --member serviceAccount:$SA \
  --role roles/bigquery.jobUser --condition None
gcloud projects add-iam-policy-binding $P --member serviceAccount:$SA \
  --role roles/aiplatform.user --condition None
# dataViewer on the dfs dataset only
bq add-iam-policy-binding --member serviceAccount:$SA \
  --role roles/bigquery.dataViewer $P:dfs

# Service: private, scales to zero
gcloud run deploy news-mcp --project $P --region $R --image $IMG \
  --service-account $SA --no-allow-unauthenticated \
  --cpu 1 --memory 1Gi --min-instances 0 --max-instances 2 --timeout 60 \
  --set-env-vars GOOGLE_CLOUD_PROJECT=$P,GOOGLE_GENAI_USE_VERTEXAI=TRUE
```

## Connect Claude Code

The service only takes requests with a Google identity token. The simplest
way from the laptop is the gcloud proxy, which adds your token to every request:

```bash
gcloud run services proxy news-mcp --project fantasy-football-ml-503718 \
  --region us-central1 --port 8799            # leave it running
claude mcp add --transport http nfl-news http://127.0.0.1:8799/mcp
```

Then in Claude Code: "what has the archive said about Justin Jefferson's ankle?"

Without the cloud (laptop only, your own gcloud credentials):

```bash
claude mcp add nfl-news -- python C:/new_career2026/nfl-dfs/mcp_server/server.py --stdio
```

## Redeploy

```bash
gcloud builds submit --project $P --config deploy/cloudbuild.mcp.yaml --substitutions _IMAGE=$IMG .
gcloud run deploy news-mcp --project $P --region $R --image $IMG
```

Any change to `gameday_agent/archive.py` reaches the agent on the agent's
deploy and the MCP server on this one — redeploy both.

## Tests

`tests/test_mcp_server.py` drives the server through a real MCP client
in-process (BigQuery faked). It is skipped in CI, which installs neither ADK nor
the MCP SDK — same as `tests/test_archive.py`.
