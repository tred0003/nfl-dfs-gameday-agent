# NFL DFS: projection model, lineup optimizer and a gameday AI agent on Google Cloud

Every week this system predicts how many DraftKings points each NFL player will score,
builds the best legal lineup under the $50,000 salary cap, and then, before kickoff, runs an
AI agent that checks each user's real lineup against injuries, weather, Vegas lines and the
news. It has run in production since September 2026 for several signed-in users.

The model is deliberately simple. **The system around it is deliberately production-grade**:
evals that gate every deploy, an LLM judge checked against human grades, cost metering,
prompt-injection defense, monitoring against a service-level objective, and CI/CD.

```mermaid
flowchart LR
    subgraph ML["ML project"]
        NV[nflverse history<br/>2016-2026] --> XGB[XGBoost model<br/>Vertex AI Model Registry]
        DK[DraftKings slate] --> PS[Cloud Run job<br/>project-slate]
        XGB --> PS --> BQ[(BigQuery<br/>projections)]
        NEWS[News feeds + podcasts] --> NA[Cloud Run job<br/>news-archive] --> VEC[(BigQuery<br/>vector search)]
        VEC --> MCP[MCP server<br/>news-mcp]
    end
    subgraph AG["Agent project"]
        SCH[Cloud Scheduler] --> RPT[Cloud Run job<br/>gameday-report]
        RPT --> AE[ADK agent on<br/>Vertex AI Agent Engine<br/>Gemini 2.5 Flash]
        AE --> FS[(Firestore<br/>users, lineups, reports)]
        RPT --> DISC[Discord]
        WEB[Web app on Cloud Run<br/>lineup picker, Search, Ask] --> FS
    end
    BQ --> AE
    VEC --> AE
    BQ --> WEB
    MCP --> WEB
```

## Three layers, kept separate on purpose

| Layer | What it does | What kind of code |
|---|---|---|
| **1. Projection model** | Predicts DK points per player (a mean and an 85th-percentile ceiling) | Machine learning: XGBoost trained on 2016-2026 data, walk-forward validated |
| **2. Optimizer** | Picks the 9 players that maximize projected points under the cap and roster rules | **Not AI.** An integer linear program (PuLP + CBC). One exact, reproducible answer |
| **3. Gameday agent** | Reads the news and reconciles it with the lineup: who's out, who's limited, what changed since yesterday | A Google ADK agent on Vertex AI Agent Engine. Most tools are plain Python; the LLM decides which to call and writes the report |

The guiding rule: **anything with one right answer is code; the LLM only handles what exists
only as sentences.** Salaries, injury tags, Vegas totals and projections are Python. Beat
reports and practice notes are where the model earns its keep.

## Results worth knowing

**Evaluation.** The agent is probabilistic, so "it works" is a percentage, not an assert. The
harness in [`gameday_agent/eval/`](gameday_agent/eval/README.md) runs the real agent against
synthetic scenarios (DK outage, prompt injection, an emergency scratch, a clean lineup...) and
reports three numbers that are never averaged together:

- **Checks:** deterministic assertions on the report and the tool trace. Any failure blocks the deploy.
- **Quality:** an LLM judge (Gemini 2.5 Pro) scoring a six-criterion rubric, compared against my own hand grades.
- **Trajectory:** what the agent did: which tools, how many calls, how long.

*"A run can score 100 on quality and be broken."* That's why checks and trajectory gate independently.

**Cost: -68% per report.** Before optimizing anything, I metered everything, and found the
cost meter had been missing 63-87% of the real bill (thinking tokens bill as output, at ~8x the
input price). One lever per commit, each verified on the eval suite and a live dry run: the
daily report went from about $0.035 to $0.011, with every check still passing and the
injection screen still catching 28 of 28 attacks.

**Agent vs pipeline, measured.** For the archive's "Ask" feature I built both designs and
scored 680 answers against an answer key. The fixed pipeline beat the agent by +0.33 on a 0-2
correctness scale (95% CI +0.23 to +0.43). The agent's main failure: it skipped retrieval on
61 of 340 answers. Write-up: [`ask_eval/README.md`](ask_eval/README.md).

**Search quality.** I built a 154-question test set for the news-archive search, graded by
an LLM judge and cross-checked by a second model (85% agreement). I then tried fine-tuning
embeddings vs adding a re-ranker. The re-ranker won (+0.15 nDCG@10) and is in production.
Write-up: [`embed_tune/README.md`](embed_tune/README.md).

**The model itself, honestly.** It doesn't beat FantasyData's paid projections (Spearman
0.644 vs 0.671 on dev; a statistical tie on lineup outcomes). The point was the lifecycle, not
the leaderboard.

## Production

| Piece | Where it runs |
|---|---|
| Weekly projections | Cloud Run job + Cloud Scheduler, results in BigQuery |
| Gameday reports, per user | Cloud Run job calling the agent on Agent Engine; stored in Firestore, owner's copy to Discord |
| Lineup web app (Search + Ask over the news archive) | Cloud Run service, Firebase Auth sign-in |
| News archive | Daily Cloud Run job, embedded with `text-embedding-005`, searched with BigQuery `VECTOR_SEARCH` + Vertex ranking |
| MCP server | Private Cloud Run service exposing the archive to the web app and to AI clients like Claude Code |
| CI/CD | Cloud Build: unit tests on every push, the eval gate before every agent deploy, then a post-deploy canary |
| Monitoring | Log-based metrics, alert policies, an SLO-based dashboard, and a weekly job that has the LLM judge review real production reports |

**Security:** tool results are screened for prompt injection before the agent sees them (tested
against a labeled attack set), each job has its own least-privilege service account, and the two
Google Cloud projects connect through a handful of narrow, documented grants.

## Repo map

```
src/              model: data pulls, features, training, optimizer, slate projection
gameday_agent/    the ADK agent, its tools, injection screen, canary, weekly review
  eval/           eval harness: scenarios, rubric, LLM judge, baseline gate
webapp/           Cloud Run web app: lineup picker, Search, Ask (with citation checks)
mcp_server/       news archive as an MCP server
news_rag/  podcast_rag/   news and podcast ingestion into the archive
embed_tune/       search-quality experiments (embeddings, fine-tuning, re-ranking)
ask_eval/         agent-vs-pipeline experiment for Ask
deploy/           Dockerfiles, Cloud Build configs, runbooks (DEPLOY, WEB, MCP, MONITORING...)
tests/            unit tests (run in CI without cloud access)
FEATURES.md       the feature spec and backtests of common DFS heuristics
```

## Data and terms

All sources are free. Historical stats come from [nflverse](https://github.com/nflverse).
DraftKings salaries come from an unofficial, undocumented endpoint, isolated in
`src/dk_client.py` so that when it changes, exactly one file breaks. Pulled data is never
committed or redistributed. This is a personal learning project; no real money is wagered on it.

## Run it locally

```
pip install -r requirements.txt
python src/fetch_nflverse.py          # one-time history pull
python src/features.py                # build the model table
python src/train.py                   # walk-forward validation
python src/dk_client.py               # snapshot the Sunday main slate
python src/optimize.py --slate <projections_*.parquet> --n 3
python -m unittest discover -s tests  # the CI test suite
```

The cloud setup is in [`deploy/`](deploy/); start with `deploy/DEPLOY.md`.

---

Built with [Claude Code](https://claude.com/claude-code) as a coding partner, alongside
Google's Professional Machine Learning Engineer certification.
