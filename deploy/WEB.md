# The web app — multi-user step 2 (built 2026-10-16)

**URL:** https://gameday-web-389876399573.us-central1.run.app

Sign in, edit your lineup, read your latest report. Four users (u1-u4), all
the same person via Gmail plus-addressing (`tred0003+uN@gmail.com`).

```
browser ── Firebase Auth (email/password) ──► ID token
browser ── Authorization: Bearer <token> ──► gameday-web (Cloud Run, Flask)
                                                │ verify token → uid
                                                ▼
                                   Firestore users/{uid}, users/{uid}/reports/*
                                                ▲
                       gameday-report job (12:30) writes the reports
```

## Isolation: the interview point

- The uid comes **only** from the verified token (`webapp/main.py`
  `current_uid()`). No route reads a uid from the URL, query or body; a body
  `{"uid": "u1"}` is ignored. User 2 can't even *name* user 1's data.
- The browser never touches Firestore. `deploy/firestore.rules` denies all
  client access; the server reads and writes as its own service account.
- Signed in but no `users/{uid}` doc → **403**, not an empty lineup.
- `tests/test_webapp.py` (9 tests, in CI): no token 401, forged token 401,
  unprovisioned 403, `?uid=u1` and body `uid` ignored, lineup validation.

**Verified live 10/16:** no token → 401; forged token → 401; a real Firebase
token for a throwaway account `u9` (created, used, deleted) → 403 on read and
on a PUT that named `uid: u1`; u1's lineup was unchanged.

## What was set up (all in `advisor-agent-2026`)

| Piece | Detail |
|---|---|
| Firebase | Added to the GCP project (`addFirebase`); web app `gameday-web` registered |
| Auth | Identity Platform initialised; Email/Password provider on. Accounts u1-u4 created by `deploy/create_auth_users.py` with uids matching the Firestore docs, **no passwords** |
| Rules | `deploy/firestore.rules` (deny all clients) released to `cloud.firestore` |
| Identity | `web-app@` — `roles/datastore.user`; since 10/19 READER on `dfs` + `bigquery.jobUser` in the ML project (player pool); since 10/22 `aiplatform.user` here (Ask agent) and `run.invoker` on `news-mcp` |
| Service | Cloud Run `gameday-web`, public (auth is in-app), max 2 instances, env `FIREBASE_API_KEY`, `FIREBASE_AUTH_DOMAIN`, `GOOGLE_CLOUD_PROJECT` |
| Image | `deploy/Dockerfile.web`, `deploy/cloudbuild.web.yaml`, in the `agent-reports` repo as `web:<sha>` |

## News archive card (added 2026-10-22)

- `POST /api/news/history` — four weeks of notes on your lineup, one direct
  MCP call to `news-mcp` (`webapp/news.py`).
- `POST /api/news/ask` — a question answered by a Gemini 2.5 Flash agent whose
  tools come from `news-mcp`'s `tools/list` (`webapp/ask.py`); at most 4 tool
  calls, 20 questions per user per day, one `event=archive_ask` log line each
  (~$0.002, 3-11 s).
- Grants: `web-app@` has `run.invoker` on `news-mcp` (ML project) and
  `aiplatform.user` in `advisor-agent-2026`. Remove the first and the card
  answers 503; the rest of the site is unaffected.
- Live: `gameday-web-00005-jdr`, image `web:4db1051`. See `deploy/MCP.md`.

## Lineup import from a screenshot (added 2026-10-30)

"Import from screenshot" on the lineup card: one or two DK lineup screenshots (a
lineup is taller than a phone screen; both are read as one) go to
`POST /api/lineup/import` (`webapp/lineup_import.py`). Document AI's **OCR
processor** reads the lines; code matches them against this week's slate by
exact name (case, punctuation and Jr./Sr./III ignored, never a first-initial
guess). The page shows what was found and replaces the lineup only if the user
says yes; the normal save does the write.

- Processor: `lineup-screenshot-ocr`, `projects/advisor-agent-2026/locations/us/processors/6502beabda51e08f`
  (env `DOCAI_PROCESSOR` overrides). Created by REST, `type: OCR_PROCESSOR`.
- `web-app@` needs `roles/documentai.apiUser`:
  `gcloud projects add-iam-policy-binding advisor-agent-2026 --member=serviceAccount:web-app@advisor-agent-2026.iam.gserviceaccount.com --role=roles/documentai.apiUser --condition=None`
- Checked before sending: PNG/JPEG/WebP/PDF by magic bytes, 5 MB max, 20 imports per user per day.
- One log line per import, `event=lineup_import` (lines, matched, not_recognised, warnings).
- Cost: one OCR page each, about $0.0015.

## First sign-in (per user)

Type the email, click **"Email me a link to set my password"**, open the mail
in Gmail (check spam: it comes from `noreply@advisor-agent-2026.firebaseapp.com`),
set a password, sign in.

## Gotchas hit

1. **Auth config 404 `CONFIGURATION_NOT_FOUND`** until
   `identityPlatform:initializeAuth` was called — adding Firebase alone does
   not initialise Auth. The first initialise call returned 503; the retry worked.
2. **403 "Identity Toolkit API has not been used in project
   fantasy-football-ml"** from the laptop. ADC's quota project wins over an
   `x-goog-user-project` header; fixed with `creds.with_quota_project(...)`.

## Rebuild / redeploy

```bash
P=advisor-agent-2026; R=us-central1
IMG=$R-docker.pkg.dev/$P/agent-reports/web:$(git rev-parse --short HEAD)
gcloud builds submit . --project $P --region $R --config deploy/cloudbuild.web.yaml \
  --service-account=projects/$P/serviceAccounts/report-builder@$P.iam.gserviceaccount.com \
  --substitutions=_IMAGE=$IMG
gcloud run deploy gameday-web --project $P --region $R --image $IMG
```
