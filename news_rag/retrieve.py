"""Point-in-time retrieval over dfs.news_chunks.

The filter runs INSIDE VECTOR_SEARCH, on the base table, before the nearest
neighbours are picked. Filtering after top-k would be the classic mistake:
ask for 8, have 6 of them published after kickoff, and get 2 back -- or, with
no filter at all, get the game recap and "predict" what already happened.

Two modes, kept side by side on purpose because the difference is the lesson:

  vector only   nearest chunks by meaning, inside the time window. Finds
                anything that reads like the question -- including news about
                a different Wilson.
  filtered      the same search restricted to chunks tagged with the player,
                or whose text names him in full. Metadata does the precision,
                the vector does the ranking.
"""

from __future__ import annotations

from datetime import datetime

from google.cloud import bigquery

from news_rag.index import BQ_PROJECT, TABLE, _client, embed

_bq = None
_genai = None

SQL = f"""
SELECT base.chunk_id, base.published, base.kind, base.source, base.headline,
       base.text, base.players, base.teams, base.url, distance
FROM VECTOR_SEARCH(
  (SELECT * FROM `{TABLE}`
   WHERE published < @cutoff
     AND published >= TIMESTAMP_SUB(@cutoff, INTERVAL @days DAY)
     {{extra}}),
  'embedding',
  (SELECT @q AS embedding),
  top_k => @k, distance_type => 'COSINE')
ORDER BY distance
"""

PLAYER = "AND (@name IN UNNEST(players) OR STRPOS(text, @name) > 0)"
TEAMS = "AND EXISTS (SELECT 1 FROM UNNEST(teams) t WHERE t IN UNNEST(@teams))"


def search(query: str, cutoff: datetime, *, name: str | None = None,
           teams: list[str] | None = None, days: int = 21, k: int = 8) -> list[dict]:
    global _bq, _genai
    _bq = _bq or bigquery.Client(project=BQ_PROJECT)
    _genai = _genai or _client()
    vec = embed([query], task="RETRIEVAL_QUERY", client=_genai)[0]

    extra = (PLAYER if name else "") + (TEAMS if teams else "")
    params = [
        bigquery.ScalarQueryParameter("cutoff", "TIMESTAMP", cutoff),
        bigquery.ScalarQueryParameter("days", "INT64", days),
        bigquery.ScalarQueryParameter("k", "INT64", k),
        bigquery.ArrayQueryParameter("q", "FLOAT64", vec),
    ]
    if name:
        params.append(bigquery.ScalarQueryParameter("name", "STRING", name))
    if teams:
        params.append(bigquery.ArrayQueryParameter("teams", "STRING", teams))
    job = _bq.query(SQL.format(extra=extra), job_config=bigquery.QueryJobConfig(query_parameters=params))
    return [dict(r) for r in job.result()]


LATEST = f"""
SELECT chunk_id, published, kind, source, headline, text, players, teams, url,
       CAST(NULL AS FLOAT64) AS distance
FROM `{TABLE}`
WHERE published < @cutoff
  AND published >= TIMESTAMP_SUB(@cutoff, INTERVAL @days DAY)
  {{extra}}
ORDER BY published DESC
LIMIT @n
"""


def hybrid(query: str, cutoff: datetime, *, name: str | None = None,
           teams: list[str] | None = None, days: int = 21, k: int = 8,
           recent: int = 3) -> list[dict]:
    """The `recent` newest chunks, then the best semantic matches up to k.

    Similarity alone missed the most important Week 5 item in the archive:
    Sunday 11:57 ET, "Warren will remain the Steelers' unquestioned lead back
    ... Rico Dowdle (toe) declared inactive". Embeddings encode meaning, not
    time, so an older "Warren is expected to play" scored closer to a status
    question. For news, newest-first is a ranking signal the vector cannot
    supply, so it gets a guaranteed slot.
    """
    global _bq
    _bq = _bq or bigquery.Client(project=BQ_PROJECT)
    extra = (PLAYER if name else "") + (TEAMS if teams else "")
    params = [
        bigquery.ScalarQueryParameter("cutoff", "TIMESTAMP", cutoff),
        bigquery.ScalarQueryParameter("days", "INT64", days),
        bigquery.ScalarQueryParameter("n", "INT64", recent),
    ]
    if name:
        params.append(bigquery.ScalarQueryParameter("name", "STRING", name))
    if teams:
        params.append(bigquery.ArrayQueryParameter("teams", "STRING", teams))
    newest = [dict(r) for r in _bq.query(LATEST.format(extra=extra),
                                         job_config=bigquery.QueryJobConfig(query_parameters=params)).result()]
    seen = {h["chunk_id"] for h in newest}
    similar = [h for h in search(query, cutoff, name=name, teams=teams, days=days, k=k)
               if h["chunk_id"] not in seen]
    return sorted(newest + similar[:k - len(newest)], key=lambda h: h["published"], reverse=True)
