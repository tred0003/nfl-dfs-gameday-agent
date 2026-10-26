"""Freeze the news archive for the search test.

    python embed_tune/snapshot.py      # writes data/embed_tune/snapshot.parquet (~120 MB)

Everything published before CUTOFF, with its stored text-embedding-005
vector. The test runs on this copy, never the live table: the archive grows
every day, and a new passage that answers an old question would change the
scores without anything about the model changing.

The file is not committed (data/ is ignored). Re-running this before
CUTOFF moves gives back the same rows, since nothing earlier is rewritten.
"""

from __future__ import annotations

from google.cloud import bigquery

from common import CUTOFF, DATA, SNAPSHOT

TABLE = "fantasy-football-ml-503718.dfs.news_chunks"


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    client = bigquery.Client(project=TABLE.split(".")[0])
    sql = f"""
    SELECT chunk_id, published, kind, source, headline, text, players, player_keys,
           teams, url, embedding
    FROM `{TABLE}`
    WHERE published < @cutoff AND screen = 'clean'"""
    job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("cutoff", "TIMESTAMP", CUTOFF.to_pydatetime())]))
    df = job.to_dataframe()
    df.to_parquet(SNAPSHOT)
    print(f"{len(df)} chunks, {df.published.min()} to {df.published.max()} -> {SNAPSHOT}")


if __name__ == "__main__":
    main()
