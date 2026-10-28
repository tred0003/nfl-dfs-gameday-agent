"""LlamaIndex's retriever against the shipped archive search, on the same gate.

    python embed_tune/llamaindex_gate.py            # build the LlamaIndex table, score both
    python embed_tune/llamaindex_gate.py --no-build # table already built

The question: what does the hand-built search in gameday_agent/archive.py buy
over what a framework gives out of the box? Both run against the same
BigQuery archive, same embeddings, same 150 questions, same frozen clock,
scored by gate.py's own code (gate.run takes the search as an argument).

  shipped     archive.search_news_archive: time window and player filter
              inside VECTOR_SEARCH, 2 newest matches guaranteed, 50 candidates
              re-ranked by the Vertex AI Ranking API, newest-of-the-relevant
              first, exact-text repeats dropped
  llamaindex  VectorStoreIndex over BigQueryVectorStore, as_retriever(top 6)
              with MetadataFilters for the time window and the player -- the
              framework's standard retriever, nothing added

BigQueryVectorStore owns its schema (node_id, text, metadata JSON,
embedding), so the archive is copied into that shape -- dfs.news_chunks_li,
clean rows only, the same embedding vectors -- rather than re-embedded.

Two bugs in llama-index-vector-stores-bigquery 0.3.0, both patched by
_patch_filters below (same clauses, named parameters):
  - EVERY filtered query fails. Filters are built with positional `?`
    parameters, the query itself uses named @top_k / @distance_type, and
    BigQuery rejects the mix: "NAMED query parameters cannot be used in
    POSITIONAL parameter mode". A MetadataFilter is unusable as shipped.
  - TEXT_MATCH puts its value into the SQL string unescaped
    (utils._build_filter_clause: LIKE '{value}'): a name like "Ja'Marr
    Chase" breaks the query, and user text there is a SQL injection.

One thing the framework could not express:
  - No "array contains" filter, so player_keys is stored as one string,
    "|kenneth walker|...|", and matched with TEXT_MATCH '%|key|%'. The
    shipped filter also accepts the full name anywhere in the text; that
    needs an OR across a metadata field and the text column, which
    MetadataFilters cannot say.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import timedelta

import pandas as pd

import gate
from common import CUTOFF, EMBED_PROJECT, KEY, WINDOW_DAYS

PROJECT, DATASET, TABLE = "fantasy-football-ml-503718", "dfs", "news_chunks_li"
OUT = KEY / "llamaindex_gate.json"
FMT = "%Y-%m-%dT%H:%M:%S"

BUILD = f"""
CREATE OR REPLACE TABLE `{PROJECT}.{DATASET}.{TABLE}` AS
SELECT CONCAT(chunk_id, '@', FORMAT_TIMESTAMP('{FMT}', published)) AS node_id,
       text,
       TO_JSON(STRUCT(chunk_id, FORMAT_TIMESTAMP('{FMT}', published) AS published,
                      CONCAT('|', ARRAY_TO_STRING(player_keys, '|'), '|') AS players)) AS metadata,
       embedding
FROM `{PROJECT}.{DATASET}.news_chunks`
WHERE screen = 'clean'
"""


def _patch_filters() -> None:
    """Named parameters for MetadataFilter clauses (see the module docstring).
    Same clauses as utils._build_filter_clause for the operators used here."""
    from itertools import count

    import llama_index.vector_stores.bigquery.utils as u
    from google.cloud import bigquery
    from llama_index.core.vector_stores import FilterOperator

    n = count()
    ops = {FilterOperator.LT: "<", FilterOperator.GTE: ">=", FilterOperator.GT: ">",
           FilterOperator.LTE: "<=", FilterOperator.EQ: "=", FilterOperator.TEXT_MATCH: "LIKE"}

    def clause(f):
        if f.operator not in ops:
            raise ValueError(f"operator {f.operator} not patched")
        name = f"f{next(n)}"
        return (f"SAFE.JSON_VALUE(metadata, '$.\"{f.key}\"') {ops[f.operator]} @{name}",
                [bigquery.ScalarQueryParameter(name, "STRING", f.value)])

    u._build_filter_clause = clause


def retriever_search():
    from llama_index.core import VectorStoreIndex
    from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
    from llama_index.embeddings.google_genai.base import VertexAIConfig
    from llama_index.vector_stores.bigquery import BigQueryVectorStore

    from gameday_agent.archive import _when, name_keys

    # metadata_dict_to_node warns on every row: these rows were not written
    # by LlamaIndex, so it falls back to a plain TextNode. Expected, and loud.
    logging.getLogger("llama_index.vector_stores.bigquery").setLevel(logging.ERROR)
    _patch_filters()
    store = BigQueryVectorStore(table_id=TABLE, dataset_id=DATASET, project_id=PROJECT,
                                distance_type="COSINE")
    index = VectorStoreIndex.from_vector_store(store, embed_model=GoogleGenAIEmbedding(
        model_name="text-embedding-005",
        vertexai_config=VertexAIConfig(project=EMBED_PROJECT, location="us-central1")))
    cutoff = CUTOFF.to_pydatetime()
    window = [MetadataFilter(key="published", operator=FilterOperator.LT, value=cutoff.strftime(FMT)),
              MetadataFilter(key="published", operator=FilterOperator.GTE,
                             value=(cutoff - timedelta(days=WINDOW_DAYS)).strftime(FMT))]

    def search(question: str, player: str) -> dict:
        filters = list(window)
        if player:
            key = (name_keys([player]) or [""])[0]
            filters.append(MetadataFilter(key="players", operator=FilterOperator.TEXT_MATCH,
                                          value=f"%|{key}|%"))
        nodes = index.as_retriever(similarity_top_k=6,
                                   filters=MetadataFilters(filters=filters)).retrieve(question)
        return {"passages": [{"id": n.node.metadata["chunk_id"],
                              **_when(pd.Timestamp(n.node.metadata["published"], tz="UTC"), cutoff)}
                             for n in nodes],
                "meta": {"ranked": None}}

    return search


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()
    if not args.no_build:
        from google.cloud import bigquery
        bigquery.Client(project=PROJECT).query_and_wait(BUILD)
        print(f"built {PROJECT}.{DATASET}.{TABLE}")
    kept = gate.LAST.read_text(encoding="utf-8")       # gate.run overwrites the shipped search's file
    try:
        res = gate.run(retriever_search())
        rows = json.loads(gate.LAST.read_text(encoding="utf-8"))
    finally:
        gate.LAST.write_text(kept, encoding="utf-8")
    OUT.write_text(json.dumps({"summary": res, "per_question": rows}, indent=1), encoding="utf-8")
    base = json.loads(gate.BASELINE.read_text(encoding="utf-8"))
    print(f"\n{'retriever':<12}{'hit@6':>8}{'newest@6':>10}")
    print(f"{'shipped':<12}{base['hit_at_6']:>8.3f}{base['newest_at_6']:>10.3f}   (key/gate_baseline.json)")
    print(f"{'llamaindex':<12}{res['hit_at_6']:>8.3f}{res['newest_at_6']:>10.3f}")
    shipped = {r["qid"]: r for r in json.loads(kept)}
    for k in ("hit", "newest"):
        pairs = [(shipped[r["qid"]][k], r[k]) for r in rows
                 if r["qid"] in shipped and r[k] is not None and shipped[r["qid"]][k] is not None]
        print(f"{k}: llamaindex better on {sum(b and not a for a, b in pairs)}, "
              f"worse on {sum(a and not b for a, b in pairs)} of {len(pairs)} questions")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
