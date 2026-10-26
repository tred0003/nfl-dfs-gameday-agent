"""Candidate passages for each test question, from three different retrievers.

    python embed_tune/pool.py        # writes embed_tune/key/pool.jsonl

The judge (judge.py) only grades what is in the pool, so a passage no
retriever surfaced can never be labeled relevant. If the pool came from the
base embedding model alone, every passage that model misses would be missing
from the answer key too, and the test would grade the base model against its
own blind spots. So the pool is the union of:

  vector       base text-embedding-005, whole window, top VEC_K
  vector+name  the same, restricted to the question's player (the production
               filter: player_keys or the name in the text), top NAME_K
  bm25         plain keyword ranking (no embeddings at all), top BM25_K
  seed         the passage the question was written from

Everything runs on the frozen snapshot (data/embed_tune/snapshot.parquet, built by snapshot.py,
published before CUTOFF), inside the production search window of WINDOW_DAYS,
so the answer key does not drift as the archive grows.
"""

from __future__ import annotations

import json
import re

import numpy as np

from common import KEY, QUESTIONS, embed_queries, load_window, name_mask, unit

OUT = KEY / "pool.jsonl"
VEC_K, NAME_K, BM25_K = 20, 15, 15

_TOKEN = re.compile(r"[a-z0-9']+")


def tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def main() -> None:
    df = load_window()
    qs = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8")]
    print(f"{len(df)} chunks in the window, {len(qs)} questions")

    docs = unit(np.vstack(df.embedding.to_numpy()))
    sims = embed_queries([q["question"] for q in qs]) @ docs.T

    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi([tokens(t) for t in df.text])

    ids = df.doc_id.to_numpy()
    with OUT.open("w", encoding="utf-8") as f:
        for i, q in enumerate(qs):
            found: dict[str, list[str]] = {}

            def add(idx, how):
                for j in idx:
                    found.setdefault(ids[j], []).append(how)

            add(np.argsort(-sims[i])[:VEC_K], "vector")
            if q["player"]:
                mask = name_mask(df, q["player"])
                masked = np.where(mask, sims[i], -np.inf)
                add([j for j in np.argsort(-masked)[:NAME_K] if mask[j]], "vector+name")
            add(np.argsort(-bm25.get_scores(tokens(q["question"])))[:BM25_K], "bm25")
            seeds = np.flatnonzero(df.chunk_id.to_numpy() == q["seed"])
            if len(seeds):
                add(seeds, "seed")
            else:
                print(f"  {q['qid']}: seed {q['seed']} is outside the window")
            f.write(json.dumps({"qid": q["qid"], "candidates": [
                {"doc_id": c, "from": sorted(set(how))} for c, how in found.items()]}) + "\n")
    sizes = [len(json.loads(l)["candidates"]) for l in OUT.open(encoding="utf-8")]
    print(f"pool: {sum(sizes)} candidates, {np.mean(sizes):.1f} per question -> {OUT}")


if __name__ == "__main__":
    main()
