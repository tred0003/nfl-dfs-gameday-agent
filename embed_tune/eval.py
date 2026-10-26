"""Score a retriever on the test set: how good is search, as a number.

    python embed_tune/eval.py                 # the base model, text-embedding-005
    python embed_tune/eval.py --approved      # only questions the review approved

Three numbers per run, each averaged over questions:

  hit@6     did at least one passage that answers it (grade 2) land in the
            top 6? Six is what the archive search returns, so this is "would
            Ask have had something to work with".
  recall@10 of the passages that answer it, what share made the top 10
            (out of at most 10, so a question with 30 good passages is not
            impossible).
  nDCG@10   the standard ranking score (the one embedding benchmarks report):
            rewards a 2 more than a 1, and the top of the list more than the
            bottom. 1.0 is a perfect order.

Two modes, because production uses both:
  open      the whole window, no filter
  player    restricted to the question's player (only questions that name one),
            the filter the Ask agent and news-mcp apply

Passages the judge never saw count as not relevant. That is fine for the base
model, which helped build the pool; before a tuned model is compared, its own
top 10 must be added to the pool and judged (pool.py --add), or it gets no
credit for finding something new.

Pure vector search on purpose: no "two newest" slot, no keyword step. The
embedding model is the only thing that changes between runs.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from common import KEY, QUESTIONS, embed_queries, load_window, name_mask, unit

JUDGED = KEY / "judged.jsonl"
REVIEW = KEY / "review.json"
K_HIT, K = 6, 10


def ndcg(gains: list[int], ideal: list[int], k: int = K) -> float:
    disc = 1 / np.log2(np.arange(2, k + 2))
    dcg = sum((2 ** g - 1) * d for g, d in zip(gains[:k], disc))
    idcg = sum((2 ** g - 1) * d for g, d in zip(sorted(ideal, reverse=True)[:k], disc))
    return dcg / idcg if idcg else 0.0


def score(qs, qrels, qvec, docs, df) -> dict:
    ids = df.doc_id.to_numpy()
    out = {"open": [], "player": []}
    for i, q in enumerate(qs):
        rel = qrels[q["qid"]]
        if not any(g == 2 for g in rel.values()):
            continue
        sims = docs @ qvec[i]
        modes = [("open", sims)]
        if q["player"]:
            modes.append(("player", np.where(name_mask(df, q["player"]), sims, -np.inf)))
        for mode, s in modes:
            top = [ids[j] for j in np.argsort(-s)[:K] if np.isfinite(s[j])]
            gains = [rel.get(d, 0) for d in top]
            n2 = sum(g == 2 for g in rel.values())
            out[mode].append({
                "qid": q["qid"],
                "hit": float(any(g == 2 for g in gains[:K_HIT])),
                "recall": sum(g == 2 for g in gains) / min(n2, K),
                "ndcg": ndcg(gains, list(rel.values())),
            })
    return out


def summary(res: dict) -> str:
    lines = [f"{'mode':<8}{'questions':>10}{'hit@6':>8}{'recall@10':>11}{'nDCG@10':>9}"]
    for mode, rows in res.items():
        if rows:
            m = {k: np.mean([r[k] for r in rows]) for k in ("hit", "recall", "ndcg")}
            lines.append(f"{mode:<8}{len(rows):>10}{m['hit']:>8.3f}{m['recall']:>11.3f}{m['ndcg']:>9.3f}")
    return "\n".join(lines)


def load_qrels(approved_only: bool):
    qs = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8")]
    qrels = {}
    for line in JUDGED.open(encoding="utf-8"):
        r = json.loads(line)
        qrels[r["qid"]] = {g["doc_id"]: g["grade"] for g in r["grades"]}
    if REVIEW.exists():           # the person's fixes override the judge
        review = json.loads(REVIEW.read_text(encoding="utf-8"))
        for qid, fix in review.items():
            for doc_id, grade in fix.get("grades", {}).items():
                qrels.setdefault(qid, {})[doc_id] = grade
        if approved_only:
            qs = [q for q in qs if review.get(q["qid"], {}).get("status") == "ok"]
        qs = [q for q in qs if review.get(q["qid"], {}).get("status") != "cut"]
    return [q for q in qs if q["qid"] in qrels], qrels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--approved", action="store_true", help="only questions marked ok in review")
    ap.add_argument("--out", default=str(KEY / "eval_base.json"))
    args = ap.parse_args()

    qs, qrels = load_qrels(args.approved)
    df = load_window()
    docs = unit(np.vstack(df.embedding.to_numpy()))
    qvec = embed_queries([q["question"] for q in qs])
    res = score(qs, qrels, qvec, docs, df)
    print("base text-embedding-005, pure vector search")
    print(summary(res))
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
