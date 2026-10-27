"""Every run against today's search (base), question by question.

    python embed_tune/compare.py      # after eval.py --rescore

One average hides a lot: v2 BGE "lost 0.003" while 66 questions got better
and 69 worse. For each run this prints the nDCG@10 change against base with a
95% bootstrap interval (resample the 150 questions 5,000 times), how many
questions moved each way, and the change by kind of question.
"""

from __future__ import annotations

import collections
import json

import numpy as np

from common import KEY, QUESTIONS

BASE = "base"


def main() -> None:
    kind = {q["qid"]: q["kind"] for q in map(json.loads, QUESTIONS.open(encoding="utf-8"))}
    runs = {f.stem.removeprefix("eval_"): {r["qid"]: r["ndcg"] for r in json.loads(
                f.read_text(encoding="utf-8"))["open"]}
            for f in sorted(KEY.glob("eval_*.json")) if "crosschecked" not in f.stem}
    base = runs.pop(BASE)
    qids = sorted(base)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(qids), size=(5000, len(qids)))
    kinds = sorted({kind[q] for q in qids})

    print(f"open mode, {len(qids)} questions, nDCG@10 vs base ({np.mean(list(base.values())):.3f})\n")
    print(f"{'run':<22}{'nDCG':>7}{'change':>8}{'95% interval':>18}{'up/down':>9}  "
          + "".join(f"{k[:6]:>8}" for k in kinds))
    order = sorted(runs, key=lambda n: -np.mean(list(runs[n].values())))
    for name in order:
        d = np.array([runs[name][q] - base[q] for q in qids])
        lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
        by = collections.defaultdict(list)
        for q, x in zip(qids, d):
            by[kind[q]].append(x)
        print(f"{name:<22}{np.mean([runs[name][q] for q in qids]):>7.3f}{d.mean():>+8.3f}"
              f"{f'{lo:+.3f}..{hi:+.3f}':>18}{f'{(d > .01).sum()}/{(d < -.01).sum()}':>9}  "
              + "".join(f"{np.mean(by[k]):>+8.3f}" for k in kinds))
    print("\nquestions per kind: " + ", ".join(
        f"{k} {sum(kind[q] == k for q in qids)}" for k in kinds))


if __name__ == "__main__":
    main()
