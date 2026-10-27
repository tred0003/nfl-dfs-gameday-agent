"""Hard negatives: for each training question, a passage that LOOKS like the
answer and is not.

    python embed_tune/mine.py       # key/train_pairs_v2.jsonl -> key/train_triplets.jsonl

In-batch negatives (the other 31 passages in a batch) are mostly easy: a
question about Puka Nacua's ankle against a Bears game recap. The model learns
little from beating them. A hard negative is the near miss -- another Rams
note, another ankle -- and that is the distinction search gets wrong.

Mined with today's model, text-embedding-005, whose passage vectors are
already in the snapshot: rank every training-eligible passage for the
question and take the highest one that is safe to call wrong:

  - not from the same article as the answer
  - scoring below 95% of the answer's own score. Anything closer may well be
    a second answer (another outlet's write-up of the same injury), and
    training the model to push a real answer away is worse than no negative.
    (The "positive-aware" filter from NVIDIA's NV-Retriever paper.)

Only training-eligible passages are searched (gen_pairs.candidates()), so no
test article becomes a negative either.
"""

from __future__ import annotations

import json

import numpy as np

from common import KEY, embed_queries, unit
from gen_pairs import OUT as PAIRS, candidates

OUT = KEY / "train_triplets.jsonl"
MARGIN = 0.95


def main() -> None:
    df = candidates()
    docs = unit(np.vstack(df.embedding.to_numpy()))
    article = df.chunk_id.str.split("#").str[0].to_numpy()
    row = {d: i for i, d in enumerate(df.doc_id)}
    pairs = [json.loads(line) for line in PAIRS.open(encoding="utf-8")]
    pairs = [p for p in pairs if p["doc_id"] in row]
    qvec = embed_queries([p["question"] for p in pairs])

    out, pos_rank = [], []
    for p, q in zip(pairs, qvec):
        sims = docs @ q
        i = row[p["doc_id"]]
        pos_rank.append(int((sims > sims[i]).sum()))
        ok = (article != article[i]) & (sims < MARGIN * sims[i])
        if not ok.any():
            continue
        j = int(np.argmax(np.where(ok, sims, -np.inf)))
        out.append({**p, "neg_id": df.doc_id[j]})
    with OUT.open("w", encoding="utf-8") as f:
        for t in out:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    r = np.array(pos_rank)
    print(f"{len(out)} triplets -> {OUT.name}")
    print(f"005 on the training questions: answer ranked first {np.mean(r == 0):.0%}, "
          f"top 10 {np.mean(r < 10):.0%}, median rank {int(np.median(r)) + 1}")


if __name__ == "__main__":
    main()
