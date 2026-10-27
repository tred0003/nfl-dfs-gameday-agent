# embed_tune: measuring (then tuning) the archive search

Goal: fine-tune the embedding model behind the news-archive search and prove
whether it helps. Step 1, done 2026-10-26, is the yardstick: a test set and a
baseline score for today's model, `text-embedding-005`.

## Run order

```
python embed_tune/snapshot.py          # freeze the archive (data/, not committed)
python embed_tune/write_questions.py   # 154 questions -> questions.jsonl
python embed_tune/pool.py              # ~31 candidates/question from 3 retrievers
python embed_tune/judge.py             # Gemini 2.5 Pro grades each 0/1/2 (~$7)
python embed_tune/crosscheck.py <dir>  # Claude agents' verdicts -> key/review.json
python embed_tune/eval.py              # hit@6, recall@10, nDCG@10
python embed_tune/gen_pairs.py         # training pairs (Gemini Flash), no test articles
```

## Baseline (answers both Gemini and Claude agree on)

Claude agents re-read all 1,613 judge answers blind: 85.1% agreement, 241
dropped to grade 1, 4 questions cut (key/crosscheck.json). `review_app.py`
builds a click-through page for a human pass, if one is wanted later.

| mode   | questions | hit@6 | recall@10 | nDCG@10 |
|--------|-----------|-------|-----------|---------|
| open   | 150       | 0.973 | 0.685     | 0.732   |
| player | 110       | 0.991 | 0.699     | 0.756   |

Training pairs: 2,699 (key/train_pairs.jsonl), none from any article in the
test pool.

Flattering to the base model: it built about half the candidate pool, so its
blind spots are partly missing from the answer key. A tuned model's own top
10 must be pooled and judged before the two are compared.

## BGE (open source, fine-tuned on a laptop CPU)

```
python embed_tune/bge.py train                 # bge-small-en-v1.5, 32 min on CPU
python embed_tune/bge.py eval                  # base + tuned BGE -> key/eval_bge_*.json
python embed_tune/pool_add.py key/eval_bge_small.json key/eval_bge_small_tuned.json
python embed_tune/pool_add.py --merge <dir>    # Claude's verdicts on the new 2s
python embed_tune/eval.py                      # re-score 005 on the bigger key too
```

Fair pool: BGE's top 10s held 563 passages the key had never seen; Gemini Pro
graded them (106 answers), Claude rejected 26 of those (75.5% agreement).
All three models scored on the same enlarged key, open mode, 150 questions:

| model                     | hit@6 | recall@10 | nDCG@10 |
|---------------------------|-------|-----------|---------|
| text-embedding-005 (base) | 0.973 | 0.668     | 0.722   |
| bge-small-en-v1.5         | 0.927 | 0.580     | 0.634   |
| bge-small, fine-tuned     | 0.880 | 0.543     | 0.619   |

Tuning helped on its own kind of question (200 held-out training pairs: top-1
0.77 -> 0.79, MRR 0.851 -> 0.876) and did not carry over to the test: 68
questions better, 73 worse. Role questions rose (0.619 -> 0.657), injury fell
(0.638 -> 0.601). Likely cause: the training questions are short and
templated (6.7 words; "What is", "How did", "Who is" open 27% of them),
the test's are longer and varied (8.6 words) -- the model learned Gemini's
phrasing. The different-writer rule is what exposed it.

## v2: better data, trained on Vertex AI

```
python embed_tune/gen_pairs.py                 # styled questions -> key/train_pairs_v2.jsonl
python embed_tune/mine.py                      # + one hard negative each -> key/train_triplets.jsonl
python embed_tune/bge.py cloud --data v2       # Vertex AI custom job, c2-standard-16
python embed_tune/bge.py fetch --data v2       # download the model when it succeeds
```

- Questions: each passage gets an assigned style (worried manager, why/how
  bad, start/sit, terse query, last name only, team-level). 3,240 pairs,
  10.5 words on average (v1: 6.7), sd 4.0 (v1: 1.7).
- Hard negatives: the highest-ranked passage under text-embedding-005 that is
  from a different article and scores below 95% of the answer (positive-aware
  filter, so a second real answer is not used as a "wrong" one).
- Where it ran: the laptop ran out of memory twice (65 s a step). Vertex
  custom job: the project has 0 on-demand GPU quota and its 1 Spot T4 sat
  1 h 40 min on "resources are insufficient", so 16 CPUs, 17 min, under $1.
- Fair pool again: 86 more unseen passages, 12 new answers, 6 rejected by
  the cross-check (all rounds: 118 new answers, 72.9% agreement).

Final, open mode, 150 questions, one enlarged answer key:

| model                       | hit@6 | recall@10 | nDCG@10 |
|-----------------------------|-------|-----------|---------|
| text-embedding-005 (base)   | 0.973 | 0.667     | 0.721   |
| bge-small-en-v1.5           | 0.927 | 0.579     | 0.634   |
| bge-small tuned, v1 data    | 0.880 | 0.542     | 0.618   |
| bge-small tuned, v2 data    | 0.887 | 0.569     | 0.631   |

v2 recovered most of v1's loss and ties untuned BGE: nDCG -0.003, 95%
bootstrap interval -0.027 to +0.022, 66 questions better and 69 worse. It
gains on role (+0.040) and team (+0.033) questions and loses on injury
(-0.036), the biggest group. Held-out training pairs: top-1 0.825 -> 0.865.
Hypothesis for injury, untested: injury news is time-sensitive, and an older
note on the same player's injury is a likely "hard negative" that is close
to a real answer.

Conclusion: on this archive, fine-tuning a small open model does not beat
it off the shelf, and neither comes near text-embedding-005 (0.721).

## Next

1. Tune `text-embedding-005` on Vertex AI (paid GPU job + temporary endpoint)
   with the v2 pairs -- needs GPU quota first (0 on demand).
