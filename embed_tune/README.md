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
python embed_tune/packet.py            # review-packet.pdf + review.txt
python embed_tune/packet.py --import   # your marks -> key/review.json
python embed_tune/eval.py              # hit@6, recall@10, nDCG@10
```

## Baseline (judge's labels, before human review)

| mode   | questions | hit@6 | recall@10 | nDCG@10 |
|--------|-----------|-------|-----------|---------|
| open   | 153       | 0.974 | 0.678     | 0.736   |
| player | 113       | 0.982 | 0.680     | 0.758   |

Flattering to the base model: it built about half the candidate pool, so its
blind spots are partly missing from the answer key. A tuned model's own top
10 must be pooled and judged before the two are compared.

## Next

1. Human review of the answer key (`review-packet.pdf`).
2. Pool + judge each new model's top 10 before comparing.
3. Training pairs: Gemini writes questions for archive passages, never for test passages.
4. Tune `text-embedding-005` on Vertex AI; fine-tune an open model (BGE) with
   sentence-transformers. Compare all three on this test.
