"""The open-source route: BGE, fine-tuned locally with sentence-transformers.

    python embed_tune/bge.py train                  # bge-small -> data/embed_tune/models/bge-small-tuned
    python embed_tune/bge.py eval                   # base BGE and tuned BGE on the test

BGE (BAAI General Embedding) is an open model anyone can download and run on
a plain CPU -- no endpoint, no per-call bill. bge-small-en-v1.5 is 33M
parameters and makes 384 numbers per text (text-embedding-005 makes 768).

Training: each pair is (question, passage that answers it). The loss is
MultipleNegativesRankingLoss: in a batch of 32 pairs, each question must score
its own passage above the 31 other passages in the batch. The other passages
are free "wrong answers" (in-batch negatives); no one labels them. Batches
never hold the same passage twice, or a question would be punished for
matching its own answer.

A slice of the training pairs is held out (never trained on) and scored before
and after: proof the model learned something general, measured without
touching the test. The test set is only read by `eval`.

BGE expects questions to carry an instruction prefix and passages not to; that
is kept in training and at search time.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from common import DATA, KEY, SNAPSHOT, load_window, unit
from eval import load_qrels, score, summary

BASE = "BAAI/bge-small-en-v1.5"
TUNED = DATA / "models" / "bge-small-tuned"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
PAIRS = KEY / "train_pairs.jsonl"
HOLDOUT = 200
MAX_LEN = 384          # tokens; ~95% of passages fit
EPOCHS, BATCH, LR = 1, 32, 2e-5
SEED = 11


def pairs() -> tuple[pd.DataFrame, pd.DataFrame]:
    snap = pd.read_parquet(SNAPSHOT, columns=["chunk_id", "published", "text"])
    snap["doc_id"] = snap.chunk_id + "@" + snap.published.dt.strftime("%Y-%m-%dT%H:%M")
    text = dict(zip(snap.doc_id, snap.text))
    p = pd.DataFrame([json.loads(line) for line in PAIRS.open(encoding="utf-8")])
    p["passage"] = p.doc_id.map(text)
    p = p.dropna(subset=["passage"]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return p.iloc[HOLDOUT:], p.iloc[:HOLDOUT]


def holdout_score(model, held: pd.DataFrame) -> dict:
    """Each held-out question against all 200 held-out passages: where does its own land?"""
    q = model.encode(held.question.tolist(), prompt=QUERY_PREFIX, normalize_embeddings=True)
    d = model.encode(held.passage.tolist(), normalize_embeddings=True)
    rank = (q @ d.T > np.diag(q @ d.T)[:, None]).sum(1)      # 0 = its passage came first
    return {"top1": float((rank == 0).mean()), "top5": float((rank < 5).mean()),
            "mrr": float((1 / (rank + 1)).mean())}


def train() -> None:
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    tr, held = pairs()
    model = SentenceTransformer(BASE, device="cpu")
    model.max_seq_length = MAX_LEN
    before = holdout_score(model, held)
    print(f"{len(tr)} training pairs, {len(held)} held out")
    print(f"held-out before: {before}")

    ckpt = DATA / "models" / "checkpoints"
    args = SentenceTransformerTrainingArguments(
        output_dir=str(ckpt), num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH, learning_rate=LR, warmup_ratio=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES, prompts={"anchor": QUERY_PREFIX},
        save_strategy="steps", save_steps=15, save_total_limit=1,
        logging_steps=10, report_to="none", seed=SEED, dataloader_pin_memory=False)
    ds = Dataset.from_pandas(tr[["question", "passage"]].rename(
        columns={"question": "anchor", "passage": "positive"}), preserve_index=False)
    # Cached ("GradCache") loss: the 32-pair batch, 31 negatives per question,
    # unchanged, but pushed through the model 8 texts at a time. The first run,
    # all 32 at once, ran the laptop out of memory at step 70 of 79.
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=8)
    resume = ckpt.exists() and any(ckpt.glob("checkpoint-*"))   # stopped midway: pick up there
    t0 = time.time()
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds,
                               loss=loss).train(resume_from_checkpoint=resume or None)
    print(f"trained in {(time.time() - t0) / 60:.1f} min")

    after = holdout_score(model, held)
    print(f"held-out after:  {after}")
    model.save(str(TUNED))
    (KEY / "bge_train.json").write_text(json.dumps({
        "base": BASE, "pairs": len(tr), "holdout": len(held), "epochs": EPOCHS,
        "batch": BATCH, "lr": LR, "max_len": MAX_LEN, "minutes": round((time.time() - t0) / 60, 1),
        "holdout_before": before, "holdout_after": after}, indent=1), encoding="utf-8")


def evaluate(name: str, path: str, df, qs, qrels) -> dict:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(path, device="cpu")
    model.max_seq_length = MAX_LEN
    cache = DATA / f"emb_{name}.npy"
    if cache.exists() and len(np.load(cache)) == len(df):
        docs = np.load(cache)
    else:
        t0 = time.time()
        docs = model.encode(df.text.tolist(), batch_size=16, normalize_embeddings=True,
                            show_progress_bar=True)
        np.save(cache, docs)
        print(f"embedded {len(df)} passages in {(time.time() - t0) / 60:.1f} min")
    qvec = model.encode([q["question"] for q in qs], prompt=QUERY_PREFIX,
                        normalize_embeddings=True)
    res = score(qs, qrels, unit(qvec), unit(docs), df)
    print(f"\n{name}, pure vector search")
    print(summary(res))
    with open(KEY / f"eval_{name}.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["train", "eval"])
    args = ap.parse_args()
    if args.step == "train":
        train()
        return
    qs, qrels = load_qrels(False)
    df = load_window()
    evaluate("bge_small", BASE, df, qs, qrels)
    if TUNED.exists():
        evaluate("bge_small_tuned", str(TUNED), df, qs, qrels)


if __name__ == "__main__":
    main()
