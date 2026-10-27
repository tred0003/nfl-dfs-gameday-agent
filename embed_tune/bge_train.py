"""The BGE training loop on its own, so it runs the same on the laptop and in a
Vertex AI custom training job.

    python embed_tune/bge_train.py <data.jsonl> <out_dir> [--batch 32 --mini 8]

<data.jsonl> rows: {"anchor", "positive", "negative"?, "held": bool}, texts
already filled in (bge.py export), so this file needs nothing but the data:
no snapshot, no BigQuery, no project code. In the cloud job both paths are
under /gcs/<bucket>/..., which Vertex mounts as ordinary folders (Cloud
Storage FUSE).

Writes the model and train_record.json (settings, minutes, held-out scores)
to <out_dir>.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

BASE = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_LEN = 384          # tokens; ~95% of passages fit
EPOCHS, LR, SEED = 1, 2e-5, 11


def holdout_score(model, held: list[dict]) -> dict:
    """Each held-out question against all held-out passages: where does its own land?"""
    q = model.encode([h["anchor"] for h in held], prompt=QUERY_PREFIX, normalize_embeddings=True)
    d = model.encode([h["positive"] for h in held], normalize_embeddings=True)
    s = q @ d.T
    rank = (s > np.diag(s)[:, None]).sum(1)      # 0 = its passage came first
    return {"top1": float((rank == 0).mean()), "top5": float((rank < 5).mean()),
            "mrr": float((1 / (rank + 1)).mean())}


def main() -> None:
    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--mini", type=int, default=8)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.data).open(encoding="utf-8")]
    held = [r for r in rows if r["held"]]
    cols = ["anchor", "positive"] + (["negative"] if "negative" in rows[0] else [])
    tr = {c: [r[c] for r in rows if not r["held"]] for c in cols}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(tr['anchor'])} training rows ({', '.join(cols)}), {len(held)} held out, on {device}")

    model = SentenceTransformer(BASE, device=device)
    model.max_seq_length = MAX_LEN
    before = holdout_score(model, held)
    print(f"held-out before: {before}")

    out = Path(args.out)
    ckpt = out / "checkpoints"
    targs = SentenceTransformerTrainingArguments(
        output_dir=str(ckpt), num_train_epochs=EPOCHS,
        per_device_train_batch_size=args.batch, learning_rate=LR, warmup_ratio=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES, prompts={"anchor": QUERY_PREFIX},
        save_strategy="steps", save_steps=15, save_total_limit=1,
        logging_steps=10, report_to="none", seed=SEED, dataloader_pin_memory=False)
    # Cached ("GradCache") loss: the whole batch is the pool of negatives, but
    # texts go through the model `mini` at a time -- the laptop ran out of
    # memory doing all 32 at once. A third column adds one hard negative each.
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini)
    resume = ckpt.exists() and any(ckpt.glob("checkpoint-*"))   # stopped midway: pick up there
    t0 = time.time()
    SentenceTransformerTrainer(model=model, args=targs, train_dataset=Dataset.from_dict(tr),
                               loss=loss).train(resume_from_checkpoint=resume or None)
    minutes = round((time.time() - t0) / 60, 1)
    after = holdout_score(model, held)
    print(f"trained in {minutes} min\nheld-out after:  {after}")

    model.save(str(out / "model"))
    (out / "train_record.json").write_text(json.dumps({
        "base": BASE, "rows": len(tr["anchor"]), "columns": cols, "holdout": len(held),
        "epochs": EPOCHS, "batch": args.batch, "mini_batch": args.mini, "lr": LR,
        "max_len": MAX_LEN, "device": device, "minutes": minutes,
        "holdout_before": before, "holdout_after": after}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
