"""The open-source route: BGE, fine-tuned with sentence-transformers.

    python embed_tune/bge.py train [--data v1|v2]   # on this laptop (CPU)
    python embed_tune/bge.py cloud [--data v1|v2]   # submit a Vertex AI custom job (16 CPUs; --gpu: Spot T4)
    python embed_tune/bge.py fetch [--data v1|v2]   # once it succeeds: download the model
    python embed_tune/bge.py eval                   # base BGE and every tuned BGE on the test

BGE (BAAI General Embedding) is an open model anyone can download and run on
a plain CPU -- no endpoint, no per-call bill. bge-small-en-v1.5 is 33M
parameters and makes 384 numbers per text (text-embedding-005 makes 768).

Training (bge_train.py): each pair is (question, passage that answers it). The
loss is MultipleNegativesRankingLoss: in a batch of 32 pairs, each question
must score its own passage above the 31 other passages in the batch. The
other passages are free "wrong answers" (in-batch negatives); no one labels
them. Batches never hold the same passage twice, or a question would be
punished for matching its own answer. v2 adds one mined hard negative per
question (mine.py).

A slice of the training pairs is held out (never trained on) and scored before
and after: proof the model learned something general, measured without
touching the test. The test set is only read by `eval`.

Why `cloud`: v2 (a third passage per row) ran the laptop out of memory twice,
at 65 s a step. The same bge_train.py runs as a Vertex AI custom training job
on Google's machines; data and model go through Cloud Storage. Its images
are Python 3.10, too old for sentence-transformers 6, so the job pins 3.4 --
same loss, same settings.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time

import numpy as np
import pandas as pd

from common import DATA, KEY, SNAPSHOT, load_window, unit
from eval import load_qrels, score, summary

BASE = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MAX_LEN = 384
# v1: gen_pairs' first questions, in-batch negatives only (no gain on the test).
# v2: styled questions plus one mined hard negative each (mine.py).
DATASETS = {"v1": (KEY / "train_pairs.jsonl", "bge-small-tuned"),
            "v2": (KEY / "train_triplets.jsonl", "bge-small-tuned-v2")}
HOLDOUT = 200
SEED = 11

PROJECT = "fantasy-football-ml-503718"
BUCKET = "fantasy-football-ml-503718-vertex"
GPU_CONTAINER = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest"
# The prebuilt PyTorch training images refuse to start without a GPU, and the
# CPU one stops at PyTorch 1.4; on CPU the job uses Google's plain Deep
# Learning base image and installs PyTorch itself.
CPU_CONTAINER = "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/base-cpu.py310:latest"
REQUIREMENTS = ["sentence-transformers==3.4.1", "transformers==4.46.3",
                "datasets==3.2.0", "accelerate==1.2.1"]
CPU_REQUIREMENTS = ["torch==2.4.1"] + REQUIREMENTS


def export(data: str) -> "Path":
    """The training file bge_train.py reads: texts filled in, held-out rows marked."""
    path, _ = DATASETS[data]
    snap = pd.read_parquet(SNAPSHOT, columns=["chunk_id", "published", "text"])
    snap["doc_id"] = snap.chunk_id + "@" + snap.published.dt.strftime("%Y-%m-%dT%H:%M")
    text = dict(zip(snap.doc_id, snap.text))
    p = pd.DataFrame([json.loads(line) for line in path.open(encoding="utf-8")])
    p["positive"] = p.doc_id.map(text)
    if "neg_id" in p:
        p["negative"] = p.neg_id.map(text)
    p = p.dropna(subset=["positive"]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    p["held"] = p.index < HOLDOUT
    out = DATA / f"train_{data}.jsonl"
    cols = ["question", "positive", "held"] + (["negative"] if "negative" in p else [])
    p[cols].rename(columns={"question": "anchor"}).to_json(
        out, orient="records", lines=True, force_ascii=False)
    print(f"{len(p)} rows ({HOLDOUT} held out) -> {out}")
    return out


def train_local(data: str) -> None:
    src = export(data)
    out = DATA / "models" / DATASETS[data][1]
    subprocess.run(["python", str(KEY.parent / "bge_train.py"), str(src), str(out)], check=True)


def train_cloud(data: str, gpu: bool) -> None:
    """Submit and return; `fetch` downloads the model once the job is done.

    Default machine: c2-standard-16, 16 compute-optimized CPUs, on demand.
    --gpu: one Spot T4. The project has no on-demand GPU quota (0 for every
    type) but 1 Spot T4 -- spare capacity, cheaper, can be taken back mid-job
    (bge_train.py checkpoints every 15 steps and resumes). The first try sat
    1 h 40 min in us-central1 on "resources are insufficient" and never
    started, so CPU is the default: slower per step, but it actually runs."""
    from google.cloud import aiplatform
    from google.cloud.aiplatform.compat.types import custom_job
    src = export(data)
    run = f"embed_tune/runs/{DATASETS[data][1]}-{time.strftime('%Y%m%d-%H%M%S')}"
    subprocess.run(["gcloud", "storage", "cp", str(src), f"gs://{BUCKET}/{run}/{src.name}"],
                   check=True, shell=True)
    aiplatform.init(project=PROJECT, location="us-central1", staging_bucket=f"gs://{BUCKET}")
    machine = (dict(machine_type="n1-standard-8", accelerator_type="NVIDIA_TESLA_T4",
                    accelerator_count=1) if gpu else dict(machine_type="c2-standard-16"))
    job = aiplatform.CustomJob.from_local_script(
        display_name=f"bge-tune-{data}", script_path=str(KEY.parent / "bge_train.py"),
        container_uri=GPU_CONTAINER if gpu else CPU_CONTAINER,
        requirements=REQUIREMENTS if gpu else CPU_REQUIREMENTS,
        args=[f"/gcs/{BUCKET}/{run}/{src.name}", f"/gcs/{BUCKET}/{run}"], **machine)
    strategy = custom_job.Scheduling.Strategy.SPOT if gpu else None
    job.submit(scheduling_strategy=strategy, restart_job_on_worker_restart=gpu)
    job.wait_for_resource_creation()
    (DATA / f"cloud_run_{data}.json").write_text(
        json.dumps({"job": job.resource_name, "run": run}), encoding="utf-8")
    print(f"submitted {job.resource_name}\nwhen it succeeds: python embed_tune/bge.py fetch --data {data}")


def fetch(data: str) -> None:
    from google.cloud import aiplatform
    info = json.loads((DATA / f"cloud_run_{data}.json").read_text(encoding="utf-8"))
    state = aiplatform.CustomJob.get(info["job"], project=PROJECT, location="us-central1").state.name
    if state != "JOB_STATE_SUCCEEDED":
        raise SystemExit(f"{info['job']} is {state}")
    run = info["run"]
    local = DATA / "models" / DATASETS[data][1]
    local.mkdir(parents=True, exist_ok=True)
    for part in ("model", "train_record.json"):
        subprocess.run(["gcloud", "storage", "cp", "-r", f"gs://{BUCKET}/{run}/{part}", str(local)],
                       check=True, shell=True)
    rec = (local / "train_record.json").read_text(encoding="utf-8")
    (KEY / f"bge_train_{data}.json").write_text(rec, encoding="utf-8")
    print(rec)


def model_path(d) -> str:
    """bge_train.py saves to <dir>/model; the first (v1) run saved to <dir>."""
    return str(d / "model" if (d / "model").exists() else d)


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
    ap.add_argument("step", choices=["train", "cloud", "fetch", "eval"])
    ap.add_argument("--data", choices=list(DATASETS), default="v2")
    ap.add_argument("--gpu", action="store_true", help="cloud: one Spot T4 instead of 16 CPUs")
    args = ap.parse_args()
    if args.step == "train":
        train_local(args.data)
    elif args.step == "cloud":
        train_cloud(args.data, args.gpu)
    elif args.step == "fetch":
        fetch(args.data)
    else:
        qs, qrels = load_qrels(False)
        df = load_window()
        evaluate("bge_small", BASE, df, qs, qrels)
        for _, name in DATASETS.values():
            if (DATA / "models" / name).exists():
                evaluate(name.replace("-", "_"), model_path(DATA / "models" / name),
                         df, qs, qrels)


if __name__ == "__main__":
    main()
