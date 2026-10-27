"""A bigger model instead of a tuned small one: gemini-embedding-001.

    python embed_tune/gemini_embed.py     # embeds the window (~3M tokens, ~$0.50), scores it

Fine-tuning BGE-small twice never beat BGE off the shelf, and both trailed
text-embedding-005 by ~0.09 nDCG -- the gap was the base model, not the
tuning. So the next question is the opposite one: does a stronger base model
help? gemini-embedding-001 is Google's embedding model built from Gemini,
pay-per-call like 005 (no endpoint to host), 3,072 numbers per text.

It is trained with Matryoshka representation learning: the first N numbers
are a usable embedding on their own. So the same vectors are also scored cut
to 768 (renormalized) -- the size of today's BigQuery column. If 768 holds
up, switching models would not change the table's shape.

Same task types as production: RETRIEVAL_DOCUMENT for passages,
RETRIEVAL_QUERY for questions.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import DATA, KEY, genai_client, load_window, unit
from eval import load_qrels, score, summary

MODEL = "gemini-embedding-001"
BATCH = 25
PART = 1000
CACHE = DATA / f"emb_{MODEL}.npy"


def embed(client, texts: list[str], task: str) -> np.ndarray:
    from google.genai import types
    batches = [texts[i:i + BATCH] for i in range(0, len(texts), BATCH)]

    def run(b):
        # The client retries HTTP 429/5xx; a dropped connection (the first full
        # run died on one, WinError 10053) raises instead, so retry it here.
        for attempt in range(6):
            try:
                r = client.models.embed_content(model=MODEL, contents=b,
                                                config=types.EmbedContentConfig(task_type=task))
                return [e.values for e in r.embeddings]
            except Exception:  # noqa: BLE001
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)

    with ThreadPoolExecutor(max_workers=6) as ex:
        return np.asarray([v for got in ex.map(run, batches) for v in got], dtype=np.float32)


def embed_passages(client, texts: list[str]) -> np.ndarray:
    """In parts of 1,000, each saved as it finishes, so a failure resumes."""
    parts = DATA / f"emb_{MODEL}_parts"
    parts.mkdir(exist_ok=True)
    out = []
    for i in range(0, len(texts), PART):
        f = parts / f"{i:06d}.npy"
        if not f.exists():
            np.save(f, embed(client, texts[i:i + PART], "RETRIEVAL_DOCUMENT"))
            print(f"  {min(i + PART, len(texts))}/{len(texts)}", flush=True)
        out.append(np.load(f))
    return np.vstack(out)


def main() -> None:
    qs, qrels = load_qrels(False)
    df = load_window()
    client = genai_client()
    if CACHE.exists() and len(np.load(CACHE)) == len(df):
        docs = np.load(CACHE)
    else:
        t0 = time.time()
        docs = embed_passages(client, df.text.tolist())
        np.save(CACHE, docs)
        print(f"embedded {len(df)} passages in {(time.time() - t0) / 60:.1f} min "
              f"({df.text.str.len().sum() / 4 / 1e6:.1f}M tokens, roughly)")
    qvec = embed(client, [q["question"] for q in qs], "RETRIEVAL_QUERY")

    for dims, name in ((3072, "gemini_emb"), (768, "gemini_emb_768")):
        res = score(qs, qrels, unit(qvec[:, :dims]), unit(docs[:, :dims]), df)
        print(f"\n{MODEL} at {dims} dims, pure vector search")
        print(summary(res))
        with open(KEY / f"eval_{name}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
