"""Chunk-size sweep: was 1,200 characters a good choice? Scored without re-judging.

    python embed_tune/chunk_size.py                 # every config in CONFIGS
    python embed_tune/chunk_size.py --sizes 400 2000

news_rag/index.py cuts stories, headline news, previews and recaps into
~1,200-character chunks (player blurbs stay whole). The number was picked,
never tested. Two problems with testing it, and what this does about each:

1. The answer key grades CHUNKS (story-49964107#0@...). New sizes, new ids --
   the key no longer points at anything, and re-judging is ~$7 per size.
   So the key is moved from ids to TEXT SPANS: every chunking is cut from the
   same article body, each graded chunk covers a known character range of it,
   and a new chunk is scored by how it overlaps the graded ranges.

     strict   the new chunk CONTAINS >= 90% of a grade-2 range. Exact for
              chunks at least as big as the key's (1,200+): if it holds the
              whole graded passage, it holds the answer.
     loose    the new chunk overlaps a grade-2 range by >= OVERLAP_MIN chars.
              An UPPER bound for small chunks: a 400-char piece of a relevant
              1,200-char passage may or may not be the piece with the answer.
   A small size whose loose (best case) score is below 1,200's strict score
   loses without any judging. If a small size looks close, its returned
   chunks have to be judged before it can win.

2. Smaller chunks mean more of them fit in a prompt; "top 6" would hand a
   2,000-char config 2.5x the text of a 800-char one. So every config gets the
   same CONTEXT BUDGET -- 6 x 1,200 = 7,200 characters, what Ask gets today --
   and returns as many chunks as fit. Overlapping chunks pay for their
   repeated text out of the same budget.

Metrics, per question, averaged:
  hit       a grade-2 range is covered (strict or loose) within the budget
  recall    share of the question's grade-2 ranges covered (out of at most 10)
  first     characters of context before the first covering chunk -- how much
            the model reads before it reaches an answer (misses count as budget)

Pure vector search, both modes (open, player filter), like eval.py: the
chunking is the only thing that changes. Blurbs keep their existing
embeddings; story chunks are embedded with text-embedding-005 and cached on
text hash, so 1,200/no-overlap costs nothing (its chunks ARE today's) and
doubles as the sanity check -- it must reproduce eval.py's id-based numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from common import CUTOFF, DATA, KEY, SNAPSHOT, WINDOW_DAYS, genai_client, embed_queries, name_mask, unit
from eval import load_qrels

CACHE = DATA / "chunk_embed_cache.parquet"
OUT = KEY / "chunk_size.json"
BUDGET = 7_200
STRICT, OVERLAP_MIN = 0.9, 200
MAX_REL = 10
SAVE_EVERY = 100
# (target characters, overlap as a share of the target)
CONFIGS = [(400, 0), (800, 0), (1200, 0), (2000, 0), (400, 0.15), (800, 0.15), (1200, 0.15), (2000, 0.15)]


def _sentences(text: str) -> list[str]:
    """news_rag/index.py's sentence rule (kept identical so 1,200/0 reproduces it)."""
    return re.split(r"(?<=[.!?\"”])\s+(?=[A-Z\"“])", text)


def versions() -> pd.DataFrame:
    """Every chunked article version in the search window, rebuilt as one body,
    with each original chunk's character range in it. From the full snapshot,
    not load_window: its exact-text de-dupe drops chunks of older versions."""
    df = pd.read_parquet(SNAPSHOT, columns=["chunk_id", "published", "kind", "headline", "text",
                                            "player_keys"])
    df = df[(df.published >= CUTOFF - pd.Timedelta(days=WINDOW_DAYS)) & (df.kind != "player_blurb")]
    df = df.drop_duplicates(["chunk_id", "published"])
    df["art"] = df.chunk_id.str.split("#").str[0]
    df["idx"] = df.chunk_id.str.split("#").str[1].astype(int)
    out = []
    for (art, pub), g in df.sort_values("idx").groupby(["art", "published"]):
        if list(g.idx) != list(range(len(g))):
            continue                                    # a gap: can't place the ranges
        head = g.headline.iloc[0]
        parts = [t[len(head) + 2:] for t in g.text]     # strip "headline. "
        spans, pos = {}, 0
        for i, p in zip(g.idx, parts):
            spans[f"{art}#{i}@{pub:%Y-%m-%dT%H:%M}"] = (pos, pos + len(p))
            pos += len(p) + 1
        out.append({"art": art, "published": pub, "kind": g.kind.iloc[0], "headline": head,
                    "body": " ".join(parts), "spans": spans, "player_keys": g.player_keys.iloc[0]})
    return pd.DataFrame(out)


def rechunk(body: str, size: int, overlap: float) -> list[tuple[int, int]]:
    """Sentence-packed ranges of ~size chars; with overlap, each chunk starts
    with the previous one's last sentences, up to overlap*size chars."""
    sents, pos = [], 0
    for s in _sentences(body):
        start = body.index(s, pos)
        sents.append((start, start + len(s)))
        pos = start + len(s)
    out, cur = [], []
    for a, b in sents:
        # index.py's test, len(cur) + len(s) > size, where cur and s are joined by a space
        if cur and b - cur[0][0] - 1 > size:
            out.append((cur[0][0], cur[-1][1]))
            keep = []
            for s in reversed(cur):
                if overlap and out[-1][1] - s[0] <= overlap * size:
                    keep.insert(0, s)
                else:
                    break
            cur = keep if len(keep) < len(cur) else []
        cur.append((a, b))
    if cur:
        out.append((cur[0][0], cur[-1][1]))
    return out


def corpus(vers: pd.DataFrame, blurbs: pd.DataFrame, size: int, overlap: float) -> pd.DataFrame:
    rows = []
    for v in vers.itertuples():
        for k, (a, b) in enumerate(rechunk(v.body, size, overlap)):
            rows.append({"doc": f"{v.art}@{v.published:%Y-%m-%dT%H:%M}#{k}", "ver": f"{v.art}@{v.published:%Y-%m-%dT%H:%M}",
                         "a": a, "b": b, "published": v.published, "player_keys": v.player_keys,
                         "text": f"{v.headline}. {v.body[a:b]}"})
    st = pd.DataFrame(rows)
    out = pd.concat([st, blurbs], ignore_index=True)
    # the archive's rule: exact-text repeats keep only the newest copy
    return out.sort_values("published", ascending=False).drop_duplicates("text").reset_index(drop=True)


def _load_cache() -> tuple[dict[str, int], np.ndarray]:
    """hash -> row, and the float32 matrix. A dict of Python float lists is ~8x
    the memory of float32 rows: at 400 chars that was ~700 MB, and the first
    run was killed for it."""
    if not CACHE.exists():
        return {}, np.zeros((0, 768), dtype=np.float32)
    c = pd.read_parquet(CACHE)
    return {h: i for i, h in enumerate(c.h)}, np.vstack(c.e.to_numpy()).astype(np.float32)


def _save_cache(index: dict[str, int], mat: np.ndarray) -> None:
    hs = sorted(index, key=index.get)
    pd.DataFrame({"h": hs, "e": list(mat[[index[h] for h in hs]])}).to_parquet(CACHE)


def embed_all(texts: pd.Series) -> np.ndarray:
    from google.genai import types
    hashes = texts.map(lambda t: hashlib.sha1(t.encode()).hexdigest()).to_numpy()
    index, mat = _load_cache()
    todo = sorted(set(hashes) - set(index))
    if todo:
        by_h = dict(zip(hashes, texts))
        batches, cur, n = [], [], 0
        for h in todo:
            if cur and (len(cur) == 50 or n + len(by_h[h]) > 40_000):
                batches.append(cur)
                cur, n = [], 0
            cur.append(h)
            n += len(by_h[h])
        batches.append(cur)
        client = genai_client()

        def one(b):
            r = client.models.embed_content(model="text-embedding-005", contents=[by_h[h] for h in b],
                                            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"))
            return b, np.asarray([e.values for e in r.embeddings], dtype=np.float32)

        # saved every SAVE_EVERY batches, so a killed run keeps what it paid for
        new_h, new_m = [], []
        with ThreadPoolExecutor(max_workers=4) as ex:
            for k, (b, m) in enumerate(ex.map(one, batches), 1):
                new_h += b
                new_m.append(m)
                if k % SAVE_EVERY == 0 or k == len(batches):
                    for h in new_h:
                        index[h] = len(index)
                    mat = np.vstack([mat, *new_m])
                    new_h, new_m = [], []
                    _save_cache(index, mat)
        print(f"  embedded {len(todo)} new chunks ({sum(len(by_h[h]) for h in todo):,} chars)")
    return unit(mat[[index[h] for h in hashes]])


def seed_cache(snap: pd.DataFrame) -> None:
    """Today's chunks are 1,200/no-overlap: their embeddings are already paid for."""
    pd.DataFrame({"h": snap.text.map(lambda t: hashlib.sha1(t.encode()).hexdigest()),
                  "e": snap.embedding.map(lambda v: np.asarray(v, dtype=np.float32))}
                 ).drop_duplicates("h").to_parquet(CACHE)


def relevance(qrels: dict, vers: pd.DataFrame):
    """qid -> {version: [(a, b, grade)]} for chunked docs, and qid -> {blurb doc: grade}."""
    where = {}
    for v in vers.itertuples():
        for doc, (a, b) in v.spans.items():
            where[doc] = (f"{v.art}@{v.published:%Y-%m-%dT%H:%M}", a, b)
    spans, blurbs = {}, {}
    for qid, rel in qrels.items():
        for doc, g in rel.items():
            if doc in where:
                ver, a, b = where[doc]
                spans.setdefault(qid, {}).setdefault(ver, []).append((a, b, g))
            elif doc.startswith("blurb-"):
                blurbs.setdefault(qid, {})[doc] = g
    return spans, blurbs


def score(qs, qvec, df, docs, spans, blurbs) -> dict:
    out = {}
    ids, vers, A, B, L = df.doc.to_numpy(), df.ver.to_numpy(), df.a.to_numpy(), df.b.to_numpy(), df.text.str.len().to_numpy()
    for mode in ("open", "player"):
        rows = []
        for i, q in enumerate(qs):
            sp, bl = spans.get(q["qid"], {}), blurbs.get(q["qid"], {})
            targets = [(ver, a, b) for ver, rs in sp.items() for a, b, g in rs if g == 2] + \
                      [(d, None, None) for d, g in bl.items() if g == 2]
            if not targets or (mode == "player" and not q["player"]):
                continue
            sims = docs @ qvec[i]
            if mode == "player":
                sims = np.where(name_mask(df, q["player"]), sims, -np.inf)
            used, got = 0, {"strict": set(), "loose": set()}
            first = {"strict": BUDGET, "loose": BUDGET}
            for j in np.argsort(-sims):
                if not np.isfinite(sims[j]) or used + L[j] > BUDGET:
                    break
                for t, (ver, a, b) in enumerate(targets):
                    if a is None:
                        hit_s = hit_l = ids[j] == ver
                    elif vers[j] != ver:
                        continue
                    else:
                        ov = max(0, min(B[j], b) - max(A[j], a))
                        hit_s, hit_l = ov >= STRICT * (b - a), ov >= min(OVERLAP_MIN, b - a)
                    for k, h in (("strict", hit_s), ("loose", hit_l)):
                        if h:
                            if not got[k]:
                                first[k] = used
                            got[k].add(t)
                used += L[j]
            n = min(len(targets), MAX_REL)
            rows.append({"qid": q["qid"], **{f"hit_{k}": float(bool(got[k])) for k in got},
                         **{f"recall_{k}": min(len(got[k]), MAX_REL) / n for k in got},
                         **{f"first_{k}": first[k] for k in got}})
        out[mode] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*")
    args = ap.parse_args()
    configs = [c for c in CONFIGS if not args.sizes or c[0] in args.sizes]

    qs, qrels = load_qrels(approved_only=False)
    if not CACHE.exists():
        seed_cache(pd.read_parquet(SNAPSHOT, columns=["text", "embedding"]))
    snap = pd.read_parquet(SNAPSHOT, columns=["chunk_id", "published", "kind", "text", "player_keys"])
    win = snap[snap.published >= CUTOFF - pd.Timedelta(days=WINDOW_DAYS)]
    bl = win[win.kind == "player_blurb"]
    blurbs_df = pd.DataFrame({"doc": bl.chunk_id + "@" + bl.published.dt.strftime("%Y-%m-%dT%H:%M"),
                              "ver": None, "a": 0, "b": 0, "published": bl.published,
                              "player_keys": bl.player_keys, "text": bl.text})
    vers = versions()
    spans, blurb_rel = relevance(qrels, vers)
    qs = [q for q in qs if q["qid"] in spans or q["qid"] in blurb_rel]
    print(f"{len(vers)} article versions, {len(qs)} questions")
    qvec = embed_queries([q["question"] for q in qs])

    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    for size, ov in configs:
        name = f"{size}/{int(ov * 100)}%"
        df = corpus(vers, blurbs_df, size, ov)
        st = df[df.ver.notna()]
        print(f"{name}: {len(st)} story chunks, median {st.text.str.len().median():.0f} chars")
        res = score(qs, qvec, df, embed_all(df.text), spans, blurb_rel)
        results[name] = {m: {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0] if k != "qid"}
                         | {"n": len(rows)} for m, rows in res.items()}
        OUT.write_text(json.dumps(results, indent=1))
    print(f"\n{'config':<10}{'mode':<8}{'hit strict':>11}{'hit loose':>10}{'rec strict':>11}"
          f"{'rec loose':>10}{'first strict':>13}")
    for name, r in results.items():
        for m, x in r.items():
            print(f"{name:<10}{m:<8}{x['hit_strict']:>11.3f}{x['hit_loose']:>10.3f}"
                  f"{x['recall_strict']:>11.3f}{x['recall_loose']:>10.3f}{x['first_strict']:>13.0f}")


if __name__ == "__main__":
    main()
