"""Settle 400 vs 800 vs 1,200: grade what each one actually returns, with one judge.

    python embed_tune/chunk_judge.py --count   # how many passages, before paying
    python embed_tune/chunk_judge.py           # judge (resumable) and score

chunk_size.py could only bound the small sizes: its span overlap says a
400-char piece touched a relevant passage, not that it holds the answer. This
grades the passages themselves.

Every passage any of the three configs puts inside the context budget, open
or player mode, is graded FRESH by judge.py's judge (Gemini 2.5 Pro, same
prompt) -- including 1,200's, which the key already grades. The key went
through a Claude cross-check that lowered 241 grades; new passages would not
have, so mixing the two would tilt the test toward the new sizes. One judge,
one pass, all three configs.

Each question's passages from all configs go to the judge in one shuffled
list, with no hint of which config returned them. Identical text (blurbs,
which no config re-cuts) is graded once.

Scored per config, per mode, over the same questions:
  hit       a grade-2 passage within the budget
  sources   distinct articles/blurbs with a grade-2 passage in the budget --
            comparable across sizes, unlike a passage count (one article is
            three 400-char passages but one 1,200-char one)
  answer%   share of the budget's characters that sit in grade-2 passages
  first     characters read before the first grade-2 passage
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from chunk_size import BUDGET, CUTOFF, SNAPSHOT, WINDOW_DAYS, corpus, embed_all, versions
from common import KEY, embed_queries, genai_client, name_mask
from eval import load_qrels
from judge import JUDGE_MODEL, MAX_CHARS, PROMPT, SCHEMA

CONFIGS = [(400, 0), (800, 0), (1200, 0)]
OUT = KEY / "chunk_judged.jsonl"
RESULT = KEY / "chunk_judge.json"


def returned(qs, qvec) -> dict:
    """config -> mode -> qid -> [passage dicts within the budget, in rank order]."""
    snap = pd.read_parquet(SNAPSHOT, columns=["chunk_id", "published", "kind", "text", "player_keys"])
    win = snap[snap.published >= CUTOFF - pd.Timedelta(days=WINDOW_DAYS)]
    bl = win[win.kind == "player_blurb"]
    blurbs = pd.DataFrame({"doc": bl.chunk_id + "@" + bl.published.dt.strftime("%Y-%m-%dT%H:%M"),
                           "ver": None, "a": 0, "b": 0, "published": bl.published,
                           "player_keys": bl.player_keys, "text": bl.text})
    vers = versions()
    out = {}
    for size, ov in CONFIGS:
        name = f"{size}/{int(ov * 100)}%"
        df = corpus(vers, blurbs, size, ov)
        docs = embed_all(df.text)                  # all cached by chunk_size.py
        L = df.text.str.len().to_numpy()
        out[name] = {"open": {}, "player": {}}
        for i, q in enumerate(qs):
            sims = docs @ qvec[i]
            for mode in ("open", "player"):
                if mode == "player":
                    if not q["player"]:
                        continue
                    s = np.where(name_mask(df, q["player"]), sims, -np.inf)
                else:
                    s = sims
                got, used = [], 0
                for j in np.argsort(-s):
                    if not np.isfinite(s[j]) or used + L[j] > BUDGET:
                        break
                    r = df.iloc[j]
                    got.append({"source": r.ver or r.doc, "text": r.text, "chars": int(L[j]),
                                "date": f"{r.published:%a %b %d}",
                                "h": hashlib.sha1(r.text.encode()).hexdigest()})
                    used += L[j]
                out[name][mode][q["qid"]] = got
    return out


def judge_question(client, q: dict, passages: list[dict]) -> dict:
    from google.genai import types
    passages = passages[:]
    random.Random(q["qid"]).shuffle(passages)     # no config order for the judge to read
    listing = "\n\n".join(f"[{i}] ({p['date']}) {p['text'][:MAX_CHARS]}"
                          for i, p in enumerate(passages, 1))
    prompt = PROMPT.format(question=q["question"], passages=listing,
                           player_line=f"Player: {q['player']}\n" if q["player"] else "")
    resp = client.models.generate_content(
        model=JUDGE_MODEL, contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0, response_mime_type="application/json", response_schema=SCHEMA,
            thinking_config=types.ThinkingConfig(thinking_budget=4096)))
    grades = {g["n"]: g["grade"] for g in json.loads(resp.text)}
    missing = [i for i in range(1, len(passages) + 1) if i not in grades]
    if missing:
        raise ValueError(f"{q['qid']}: no grade for passages {missing}")
    u = resp.usage_metadata
    return {"qid": q["qid"], "grades": {p["h"]: grades[i] for i, p in enumerate(passages, 1)},
            "tokens": {"in": u.prompt_token_count,
                       "out": u.candidates_token_count + (u.thoughts_token_count or 0)}}


def score(ret: dict, grades: dict) -> dict:
    res = {}
    for name, modes in ret.items():
        res[name] = {}
        for mode, per_q in modes.items():
            rows = []
            for qid, ps in per_q.items():
                if qid not in grades:
                    continue
                g = [grades[qid].get(p["h"], 0) for p in ps]
                used, first = 0, BUDGET
                for p, x in zip(ps, g):
                    if x == 2:
                        first = used
                        break
                    used += p["chars"]
                total = sum(p["chars"] for p in ps) or 1
                rows.append({"hit": float(2 in g),
                             "sources": len({p["source"].split("#")[0] for p, x in zip(ps, g) if x == 2}),
                             "answer_share": sum(p["chars"] for p, x in zip(ps, g) if x == 2) / total,
                             "first": first, "passages": len(ps)})
            res[name][mode] = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in rows[0]} | {"n": len(rows)}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", action="store_true")
    args = ap.parse_args()

    qs, _ = load_qrels(approved_only=False)
    qvec = embed_queries([q["question"] for q in qs])
    ret = returned(qs, qvec)
    per_q = {}
    for modes in ret.values():
        for m in modes.values():
            for qid, ps in m.items():
                for p in ps:
                    per_q.setdefault(qid, {})[p["h"]] = p
    n = sum(len(v) for v in per_q.values())
    print(f"{n} unique (question, passage) pairs over {len(per_q)} questions; "
          f"~${n * 0.0015:.0f} at judge.py's observed rate")
    if args.count:
        return

    done = {}
    if OUT.exists():
        done = {r["qid"]: r for r in map(json.loads, OUT.open(encoding="utf-8"))}
    todo = [q for q in qs if q["qid"] in per_q and q["qid"] not in done]
    print(f"judging {len(todo)} questions ({len(done)} already done)")
    client = genai_client()

    def run(q):
        try:
            return judge_question(client, q, list(per_q[q["qid"]].values()))
        except Exception as e:  # noqa: BLE001 -- log and leave it for the re-run
            print(f"  {q['qid']}: {type(e).__name__}: {str(e)[:150]}")
            return None

    with OUT.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=6) as ex:
        for r in ex.map(run, todo):
            if r:
                f.write(json.dumps(r) + "\n")
                f.flush()
                done[r["qid"]] = r

    tin = sum(r["tokens"]["in"] for r in done.values())
    tout = sum(r["tokens"]["out"] for r in done.values())
    print(f"{len(done)} questions judged; {tin:,} in / {tout:,} out+thinking "
          f"(~${tin * 1.25e-6 + tout * 10e-6:.2f} at list price)")
    res = score(ret, {q: r["grades"] for q, r in done.items()})
    RESULT.write_text(json.dumps(res, indent=1))
    print(f"\n{'config':<10}{'mode':<8}{'hit':>6}{'sources':>9}{'answer%':>9}{'first':>7}{'passages':>9}")
    for name, modes in res.items():
        for m, x in modes.items():
            print(f"{name:<10}{m:<8}{x['hit']:>6.3f}{x['sources']:>9.2f}{x['answer_share']:>9.1%}"
                  f"{x['first']:>7.0f}{x['passages']:>9.1f}")


if __name__ == "__main__":
    main()
