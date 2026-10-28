"""A date-aware answer key: which "relevant" passages are no longer true?

    python embed_tune/recency.py grade      # Gemini 2.5 Pro labels each relevant passage (~$1)
    python embed_tune/recency.py score      # re-score the saved runs on the dated key

Why: the shipped re-ranker is the DATED one (it is told each passage's date
and today's date). Offline it scored nDCG@10 0.811 against the undated
ranker's 0.849, and it was shipped anyway, on one example: undated, asked
when Puka Nacua would be back, it put a September hip note above this
week's calf injury. The key could not see that problem. judge.py graded
each passage against the question with no dates, so "back in Week 4" from
September is a 2 -- relevant -- even when an October note says he is hurt
again. The metric rewarded the old note.

This fixes the key, not the ranker. For the time-sensitive questions
(injury, transaction, role), every passage the key calls relevant is shown
to Gemini 2.5 Pro WITH its date, beside every other relevant passage, as of
the snapshot cutoff, and labelled:

  current      still true as of the cutoff
  background   true history that nothing newer contradicts ("hurt Oct 12")
  superseded   a newer passage in the list replaced it (status changed,
               returned, re-injured, traded, demoted)

Superseded passages then count 0, and two metrics are added:
  nDCG@10 (dated key)   the same nDCG, superseded = 0
  newest@6              is the NEWEST current grade-2 passage in the top 6?
                        For "is he playing", the one passage that must show.

Output: key/recency.json (labels) and key/recency_scores.json.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from common import CUTOFF, KEY, genai_client, load_window
from eval import K, load_qrels, ndcg

LABELS = KEY / "recency.json"
SCORES = KEY / "recency_scores.json"
KINDS = {"injury", "transaction", "role"}
MODEL = "gemini-2.5-pro"
CHARS = 700
RUNS = {"embedding only (005)": "eval_base.json",
        "ranker, undated": "eval_rerank_base.json",
        "ranker, dated (shipped)": "eval_rerank_base_dated.json",
        "gemini-embedding-001 + ranker": "eval_rerank_gemini_emb.json"}

PROMPT = """Today is {today}. A fantasy football player asked:

  "{question}"

Below are news passages already judged RELEVANT to that question, each with its
publish date, newest first. Label EVERY passage by whether what it says is
still true today, given the other passages:

  current     still true as of today (the latest word on its point)
  background  true history that no newer passage contradicts (e.g. "he was hurt
              Oct 12" when nothing says otherwise), useful context
  superseded  a NEWER passage in this list replaced it: his status changed,
              he returned or got hurt again, was traded, released, demoted or
              promoted, or a newer report corrected it

Judge only from the passages, not from your own knowledge. A passage is
superseded only if a newer passage HERE changes it.

{passages}

Return JSON: {{"labels": [{{"id": "<id>", "label": "current|background|superseded",
"why": "<10 words>"}}]}} with one entry per passage."""


def _passages(qrels: dict, df) -> dict:
    out = {}
    for qid, grades in qrels.items():
        ids = [d for d, g in grades.items() if g >= 1 and d in df.index]
        out[qid] = df.loc[ids].sort_values("published", ascending=False) if ids else None
    return out


def grade() -> None:
    from google.genai import types
    qs, qrels = load_qrels(approved_only=False)
    df = load_window().set_index("doc_id")
    pas = _passages(qrels, df)
    done = json.loads(LABELS.read_text(encoding="utf-8")) if LABELS.exists() else {}
    todo = [q for q in qs if q["kind"] in KINDS and q["qid"] not in done and pas.get(q["qid"]) is not None
            and len(pas[q["qid"]]) > 1]          # one passage cannot be superseded by another
    client = genai_client()
    print(f"{len(todo)} questions to label ({len(done)} done)", flush=True)

    def one(q):
        rows = pas[q["qid"]]
        text = "\n\n".join(f"id {d} | {p:%a %b %d, %Y} | {' '.join(str(t).split())[:CHARS]}"
                           for d, p, t in zip(rows.index, rows.published, rows.text))
        schema = {"type": "object", "required": ["labels"], "properties": {"labels": {
            "type": "array", "items": {"type": "object", "required": ["id", "label", "why"],
                                       "properties": {"id": {"type": "string"},
                                                      "label": {"type": "string",
                                                                "enum": ["current", "background", "superseded"]},
                                                      "why": {"type": "string"}}}}}}
        for attempt in range(3):     # schema-constrained, but a reply can still be cut off
            resp = client.models.generate_content(
                model=MODEL, contents=PROMPT.format(today=f"{CUTOFF:%a %b %d, %Y}",
                                                    question=q["question"], passages=text),
                config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json",
                                                   response_json_schema=schema))
            try:
                labels = {x["id"]: x for x in json.loads(resp.text)["labels"]}
                break
            except (json.JSONDecodeError, KeyError, TypeError):
                if attempt == 2:
                    raise
        return q["qid"], {d: {"label": labels.get(d, {}).get("label", "current"),
                              "why": labels.get(d, {}).get("why", "")} for d in rows.index}

    with ThreadPoolExecutor(max_workers=6) as ex:
        for n, (qid, lab) in enumerate(ex.map(one, todo), 1):
            done[qid] = lab
            if n % 10 == 0:
                LABELS.write_text(json.dumps(done, indent=1), encoding="utf-8")
                print(f"  {n}/{len(todo)}", flush=True)
    LABELS.write_text(json.dumps(done, indent=1), encoding="utf-8")
    sup = sum(v["label"] == "superseded" for lab in done.values() for v in lab.values())
    tot = sum(len(lab) for lab in done.values())
    print(f"labelled {tot} passages over {len(done)} questions: {sup} superseded")


def score() -> None:
    qs, qrels = load_qrels(approved_only=False)
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    df = load_window().set_index("doc_id")
    kind = {q["qid"]: q["kind"] for q in qs}

    def dated(qid):
        rel = dict(qrels[qid])
        for d, v in labels.get(qid, {}).items():
            if v["label"] == "superseded":
                rel[d] = 0
        return rel

    def newest_current(qid, rel):
        cur = [d for d, g in rel.items() if g == 2 and d in df.index]
        return max(cur, key=lambda d: df.loc[d, "published"]) if cur else None

    out = {}
    for name, f in RUNS.items():
        path = KEY / f
        if not path.exists():
            continue
        run = json.loads(path.read_text(encoding="utf-8"))
        res = {}
        for mode, rows in run.items():
            rows = [r for r in rows if kind.get(r["qid"]) in KINDS and r["qid"] in labels]
            if not rows:
                continue
            old, new, hit = [], [], []
            for r in rows:
                rel0, rel1 = qrels[r["qid"]], dated(r["qid"])
                old.append(ndcg([rel0.get(d, 0) for d in r["top"]], list(rel0.values())))
                new.append(ndcg([rel1.get(d, 0) for d in r["top"]], list(rel1.values())))
                n = newest_current(r["qid"], rel1)
                if n:
                    hit.append(n in r["top"][:6])
            res[mode] = {"questions": len(rows), "ndcg10_undated_key": round(float(np.mean(old)), 3),
                         "ndcg10_dated_key": round(float(np.mean(new)), 3),
                         "newest_at_6": round(float(np.mean(hit)), 3), "newest_n": len(hit)}
        out[name] = res
    out["shipped search (end to end)"] = shipped(qrels, dated, newest_current, kind, labels, df)
    SCORES.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def shipped(qrels, dated, newest_current, kind, labels, df) -> dict:
    """The search users actually get, scored -- not the ranker alone.

    eval.py scores pure vector search on purpose; production adds the two
    newest items, the ranker, a de-dup and a cut to 6, then sorts by date.
    ask_eval/run.py's pipeline design ran that exact search (frozen at the
    cutoff) for every question, so its saved evidence IS the shipped top 6.
    Order is by date, so these are set metrics: hit@6, newest@6."""
    path = KEY.parent.parent / "ask_eval" / "answers.jsonl"
    if not path.exists():
        return {}
    by_chunk = {}
    for d, p in zip(df.index, df.published):
        by_chunk.setdefault(d.split("@")[0], []).append((d, f"{p:%a %b %d}"))
    res = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if r["design"] != "pipeline" or r["rep"] != 0 or r["qid"] not in qrels:
            continue
        search = [e for e in r["evidence"] if e["tool"] == "search_news_archive"]
        if not search:
            continue
        top = [next((d for d, day in by_chunk.get(i["id"], []) if day == i["date"]), None)
               for i in search[0]["items"]]
        mode = "player" if r["player"] else "open"
        rel0 = qrels[r["qid"]]
        rel1 = dated(r["qid"]) if r["qid"] in labels else rel0
        row = res.setdefault(mode, {"hit": [], "hit_tk": [], "newest": []})
        row["hit"].append(any(rel0.get(d, 0) == 2 for d in top))
        if kind.get(r["qid"]) in KINDS and r["qid"] in labels:
            row["hit_tk"].append(any(rel1.get(d, 0) == 2 for d in top))
            n = newest_current(r["qid"], rel1)
            if n:
                row["newest"].append(n in top)
    return {m: {"questions": len(v["hit"]), "hit_at_6": round(float(np.mean(v["hit"])), 3),
                "time_sensitive_hit_at_6_dated_key": round(float(np.mean(v["hit_tk"])), 3) if v["hit_tk"] else None,
                "newest_at_6": round(float(np.mean(v["newest"])), 3) if v["newest"] else None,
                "newest_n": len(v["newest"])} for m, v in res.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["grade", "score"])
    {"grade": grade, "score": score}[ap.parse_args().step]()
