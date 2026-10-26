"""Fair pool: grade what a NEW model shows that the answer key never saw.

    python embed_tune/pool_add.py key/eval_bge_small_tuned.json [more eval files]
    python embed_tune/pool_add.py --merge <dir with addverdict*.json>

The answer key was built from three retrievers, one of them the base model.
A new model that surfaces a good passage none of them found would score it
as wrong. So, before models are compared: take each new model's top 10 (open
and player-filtered), grade the passages the key has never seen with the same
judge and prompt (judge.py), and append them to key/judged_add.jsonl.

Same both-must-agree rule as the first key: every new grade 2 is written to
key/add_to_check.jsonl, Claude agents re-read those blind, and `--merge`
writes key/crosscheck_add.json, which drops the rejected ones to 1. eval.py
reads both files. Every model is then re-scored on the bigger key, base
included, so all are scored against one answer key.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common import KEY, QUESTIONS, genai_client, load_window
from judge import judge_one

JUDGED = KEY / "judged.jsonl"
ADDED = KEY / "judged_add.jsonl"
TO_CHECK = KEY / "add_to_check.jsonl"
CHECKED = KEY / "crosscheck_add.json"


def graded() -> dict[str, set]:
    seen: dict[str, set] = {}
    for f in (JUDGED, ADDED):
        if f.exists():
            for line in f.open(encoding="utf-8"):
                r = json.loads(line)
                seen.setdefault(r["qid"], set()).update(g["doc_id"] for g in r["grades"])
    return seen


def add(files: list[str]) -> None:
    seen = graded()
    new: dict[str, dict[str, str]] = {}           # qid -> doc_id -> which model found it
    for f in files:
        name = Path(f).stem.removeprefix("eval_")
        for rows in json.loads(Path(f).read_text(encoding="utf-8")).values():
            for r in rows:
                for d in r["top"]:
                    if d not in seen.get(r["qid"], set()):
                        new.setdefault(r["qid"], {}).setdefault(d, name)
    n = sum(map(len, new.values()))
    print(f"{n} unseen passages across {len(new)} questions")
    if not n:
        return

    df = load_window()
    text = {r.doc_id: {"text": r.text, "date": r.published.strftime("%a %b %d")}
            for r in df.itertuples()}
    qs = {q["qid"]: q for q in map(json.loads, QUESTIONS.open(encoding="utf-8"))}
    client = genai_client()

    def run(qid):
        cands = [{"doc_id": d, "from": m} for d, m in new[qid].items()]
        try:
            return judge_one(client, qs[qid], cands, text)
        except Exception as e:  # noqa: BLE001 -- the re-run picks it up
            print(f"  {qid}: {type(e).__name__}: {str(e)[:150]}")
            return None

    twos = []
    with ADDED.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(run, sorted(new)):
            if r:
                f.write(json.dumps(r) + "\n")
                twos += [{"qid": r["qid"], "question": qs[r["qid"]]["question"],
                          "player": qs[r["qid"]]["player"], "doc_id": g["doc_id"],
                          "text": text[g["doc_id"]]["text"]}
                         for g in r["grades"] if g["grade"] == 2]
    with TO_CHECK.open("a", encoding="utf-8") as f:
        for t in twos:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"graded; {len(twos)} new answers (grade 2) -> {TO_CHECK.name} for the cross-check")


def merge(src: Path) -> None:
    """addverdict*.json: {qid: {"not_answer": [doc_id, ...]}} from the Claude agents."""
    verdicts = {}
    for f in sorted(src.glob("addverdict*.json")):
        verdicts.update(json.loads(f.read_text(encoding="utf-8")))
    checks = [json.loads(line) for line in TO_CHECK.open(encoding="utf-8")]
    out = {}
    rejected = 0
    for c in checks:
        if c["doc_id"] in verdicts.get(c["qid"], {}).get("not_answer", []):
            out.setdefault(c["qid"], {})[c["doc_id"]] = 1
            rejected += 1
    CHECKED.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"new answers: {len(checks)}; Claude rejected {rejected} "
          f"-> agreement {1 - rejected / max(len(checks), 1):.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("evals", nargs="*")
    ap.add_argument("--merge", type=Path)
    args = ap.parse_args()
    merge(args.merge) if args.merge else add(args.evals)


if __name__ == "__main__":
    main()
