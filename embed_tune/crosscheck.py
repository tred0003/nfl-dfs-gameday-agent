"""Merge Claude's cross-check of the judge's answers into the answer key.

    python embed_tune/crosscheck.py <dir with verdict1.json ... verdictN.json>

The judge (Gemini 2.5 Pro) marked 1,613 passages as answers (grade 2). Six
Claude agents then read every one of those passages against its question,
without being told the judge's grades, and listed the ones that do not answer
it. Different model family, same rubric.

The rule is the injection screen's: a passage stays an answer only when both
models agree. One that Claude rejects drops to a 1 (useful context). A
question with no agreed answer left, or one Claude flags as unanswerable as
written, is cut.

This writes key/crosscheck.json (the verdicts, kept as the record) and
key/review.json (what eval.py reads), and prints the agreement rate -- the
judge's precision, as far as a second model can measure it.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from common import KEY
from packet import load, shown


def main() -> None:
    src = Path(sys.argv[1])
    verdicts = {}
    for f in sorted(src.glob("verdict*.json")):
        verdicts.update(json.loads(f.read_text(encoding="utf-8")))
    qs, judged, df = load()
    missing = [q["qid"] for q in qs if q["qid"] not in verdicts]
    if missing:
        sys.exit(f"no verdict for {len(missing)} questions: {missing[:10]}")

    review, total, rejected = {}, 0, 0
    for q in qs:
        v = verdicts[q["qid"]]
        ids = shown(q["qid"], judged, df)
        bad = sorted({n for n in v.get("not_answer", []) if 1 <= n <= len(ids)})
        total += len(ids)
        rejected += len(bad)
        if v.get("question_bad") or len(bad) == len(ids):
            review[q["qid"]] = {"status": "cut", "why": v.get("why", "")}
        else:
            review[q["qid"]] = {"status": "ok", "grades": {ids[n - 1]: 1 for n in bad}}

    (KEY / "crosscheck.json").write_text(json.dumps(verdicts, indent=1), encoding="utf-8")
    (KEY / "review.json").write_text(json.dumps(review, indent=1), encoding="utf-8")
    status = Counter(r["status"] for r in review.values())
    print(f"judge's answers: {total}; Claude rejected {rejected} "
          f"-> agreement {1 - rejected / total:.1%}")
    print(f"questions: {dict(status)}")
    for qid, r in review.items():
        if r["status"] == "cut":
            print(f"  cut {qid}: {r['why'][:120]}")


if __name__ == "__main__":
    main()
