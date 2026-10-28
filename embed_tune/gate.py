"""The search-quality gate: the SHIPPED search, scored, against a baseline.

    python embed_tune/gate.py              # run the gate (exit 1 on a regression)
    python embed_tune/gate.py --baseline   # record today's numbers as the baseline

eval.py and rerank.py score re-implementations of the search on a laptop
snapshot -- right for experiments, wrong for a gate: a change to
gameday_agent/archive.py would not move them. This calls the real
archive.search_news_archive for the 150 test questions, against the live
BigQuery table, with the archive's clock frozen at the test cutoff (every
query filters published < now, so the archive it sees is the key's archive).

Scored with no laptop data -- only the committed key:
  hit@6       a grade-2 passage among the 6 returned (the undated key)
  newest@6    the newest CURRENT grade-2 passage among the 6, for the
              time-sensitive questions (key/recency.json, the dated key)
  fallbacks   searches where the re-ranker did not run
  pin check   the pinned ranker scores a fixed pair differently from a
              nonsense model name. The Ranking API accepts any name and
              silently falls back to an older ranker; identical scores mean
              the pin is not a real model.

Fails when hit@6 drops more than HIT_TOL or newest@6 more than NEWEST_TOL
below key/gate_baseline.json, when any search fell back, or the pin check
fails. ~150 searches: ~$0.25 (ranker + embeddings), a few minutes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "embed_tune")]
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "fantasy-football-ml-503718")

from common import CUTOFF, KEY  # noqa: E402
from eval import load_qrels  # noqa: E402

BASELINE = KEY / "gate_baseline.json"
LAST = KEY / "gate_last.json"
LABELS = KEY / "recency.json"
KINDS = {"injury", "transaction", "role"}
HIT_TOL = 0.03
NEWEST_TOL = 0.05


def _day(doc_id: str) -> tuple[str, str]:
    """doc_id = chunk_id@YYYY-MM-DDTHH:MM -> (chunk_id, 'Wed Oct 21'), the
    form a search passage carries (id + date)."""
    chunk, stamp = doc_id.rsplit("@", 1)
    return chunk, f"{datetime.fromisoformat(stamp):%a %b %d}"


def pin_check() -> dict:
    from gameday_agent import archive
    r = archive._Ranker()
    if not r.creds.valid:
        r.creds.refresh(r.req)
    texts = ["He sprained his ankle Sunday and is day to day.", "The team traded a draft pick."]
    scores = {}
    for name in (archive.RERANK_MODEL, "nonsense-model"):
        resp = r.http.post(r.url, timeout=20, json={
            "model": name, "query": "is his ankle hurt", "ignoreRecordDetailsInResponse": True,
            "records": [{"id": str(i), "content": t} for i, t in enumerate(texts)]},
            headers={"Authorization": f"Bearer {r.creds.token}", "x-goog-user-project": r.project})
        resp.raise_for_status()
        scores[name] = [round(x["score"], 4) for x in sorted(resp.json()["records"], key=lambda x: x["id"])]
    return {"pinned": archive.RERANK_MODEL, "scores": scores,
            "ok": scores[archive.RERANK_MODEL] != scores["nonsense-model"]}


def run() -> dict:
    from gameday_agent import archive
    cutoff = CUTOFF.to_pydatetime().astimezone(timezone.utc)
    archive._now = lambda: cutoff
    archive.RERANK_DAILY_CAP = 10_000
    pin = pin_check()

    qs, qrels = load_qrels(approved_only=False)
    labels = json.loads(LABELS.read_text(encoding="utf-8"))

    def newest_current(qid):
        rel = dict(qrels[qid])
        for d, v in labels.get(qid, {}).items():
            if v["label"] == "superseded":
                rel[d] = 0
        cur = [d for d, g in rel.items() if g == 2]
        return max(cur, key=lambda d: d.rsplit("@", 1)[1]) if cur else None

    def one(q):
        out = archive.search_news_archive(q["question"], q.get("player") or "")
        got = {(p["id"], p["date"]) for p in out.get("passages", [])}
        rel = qrels[q["qid"]]
        hit = any(g == 2 and _day(d) in got for d, g in rel.items())
        newest = None
        if q["kind"] in KINDS and q["qid"] in labels:
            n = newest_current(q["qid"])
            newest = None if n is None else _day(n) in got
        return {"qid": q["qid"], "hit": hit, "newest": newest,
                "ids": [p["id"] for p in out.get("passages", [])],   # podcast ids start "pod:"
                "ranked": out.get("meta", {}).get("ranked"),
                "fallback": out.get("meta", {}).get("fallback")}

    with ThreadPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(one, qs))
    newest = [r["newest"] for r in rows if r["newest"] is not None]
    LAST.write_text(json.dumps(rows, indent=0), encoding="utf-8")   # per question, for paired comparisons
    return {"retriever": archive.RETRIEVER_VERSION, "recency_frac": archive.RECENCY_FRAC,
            "questions": len(rows),
            "hit_at_6": round(sum(r["hit"] for r in rows) / len(rows), 3),
            "newest_at_6": round(sum(newest) / len(newest), 3), "newest_n": len(newest),
            "fallbacks": sum(1 for r in rows if not r["ranked"]),
            "fallback_reasons": sorted({str(r["fallback"]) for r in rows if not r["ranked"]}),
            "pin_check": pin}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    a = ap.parse_args()
    res = run()
    print(json.dumps(res, indent=1))
    if a.baseline:
        BASELINE.write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(f"baseline written: {BASELINE.relative_to(ROOT)}")
        return 0
    base = json.loads(BASELINE.read_text(encoding="utf-8"))
    fails = []
    if res["hit_at_6"] < base["hit_at_6"] - HIT_TOL:
        fails.append(f"hit@6 {res['hit_at_6']} < baseline {base['hit_at_6']} - {HIT_TOL}")
    if res["newest_at_6"] < base["newest_at_6"] - NEWEST_TOL:
        fails.append(f"newest@6 {res['newest_at_6']} < baseline {base['newest_at_6']} - {NEWEST_TOL}")
    if res["fallbacks"]:
        fails.append(f"{res['fallbacks']} searches fell back from the ranker: {res['fallback_reasons']}")
    if not res["pin_check"]["ok"]:
        fails.append(f"ranker pin {res['pin_check']['pinned']} scores like a nonsense model name")
    for f in fails:
        print("GATE FAIL:", f)
    print("GATE PASS" if not fails else f"GATE FAILED ({len(fails)})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.disable(logging.INFO)
    sys.exit(main())
