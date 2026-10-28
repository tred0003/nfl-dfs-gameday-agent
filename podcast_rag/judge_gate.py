"""Grade the podcast passages the gate saw, so podcasts are scored for help, not only harm.

    python podcast_rag/judge_gate.py data/podcasts/gate_staging.json

gate.py run against the staging table (news + podcasts) counts every podcast
passage as a miss: the answer key only grades news. This grades the podcast
passages that reached a top 6, with judge.py's judge and prompt, and asks:

  - of the questions whose hit was lost, did a podcast passage answer it
    instead (then the loss is not real)?
  - how many questions got a grade-2 podcast passage at all (podcasts helping)?
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "embed_tune"), str(ROOT / "src"), str(ROOT)]

from common import genai_client  # noqa: E402
from eval import load_qrels  # noqa: E402
from judge import JUDGE_MODEL, MAX_CHARS, PROMPT, SCHEMA  # noqa: E402

LIVE = ROOT / "data" / "podcasts" / "gate_live.json"
OUT = ROOT / "data" / "podcasts" / "gate_podcast_grades.json"
TABLE = "fantasy-football-ml-503718.dfs.podcast_chunks"


def main() -> None:
    from google.cloud import bigquery
    from google.genai import types
    staging = {r["qid"]: r for r in json.loads(Path(sys.argv[1]).read_text())}
    live = {r["qid"]: r for r in json.loads(LIVE.read_text())}
    qs = {q["qid"]: q for q in load_qrels(approved_only=False)[0]}
    want = {qid: [i for i in r["ids"] if i.startswith("pod:")] for qid, r in staging.items()}
    want = {k: v for k, v in want.items() if v}
    ids = sorted({i for v in want.values() for i in v})
    bq = bigquery.Client(project=TABLE.split(".")[0])
    text = {r.chunk_id: (r.published, r.text) for r in bq.query_and_wait(
        f"SELECT chunk_id, published, text FROM `{TABLE}` WHERE chunk_id IN UNNEST(@ids)",
        job_config=bigquery.QueryJobConfig(query_parameters=[
            bigquery.ArrayQueryParameter("ids", "STRING", ids)]))}
    client = genai_client()

    def one(qid):
        q, ps = qs[qid], want[qid]
        listing = "\n\n".join(f"[{n}] ({text[i][0]:%a %b %d}) {text[i][1][:MAX_CHARS]}"
                              for n, i in enumerate(ps, 1))
        resp = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=PROMPT.format(question=q["question"], passages=listing,
                                   player_line=f"Player: {q['player']}\n" if q["player"] else ""),
            config=types.GenerateContentConfig(
                temperature=0, response_mime_type="application/json", response_schema=SCHEMA,
                thinking_config=types.ThinkingConfig(thinking_budget=4096)))
        g = {x["n"]: x["grade"] for x in json.loads(resp.text)}
        return qid, {i: g.get(n, 0) for n, i in enumerate(ps, 1)}

    with ThreadPoolExecutor(max_workers=6) as ex:
        grades = dict(ex.map(one, want))
    OUT.write_text(json.dumps(grades, indent=1))

    n2 = [q for q, g in grades.items() if 2 in g.values()]
    n1 = [q for q, g in grades.items() if 1 in g.values() and 2 not in g.values()]
    allg = [x for g in grades.values() for x in g.values()]
    print(f"{len(ids)} podcast passages over {len(grades)} questions: "
          f"grade 2 {allg.count(2)}, 1 {allg.count(1)}, 0 {allg.count(0)}")
    print(f"questions with a grade-2 podcast passage: {len(n2)}; grade-1 only: {len(n1)}")
    for kind in ("hit", "newest"):
        lost = [q for q in live if live[q][kind] and staging[q][kind] is False]
        rescued = [q for q in lost if q in n2]
        print(f"{kind} lost on {len(lost)} questions; a podcast passage answered {len(rescued)} of them: {rescued}")
    for q in n2[:6]:
        i = next(i for i, g in grades[q].items() if g == 2)
        print(f"\n{q} {qs[q]['question']}\n   -> {text[i][1][:260]}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
