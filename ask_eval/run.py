"""Ask, two designs, one question set: generate the answers to be graded.

    python ask_eval/run.py                 # all questions, both designs, 2 repeats
    python ask_eval/run.py --limit 5       # a quick look first
    python ask_eval/run.py --designs pipeline_pod   # the pipeline + podcasts (2026-10-29)

The experiment (program reference, Ask section): is the Ask agent -- a model
that picks its own tools at runtime ("lazy" context) -- better than a fixed
pipeline that always fetches the same things and calls the model once
("eager" context)? Both are production code in webapp/ask.py (run and
run_pipeline); this script only feeds them questions.

Held equal between the two, so the design is the only thing that differs:
  - the same archive AS OF the test set's cutoff (embed_tune/common.CUTOFF):
    archive._now is frozen there, and every query filters published < now,
    so both designs see exactly the archive the answer key was built on
  - the same tools (mcp_server/server.py, called in-process), the same model,
    temperature, answer rules and citation check
  - each question asked twice per design, to measure how much the answer
    type wobbles run to run

Questions: the 150 retrieval-test questions that survived the cross-check
(embed_tune/questions.jsonl), plus 20 the archive cannot answer
(unanswerable.jsonl: fictional players, false premises, after the cutoff,
not NFL). Player is passed when the question has one -- what the page's
player picker does.

Output: ask_eval/answers.jsonl, one line per (qid, design, rep), appended
and resumable (a killed run skips what is done). ~$2-3 in model + ranker.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "webapp"), str(ROOT / "src"), str(ROOT), str(ROOT / "embed_tune")]
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "fantasy-football-ml-503718")

OUT = ROOT / "ask_eval" / "answers.jsonl"
UNANSWERABLE = ROOT / "ask_eval" / "unanswerable.jsonl"
GEN_PROJECT = "advisor-agent-2026"      # where the web app's Ask runs its model
CONCURRENCY = 6


def questions() -> list[dict]:
    from eval import load_qrels
    qs, _ = load_qrels(approved_only=False)
    return qs + [json.loads(line) for line in UNANSWERABLE.open(encoding="utf-8")]


class LocalMcp:
    """news-mcp's two tools, called in-process: same functions, no network hop
    (the hop is the same for both designs, so it is left out of the timing)."""

    def __init__(self):
        from mcp_server import server
        self.server = server

    async def list_tools(self):
        from mcp_types import Tool
        tools = await self.server.mcp.list_tools()
        return SimpleNamespace(tools=[Tool(name=t.name, description=t.description,
                                           input_schema=t.input_schema) for t in tools])

    async def call_tool(self, name, args):
        fn = {"player_history": self.server.player_history,
              "search_news_archive": self.server.search_news_archive,
              "search_podcasts": self.server.search_podcasts}[name]
        out = await asyncio.to_thread(fn, **args)
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(out, default=str))])


def freeze_archive_clock():
    from common import CUTOFF
    from gameday_agent import archive
    cutoff = CUTOFF.to_pydatetime().astimezone(timezone.utc)
    archive._now = lambda: cutoff
    archive.RERANK_DAILY_CAP = 10_000      # the eval alone is ~700 searches
    return cutoff


async def main(limit: int | None, repeats: int, designs: list[str]) -> None:
    import ask
    from google import genai
    from google.genai import types

    cutoff = freeze_archive_clock()
    retry = types.HttpRetryOptions(attempts=6, initial_delay=2.0, max_delay=30.0,
                                   exp_base=2.0, http_status_codes=[429, 500, 503, 504])
    client = genai.Client(vertexai=True, project=GEN_PROJECT, location="us-central1",
                          http_options=types.HttpOptions(retry_options=retry))

    async def generate(contents, decls, allow, instruction):
        return await client.aio.models.generate_content(
            model=ask.MODEL, contents=contents, config=ask.gen_config(decls, allow, instruction))

    qs = questions()[:limit] if limit else questions()
    done = set()
    if OUT.exists():
        for line in OUT.open(encoding="utf-8"):
            r = json.loads(line)
            done.add((r["qid"], r["design"], r["rep"]))
    jobs = [(q, d, rep) for rep in range(repeats) for q in qs for d in designs
            if (q["qid"], d, rep) not in done]
    print(f"{len(qs)} questions, cutoff {cutoff:%Y-%m-%d}, {len(jobs)} runs to do "
          f"({len(done)} already done)", flush=True)

    mcp, sem, lock = LocalMcp(), asyncio.Semaphore(CONCURRENCY), asyncio.Lock()
    # "pipeline" is the news-only baseline the 2026-10-28 answers were made
    # with; "pipeline_pod" adds the podcast search in its own slots.
    fns = {"agent": ask.run,
           "pipeline": lambda *a, **k: ask.run_pipeline(*a, **k, podcasts=False),
           "pipeline_pod": lambda *a, **k: ask.run_pipeline(*a, **k, podcasts=True)}
    n_ok = 0

    async def one(q, design, rep):
        nonlocal n_ok
        async with sem:
            t0 = time.monotonic()
            try:
                out = await fns[design](q["question"], q.get("player") or "", [],
                                        generate=generate, mcp=mcp)
                err = None
            except Exception as e:  # noqa: BLE001 -- record and move on; a rerun retries it
                out, err = None, f"{type(e).__name__}: {str(e)[:200]}"
            elapsed = round(time.monotonic() - t0, 2)
        if err:
            print(f"  ERROR {q['qid']} {design} {rep}: {err}", flush=True)
            return
        row = {"qid": q["qid"], "design": design, "rep": rep, "kind": q.get("kind"),
               "question": q["question"], "player": q.get("player") or "",
               "elapsed_s": elapsed, **{k: out[k] for k in (
                   "answer", "claims", "problems", "answer_type", "cited_share", "tool_calls",
                   "tokens", "cost_usd", "cost", "versions")},
               "evidence": [{"tool": e["tool"], "args": e["args"], "ranked": e.get("ranked"),
                             "fallback": e.get("fallback"),
                             "items": [{k: i.get(k) for k in ("ref", "id", "date", "days_ago",
                                                              "source", "player", "headline", "text")}
                                       for i in e["items"]]} for e in out["evidence"]]}
        async with lock:
            with OUT.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
            n_ok += 1
            if n_ok % 20 == 0:
                print(f"  {n_ok}/{len(jobs)}", flush=True)

    await asyncio.gather(*(one(*j) for j in jobs))
    print(f"done: {n_ok} written to {OUT.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import logging
    logging.disable(logging.INFO)          # the SDKs log every HTTP request
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--designs", default="agent,pipeline")
    a = ap.parse_args()
    asyncio.run(main(a.limit, a.repeats, a.designs.split(",")))
