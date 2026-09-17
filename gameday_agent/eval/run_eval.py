"""On-demand eval harness for the gameday agent -- run any day, no Sunday required.

Drives the real ADK agent (real Gemini calls, real tool execution) against
frozen DK+ESPN fixtures with scenario-specific overrides, so results are
reproducible day to day even though live injury reports change constantly
and DK's endpoint is presently unreachable at all (see dk_client.py).

    python gameday_agent/eval/run_eval.py
    python gameday_agent/eval/run_eval.py --scenario emergency_out
    python gameday_agent/eval/run_eval.py --repeat 3   # hunt flaky regressions

Run from the nfl-dfs repo root, in the ffml env.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "advisor-agent-2026")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
os.environ.setdefault("GEMINI_MODEL", "gemini-2.5-flash")
os.environ.setdefault("SEASON", "2026")
os.environ.setdefault("WEEK", "2")

import pandas as pd  # noqa: E402
import requests  # noqa: E402

import gameday_agent.tools as tools_mod  # noqa: E402
from gameday_agent.agent import root_agent  # noqa: E402
from gameday_agent.eval.fixtures import dk_fixture, espn_fixture  # noqa: E402
from gameday_agent.eval.scenarios import SCENARIOS, Scenario  # noqa: E402
from predict_slate import normalize_name  # noqa: E402

QUERY = "Check my lineup for this Sunday and tell me if anything needs my attention."


@dataclass
class Trace:
    tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    tool_results: list[tuple[str, dict]] = field(default_factory=list)
    final_text: str = ""

    def tool_names(self) -> set[str]:
        return {n for n, _ in self.tool_calls}


def _load_fixtures():
    players, games = dk_fixture()
    injuries, scoreboard = espn_fixture()
    return players, games, injuries, scoreboard


def _apply_espn_overrides(injuries: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    inj = injuries.copy()

    if scenario.espn_clear:
        clear_jn = set(normalize_name(pd.Series(scenario.espn_clear)))
        inj = inj[~inj["jn"].isin(clear_jn)]

    for name, fields in scenario.espn_overrides.items():
        jn = normalize_name(pd.Series([name]))[0]
        inj = inj[inj["jn"] != jn]
        row = {"team": None, "name": name, "position": None, "status": None,
               "body_part": None, "return_date": None, "comment": None,
               "report_date": None, "jn": jn}
        row.update(fields)
        inj = pd.concat([inj, pd.DataFrame([row])], ignore_index=True)

    return inj


def _raise(*_a, **_kw):
    raise requests.ConnectionError("simulated outage (eval harness)")


async def _run_scenario(scenario: Scenario) -> Trace:
    players, games, injuries, scoreboard = _load_fixtures()
    injuries = _apply_espn_overrides(injuries, scenario)

    os.environ["MY_LINEUP"] = ",".join(scenario.lineup)

    tools_mod.slate = _raise if scenario.dk_down else (lambda: (players, games))
    tools_mod.espn_injuries = _raise if scenario.espn_down else (lambda: injuries)
    tools_mod.espn_scoreboard = _raise if scenario.espn_down else (lambda: scoreboard)

    from google.adk.runners import InMemoryRunner
    from google.genai import types

    runner = InMemoryRunner(agent=root_agent, app_name="gameday_eval")
    session = await runner.session_service.create_session(
        app_name="gameday_eval", user_id="eval"
    )
    content = types.Content(role="user", parts=[types.Part(text=QUERY)])

    trace = Trace()
    async for event in runner.run_async(
        user_id="eval", session_id=session.id, new_message=content
    ):
        if not (event.content and event.content.parts):
            continue
        for part in event.content.parts:
            if getattr(part, "function_call", None):
                fc = part.function_call
                trace.tool_calls.append((fc.name, dict(fc.args)))
            if getattr(part, "function_response", None):
                fr = part.function_response
                trace.tool_results.append((fr.name, fr.response))
            if getattr(part, "text", None):
                trace.final_text = part.text  # last text wins -- the actual report

    return trace


async def _run_all(names: list[str] | None, repeat: int) -> bool:
    scenarios = [s for s in SCENARIOS if not names or s.name in names]
    all_ok = True

    for scenario in scenarios:
        print(f"\n=== {scenario.name} ===")
        print(f"    {scenario.description}")
        for i in range(repeat):
            trace = await _run_scenario(scenario)
            failures = [msg for check in scenario.checks if (msg := check(trace))]
            tag = f" (run {i + 1}/{repeat})" if repeat > 1 else ""
            if failures:
                all_ok = False
                print(f"  FAIL{tag}")
                for f in failures:
                    print(f"    - {f}")
                print(f"    report: {trace.final_text[:300]!r}")
            else:
                print(f"  PASS{tag}: {trace.final_text[:120]!r}")

    return all_ok


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Eval harness for the gameday agent")
    p.add_argument("--scenario", action="append",
                    help="run only this scenario (repeatable flag for several)")
    p.add_argument("--repeat", type=int, default=1,
                    help="repeat each scenario N times (flake-hunting)")
    args = p.parse_args()

    ok = asyncio.run(_run_all(args.scenario, args.repeat))
    print(f"\n{'ALL PASS' if ok else 'FAILURES ABOVE'}")
    sys.exit(0 if ok else 1)
