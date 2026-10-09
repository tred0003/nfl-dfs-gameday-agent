"""On-demand eval harness for the gameday agent -- run any day, no Sunday required.

Drives the real ADK agent (real Gemini calls, real tool execution) against
frozen DK+ESPN fixtures with scenario-specific overrides, so results are
reproducible day to day even though live injury reports change constantly
and DK's endpoint is presently unreachable at all (see dk_client.py).

    python gameday_agent/eval/run_eval.py
    python gameday_agent/eval/run_eval.py --scenario emergency_out
    python gameday_agent/eval/run_eval.py --repeat 3     # hunt flaky regressions
    python gameday_agent/eval/run_eval.py --no-judge     # checks only, free
    python gameday_agent/eval/run_eval.py --save-baseline

Run from the nfl-dfs repo root, in the ffml env.

WHAT A RUN PRODUCES. Three numbers, never averaged together, because they
fail for different reasons -- see scoring.py for why that separation is the
point rather than an inconvenience:

    checks      deterministic assertions, each one a past bug that came back
    quality     the LLM judge's rubric score, 0-100
    trajectory  tool calls, repeats, seconds -- what it DID, not what it said

The exit code is the gate. Non-zero means a check failed or quality dropped
more than the tolerance below the stored baseline, which is what lets this
be wired into CI rather than eyeballed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
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

# The scorecard contains box-drawing characters and the reports contain
# emoji. Windows consoles are cp1252 by default and would die on print --
# losing the run after paying for every model call in it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd  # noqa: E402
import requests  # noqa: E402

import gameday_agent.tools as tools_mod  # noqa: E402
from gameday_agent.agent import root_agent  # noqa: E402
from gameday_agent.eval.fixtures import (  # noqa: E402
    dk_fixture, espn_fixture, vegas_fixture,
)
from gameday_agent.eval.judge import JUDGE_MODEL, judge  # noqa: E402
from gameday_agent.eval.rubric import CRITERIA  # noqa: E402
from gameday_agent.eval.scenarios import (  # noqa: E402
    FAKE_PLAYER_PHRASES, SCENARIOS, TOOL_CALL_LIMIT, Scenario,
    must_not_contain_any, must_stay_under_tool_calls,
)
from gameday_agent.eval.scoring import (  # noqa: E402
    QUALITY_TOLERANCE, Trace, build_result, by_scenario, gate, load_baseline,
    mean_quality, save_baseline, save_run, suite_quality,
)
from predict_slate import normalize_name  # noqa: E402

QUERY = "Check my lineup for this Sunday and tell me if anything needs my attention."


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


def _raise_runtime(*_a, **_kw):
    raise RuntimeError("cornerback grades are current-season only, so they "
                       "cannot describe a historical slate")


def build_replay_scenarios(season: int, week: int, count: int) -> list[Scenario]:
    """Real lineups on a real past slate, for a human grading packet.

    The checks are deliberately thin. Against invented fixtures the harness
    knows the right answer and can assert it; against a real slate it does
    not, and an assertion written from a guess is worse than none. What is
    left is the handful of things that are wrong regardless of what the data
    says: calling a real player fake, and looping.

    Quality here comes from the judge and, more to the point, from you.
    """
    from gameday_agent.eval.replay import build_lineup, replay_slate

    data = replay_slate(season, week)
    out = []
    for i in range(count):
        # Alternate clean and injured lineups. A packet of eight quiet
        # Sundays would not tell you how the agent handles the one case
        # that actually matters.
        hurt = i % 2 == 1
        lineup = build_lineup(data, seed=i, want_injured=hurt)
        out.append(Scenario(
            name=f"replay_{season}w{week}_{i + 1}",
            description=(f"Real {season} week {week} slate, "
                         f"{'lineup carries a designation' if hurt else 'lineup looks clean'}: "
                         + ", ".join(lineup)),
            lineup=lineup,
            replay=(season, week),
            checks=[
                must_not_contain_any(*FAKE_PLAYER_PHRASES),
                must_stay_under_tool_calls(TOOL_CALL_LIMIT),
            ],
        ))
    return out


async def _run_scenario(scenario: Scenario) -> Trace:
    if scenario.replay:
        from gameday_agent.eval.replay import replay_slate

        d = replay_slate(*scenario.replay)
        players, games = d["players"], d["games"]
        injuries, scoreboard, vegas = d["injuries"], d["scoreboard"], d["vegas"]
        # Current-season cornerback grades against a years-old slate would be
        # fiction. Made to fail cleanly instead -- get_cb_matchups catches
        # RuntimeError and reports itself unavailable. See replay.py.
        tools_mod._cb_board = _raise_runtime
    else:
        players, games, injuries, scoreboard = _load_fixtures()
        injuries = _apply_espn_overrides(injuries, scenario)
        vegas = vegas_fixture()

    os.environ["MY_LINEUP"] = ",".join(scenario.lineup)

    tools_mod.slate = _raise if scenario.dk_down else (lambda: (players, games))
    tools_mod.espn_injuries = _raise if scenario.espn_down else (lambda: injuries)
    tools_mod.espn_scoreboard = _raise if scenario.espn_down else (lambda: scoreboard)
    # Patched for the same reason as the other three, and it was missing: the
    # Vegas numbers were the one input still coming from the real schedule,
    # which made "reproducible day to day" untrue and let a team's line
    # describe a game that is not on this fixture slate.
    tools_mod.vegas_for_slate = lambda *_a, **_kw: vegas

    from google.adk.runners import InMemoryRunner
    from google.genai import types

    runner = InMemoryRunner(agent=root_agent, app_name="gameday_eval")
    session = await runner.session_service.create_session(
        app_name="gameday_eval", user_id="eval"
    )
    content = types.Content(role="user", parts=[types.Part(text=QUERY)])

    trace = Trace()
    started = time.perf_counter()
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
    trace.elapsed_s = time.perf_counter() - started

    return trace


# ------------------------------------------------------------------ reporting

def _print_scenario(result, trace, verbose: bool, tag: str) -> None:
    status = "PASS" if result.checks_ok else "FAIL"
    quality = "--" if result.quality is None else f"{result.quality:5.1f}"
    print(f"  {status}{tag}  checks {result.checks_passed}/{result.checks_total}"
          f"   quality {quality}"
          f"   {result.tool_count} calls in {result.elapsed_s}s")

    for f in result.failures:
        print(f"      x {f}")
    if result.repeats:
        print(f"      ~ repeated calls: {', '.join(result.repeats)}")
    if result.judge_error:
        print(f"      ? judge unavailable: {result.judge_error}")

    # The judge's low scores are the useful part. A number tells you
    # something moved; the reason tells you what to go change. Anything
    # below a 4 gets printed, and -v prints the lot.
    for c in CRITERIA:
        score = result.judge_scores.get(c.key)
        if score is None:
            continue
        if score < 4 or verbose:
            print(f"      {score}/5 {c.key}: {result.judge_reasons.get(c.key, '')}")

    if verbose or not result.checks_ok:
        print(f"      report: {trace.final_text[:400]!r}")


def _print_summary(results, baseline) -> None:
    print("\n" + "=" * 68)
    grouped = by_scenario(results)
    width = max(len(n) for n in grouped)

    for name, runs in grouped.items():
        q = mean_quality(runs)
        prior = (baseline or {}).get("scenarios", {}).get(name, {}).get("quality")
        delta = "" if (q is None or prior is None) else f"  ({q - prior:+.1f})"
        ok = sum(1 for r in runs if r.checks_ok)
        print(f"  {name:<{width}}  checks {ok}/{len(runs)} runs clean   "
              f"quality {'--' if q is None else f'{q:5.1f}'}{delta}")

    total = suite_quality(results)
    was = (baseline or {}).get("suite_quality")
    line = f"\n  SUITE QUALITY  {'--' if total is None else total}"
    if was is not None and total is not None:
        line += f"   baseline {was}   delta {total - was:+.1f}"
    print(line)


def _print_gate(reasons) -> None:
    if not reasons:
        print("\n  GATE: PASS -- safe to deploy")
        return
    print("\n  GATE: BLOCKED")
    for r in reasons:
        print(f"    - {r}")


# ----------------------------------------------------------------------- main

async def _run_all(names, repeat: int, use_judge: bool, verbose: bool,
                   scenarios: list[Scenario] | None = None):
    scenarios = scenarios or [s for s in SCENARIOS if not names or s.name in names]
    # A typo in --scenario used to run nothing at all and report ALL PASS,
    # which is the most expensive kind of green.
    if names and (unknown := set(names) - {s.name for s in scenarios}):
        raise SystemExit(f"unknown scenario(s): {', '.join(sorted(unknown))}")

    results, traces = [], {}

    for scenario in scenarios:
        print(f"\n=== {scenario.name} ===")
        print(f"    {scenario.description}")
        for i in range(repeat):
            trace = await _run_scenario(scenario)
            failures = [msg for check in scenario.checks if (msg := check(trace))]

            verdict = None
            if use_judge:
                verdict = judge(QUERY, trace.tool_calls, trace.tool_results,
                                trace.final_text)

            result = build_result(scenario.name, trace, failures,
                                  len(scenario.checks), verdict,
                                  scenario.description)
            results.append(result)
            # Only the last repeat's trace is kept for the saved run. Three
            # near-identical reports in the file would make align.py's
            # sample redundant without making it larger.
            traces[scenario.name] = trace

            tag = f" (run {i + 1}/{repeat})" if repeat > 1 else ""
            _print_scenario(result, trace, verbose, tag)

    return results, traces


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Eval harness for the gameday agent")
    p.add_argument("--scenario", action="append",
                   help="run only this scenario (repeatable flag for several)")
    p.add_argument("--repeat", type=int, default=1,
                   help="repeat each scenario N times (flake-hunting)")
    p.add_argument("--no-judge", action="store_true",
                   help="skip the LLM judge: programmatic checks only, no cost")
    p.add_argument("--save-baseline", action="store_true",
                   help="freeze this run's scores as the bar future runs must clear")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="print every judge score and every full report")
    p.add_argument("--replay", metavar="SEASON:WEEK",
                   help="run real lineups on a real past slate from local "
                        "parquet (e.g. 2021:8) instead of the fixtures")
    p.add_argument("--count", type=int, default=8,
                   help="how many replay lineups to generate (default 8)")
    args = p.parse_args()

    use_judge = not args.no_judge
    if use_judge:
        print(f"judge: {JUDGE_MODEL}   agent: {os.environ['GEMINI_MODEL']}")

    replay_scenarios = None
    if args.replay:
        if args.save_baseline:
            raise SystemExit(
                "--replay cannot set the baseline. The baseline is the bar "
                "for the fixed regression suite; replay scenarios are "
                "generated, vary by week, and carry almost no checks."
            )
        try:
            season, week = (int(x) for x in args.replay.split(":"))
        except ValueError:
            raise SystemExit("--replay takes SEASON:WEEK, e.g. 2021:8")
        replay_scenarios = build_replay_scenarios(season, week, args.count)
        print(f"replay: {season} week {week}, {len(replay_scenarios)} lineups")

    results, traces = asyncio.run(
        _run_all(args.scenario, args.repeat, use_judge, args.verbose,
                 replay_scenarios)
    )

    baseline = load_baseline()
    # Replay scores are not comparable to the fixture baseline -- different
    # slate, different lineups, one tool switched off -- so the comparison
    # columns are suppressed rather than printed as meaningless deltas.
    _print_summary(results, None if args.replay else baseline)

    path = save_run(results, traces)
    print(f"  saved: {path.relative_to(ROOT)}")

    if args.replay:
        print("\n  Next: python gameday_agent/eval/make_packet.py")
        sys.exit(0)

    if args.save_baseline:
        save_baseline(results)
        print(f"  baseline updated -- future runs must stay within "
              f"{QUALITY_TOLERANCE} points")
        sys.exit(0)

    reasons = gate(results, baseline)
    _print_gate(reasons)
    sys.exit(1 if reasons else 0)
