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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

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

import gameday_agent.news as news_mod  # noqa: E402
import gameday_agent.tools as tools_mod  # noqa: E402
from gameday_agent.slate import LINEUP_STATE_KEY  # noqa: E402
from gameday_agent.agent import root_agent  # noqa: E402
from gameday_agent.eval.fixtures import (  # noqa: E402
    dk_fixture, espn_fixture, vegas_fixture,
)
from gameday_agent.eval.judge import JUDGE_MODEL, judge, judge_batch  # noqa: E402
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
from token_cost import add_usage, call_record, new_totals, subcalls_from_state  # noqa: E402

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


# news.py binds its own slate/_gather/_client. Kept so a scenario that patches
# them cannot leak its fake articles into the next scenario.
_NEWS_ORIGINALS = (news_mod.slate, news_mod._gather, news_mod._client)


def _news_articles(scenario: Scenario) -> list[dict]:
    now = datetime.now(timezone.utc)
    return [{k: v for k, v in a.items() if k != "hours_ago"}
            | {"published": now - timedelta(hours=a.get("hours_ago", 1))}
            for a in scenario.news_articles]


class _FakeSearchClient:
    """Answers Google-grounded calls with the scenario's text; passes every
    other call (news extraction, claim check) to the real client, so those
    defenses run for real against the planted article."""

    def __init__(self, real, answer: str):
        self._real, self._answer = real, answer
        self.models = self

    def generate_content(self, **kw):
        config = kw.get("config")
        if any(getattr(t, "google_search", None) for t in (getattr(config, "tools", None) or [])):
            web = SimpleNamespace(title="Fixture search result",
                                  uri="https://example.com/eval-fixture")
            meta = SimpleNamespace(grounding_chunks=[SimpleNamespace(web=web)])
            return SimpleNamespace(text=self._answer,
                                   candidates=[SimpleNamespace(grounding_metadata=meta)])
        return self._real.models.generate_content(**kw)


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


# Seconds to wait before each retry of a scenario that hit a Gemini 429.
RETRY_WAITS = (30, 60, 120)


def _is_quota_error(exc: BaseException) -> bool:
    """True for a Gemini 429 anywhere in the chain. ADK re-raises the genai
    ClientError as its own private _ResourceExhaustedError, so the class is
    not something to import -- the code on the original error is."""
    while exc is not None:
        if getattr(exc, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(exc):
            return True
        exc = exc.__cause__
    return False


async def _run_with_retry(scenario: Scenario) -> Trace:
    """A 429 is the quota talking, not the agent. The first cloud run passed
    three scenarios and then lost the whole suite to one, so a quota error
    reruns the scenario from scratch (new session, nothing carried over)
    after a wait. Anything else, and a 429 that outlasts the waits, still
    fails the run."""
    for wait in (*RETRY_WAITS, None):
        try:
            return await _run_scenario(scenario)
        except Exception as e:
            if wait is None or not _is_quota_error(e):
                raise
            print(f"  Gemini 429 -- retrying {scenario.name} in {wait}s")
            await asyncio.sleep(wait)


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

    # The lineup goes in through session state, the way the scheduled client
    # sends it -- not MY_LINEUP. With the env var set, a broken state path
    # would pass every scenario by quietly falling back to it, which is the
    # one regression this harness most needs to catch. So the env var is
    # removed outright, and a lineup tool that ignores state gets nothing.
    os.environ.pop("MY_LINEUP", None)

    tools_mod.slate = _raise if scenario.dk_down else (lambda: (players, games))
    tools_mod.espn_injuries = _raise if scenario.espn_down else (lambda: injuries)
    tools_mod.espn_scoreboard = _raise if scenario.espn_down else (lambda: scoreboard)
    # Patched for the same reason as the other three, and it was missing: the
    # Vegas numbers were the one input still coming from the real schedule,
    # which made "reproducible day to day" untrue and let a team's line
    # describe a game that is not on this fixture slate.
    tools_mod.vegas_for_slate = lambda *_a, **_kw: vegas

    news_mod.slate, news_mod._gather, news_mod._client = _NEWS_ORIGINALS
    # news.py has its own reference to slate(). Until 2026-10-19 it was only
    # pointed at the fixture when a scenario planted articles, so in every
    # other scenario get_team_news looked the fixture lineup up on the LIVE
    # DK slate, found nobody, and returned an error -- the news path was
    # untested in 8 of 9 scenarios. Now every scenario reads the fixture
    # slate, and one with no planted articles gets a quiet week, not live
    # feeds: live news would make runs incomparable day to day.
    news_mod.slate = tools_mod.slate
    articles = _news_articles(scenario) if scenario.news_articles is not None else []
    news_mod._gather = lambda *_a, **_kw: ([dict(a) for a in articles], [])
    if scenario.search_answer is not None:
        real = _NEWS_ORIGINALS[2]
        news_mod._client = lambda: _FakeSearchClient(real(), scenario.search_answer)

    from google.adk.runners import InMemoryRunner
    from google.genai import types

    runner = InMemoryRunner(agent=root_agent, app_name="gameday_eval")
    session = await runner.session_service.create_session(
        app_name="gameday_eval", user_id="eval",
        state={LINEUP_STATE_KEY: list(scenario.lineup)},
    )
    content = types.Content(role="user", parts=[types.Part(text=scenario.query or QUERY)])

    trace = Trace(tokens=new_totals(), lineup=list(scenario.lineup))
    texts: list[str] = []
    started = time.perf_counter()
    async for event in runner.run_async(
        user_id="eval", session_id=session.id, new_message=content
    ):
        # Every model response carries its usage; summed, that is the run's
        # token cost. Counted before the content check: a response that is
        # only a function call has no text but still cost tokens.
        if not getattr(event, "partial", False):
            add_usage(trace.tokens, getattr(event, "usage_metadata", None))
            # Gemini calls made inside tools (metering.py) ride in state deltas.
            actions = getattr(event, "actions", None)
            trace.subcalls += subcalls_from_state(getattr(actions, "state_delta", None))
        if not (event.content and event.content.parts):
            continue
        if not getattr(event, "partial", False):
            rec = call_record(
                getattr(event, "usage_metadata", None), os.environ["GEMINI_MODEL"],
                [p.function_call.name for p in event.content.parts
                 if getattr(p, "function_call", None)],
                any(getattr(p, "text", None) for p in event.content.parts))
            if rec:
                trace.model_calls.append(rec)
        for part in event.content.parts:
            if getattr(part, "function_call", None):
                fc = part.function_call
                trace.tool_calls.append((fc.name, dict(fc.args)))
            if getattr(part, "function_response", None):
                fr = part.function_response
                trace.tool_results.append((fr.name, fr.response))
            if getattr(part, "text", None):
                texts.append(part.text)
    # Every text part, joined -- the same rule run_report.py has used since
    # 10/9. This used to be "last text wins", which graded only the final
    # section of a report the model wrote between tool calls: the judge and
    # the checks saw less than the user would have.
    trace.final_text = "\n\n".join(t.strip() for t in texts if t.strip())
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
                   scenarios: list[Scenario] | None = None, batch_judge: bool = False):
    scenarios = scenarios or [s for s in SCENARIOS if not names or s.name in names]
    # A typo in --scenario used to run nothing at all and report ALL PASS,
    # which is the most expensive kind of green.
    if names and (unknown := set(names) - {s.name for s in scenarios}):
        raise SystemExit(f"unknown scenario(s): {', '.join(sorted(unknown))}")

    results, traces = [], {}
    pending = []  # batch judge: (scenario, trace, failures), graded after the loop

    for scenario in scenarios:
        print(f"\n=== {scenario.name} ===")
        print(f"    {scenario.description}")
        for i in range(repeat):
            trace = await _run_with_retry(scenario)
            failures = [msg for check in scenario.checks if (msg := check(trace))]

            verdict = None
            if use_judge and not batch_judge:
                verdict = judge(scenario.query or QUERY, trace.tool_calls, trace.tool_results,
                                trace.final_text)
            elif use_judge:
                pending.append((scenario, trace, failures))

            result = build_result(scenario.name, trace, failures,
                                  len(scenario.checks), verdict,
                                  scenario.description,
                                  agent_model=os.environ["GEMINI_MODEL"],
                                  judge_model=JUDGE_MODEL)
            results.append(result)
            # Only the last repeat's trace is kept for the saved run. Three
            # near-identical reports in the file would make align.py's
            # sample redundant without making it larger.
            traces[scenario.name] = trace

            tag = f" (run {i + 1}/{repeat})" if repeat > 1 else ""
            _print_scenario(result, trace, verbose, tag)

    if pending:
        print(f"\n=== judging {len(pending)} reports in one batch job (half price, slower) ===")
        verdicts = judge_batch([(s.query or QUERY, t.tool_calls, t.tool_results, t.final_text)
                                for s, t, _ in pending])
        results = []
        for (scenario, trace, failures), verdict in zip(pending, verdicts):
            result = build_result(scenario.name, trace, failures, len(scenario.checks), verdict,
                                  scenario.description,
                                  agent_model=os.environ["GEMINI_MODEL"],
                                  judge_model=JUDGE_MODEL)
            results.append(result)
            print(f"\n=== {scenario.name} (judged) ===")
            _print_scenario(result, trace, verbose, "")

    return results, traces


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Eval harness for the gameday agent")
    p.add_argument("--scenario", action="append",
                   help="run only this scenario (repeatable flag for several)")
    p.add_argument("--repeat", type=int, default=1,
                   help="repeat each scenario N times (flake-hunting)")
    p.add_argument("--no-judge", action="store_true",
                   help="skip the LLM judge: programmatic checks only, no cost")
    p.add_argument("--batch-judge", action="store_true",
                   help="grade every report in one Vertex batch job after the run: "
                        "half the judge's price, but the job can take many minutes")
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
                 replay_scenarios, batch_judge=args.batch_judge)
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
