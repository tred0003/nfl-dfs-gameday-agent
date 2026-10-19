"""Turning a run into numbers: the trace, the trajectory metrics, the scorecard.

The harness used to answer one question -- did every assertion pass? That is
the right question for deterministic software and the wrong one here. "It
works" is not a binary for an agent; it works some percentage of the time,
and the only useful thing to know about a change is whether that percentage
moved. PASS/FAIL cannot answer that. A score can.

So a run now produces three separate numbers, kept separate on purpose
because they fail for different reasons and get fixed in different places:

  checks   programmatic assertions. Deterministic, free, instant. These are
           regression guards: each one exists because the agent got that
           exact thing wrong at least once. A drop here is a hard failure --
           no averaging, no judgment, it either called find_replacements or
           it did not.

  quality  the LLM judge's weighted rubric score, 0-100. Subjective, costs
           a model call, and only trustworthy to the extent align.py has
           shown it tracks a human. A drop here is a regression signal, not
           a verdict.

  trajectory  what the agent DID, not what it said. Tool count, repeated
           calls, wall time. This is the half of evaluation with no
           equivalent in normal testing and the half that catches silent
           failures -- a correct-looking report produced without ever
           calling the tool that would have grounded it.

A run can score 100 on quality and still be broken, which is exactly why
the three do not get averaged into one headline number.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
RUNS_DIR = EVAL_DIR / "runs"
BASELINE = EVAL_DIR / "baseline.json"

# How far quality may slide before the suite fails.
#
# This started at 5.0, picked out of the air. Then --repeat 3 was actually
# run, and the noise floor turned out to be nothing like that. Three
# IDENTICAL runs of clean_lineup scored 33.9, 78.6 and 55.4. emergency_out
# scored 33.9, 78.6 and 92.9. That is a spread of roughly fifty points with
# no code change between runs -- the agent samples at temperature 0.2, it
# picks a different subset of tools, and the judge is scoring genuinely
# different reports.
#
# So a 5-point gate would have fired on pure noise most days, and a gate
# that cries wolf is one somebody disables within a week. The honest
# conclusion is harder than picking a bigger number: a SINGLE run's quality
# score is not a measurement of anything. Averaging is what makes it one.
QUALITY_TOLERANCE = 10.0

# Below this many runs, a scenario's quality score is not gated at all --
# only its programmatic checks are. Averaging three samples of a +/-25 point
# distribution leaves a standard error near 15, which is still wide; three is
# the point where the number starts carrying information, not the point where
# it becomes precise. The suite score is gated regardless, because it averages
# every scenario and every repeat and is correspondingly steadier.
MIN_RUNS_FOR_QUALITY_GATE = 3


@dataclass
class Trace:
    """One agent run, recorded. The unit everything else scores."""

    tool_calls: list[tuple[str, dict]] = field(default_factory=list)
    tool_results: list[tuple[str, dict]] = field(default_factory=list)
    final_text: str = ""
    elapsed_s: float = 0.0
    # Summed usage_metadata over every model call in the run (token_cost.py).
    tokens: dict = field(default_factory=dict)
    # One entry per model call: its tokens, cost, and what it was for.
    model_calls: list = field(default_factory=list)
    # One entry per Gemini call made INSIDE a tool (metering.py): screen,
    # news extraction, claim check, search. Not in `tokens` above.
    subcalls: list = field(default_factory=list)
    # The lineup the scenario handed the agent, for the dashboard.
    lineup: list = field(default_factory=list)

    def tool_names(self) -> set[str]:
        return {n for n, _ in self.tool_calls}

    def calls_to(self, tool: str) -> list[dict]:
        """Every argument dict this tool was called with, in order.

        Checks that assert on arguments need this. "Was find_replacements
        called" is a weak assertion -- called with the wrong position it is
        still a wrong answer, and the final report can look perfectly
        reasonable while containing replacements for the wrong slot.
        """
        return [args for name, args in self.tool_calls if name == tool]

    def repeated_calls(self) -> list[str]:
        """Tools called more than once with identical arguments.

        Not an error by itself, but it is what the early stage of a loop
        looks like, and it costs a model round trip every time.
        """
        seen, repeats = set(), []
        for name, args in self.tool_calls:
            key = (name, json.dumps(args, sort_keys=True, default=str))
            if key in seen:
                repeats.append(name)
            seen.add(key)
        return repeats


@dataclass
class ScenarioResult:
    scenario: str
    # Carried into the saved run so the grading packet can tell a reader what
    # lineup they are looking at. A replay scenario's name is a slug; its
    # description is the roster, which is the part a human needs.
    description: str
    checks_passed: int
    checks_total: int
    failures: list[str]
    quality: float | None          # None when the judge was skipped or failed
    judge_scores: dict[str, int]
    judge_reasons: dict[str, str]
    judge_error: str | None
    tool_count: int
    unique_tools: int
    repeats: list[str]
    elapsed_s: float
    report: str
    # What the run cost. Defaults so run files saved before 10/16 still load.
    tokens: dict = field(default_factory=dict)
    cost_usd: float | None = None
    judge_tokens: dict = field(default_factory=dict)
    judge_cost_usd: float | None = None
    # `cost_usd` is the agent's own loop. Sub-calls were added 2026-10-19;
    # older run files have none, so their total is the agent cost alone.
    subcalls: list = field(default_factory=list)
    subcall_cost_usd: float | None = None
    total_cost_usd: float | None = None

    @property
    def checks_ok(self) -> bool:
        return self.checks_passed == self.checks_total


def build_result(scenario_name: str, trace: Trace, failures: list[str],
                 total_checks: int, verdict,
                 description: str = "", agent_model: str = "",
                 judge_model: str = "") -> ScenarioResult:
    from token_cost import cost_usd, subcall_cost

    agent_cost = cost_usd(trace.tokens, agent_model)
    sub_cost = subcall_cost(trace.subcalls)
    judge_tokens = dict(getattr(verdict, "tokens", {}) or {})
    return ScenarioResult(
        scenario=scenario_name,
        description=description,
        checks_passed=total_checks - len(failures),
        checks_total=total_checks,
        failures=failures,
        quality=None if (verdict is None or verdict.error) else verdict.percent,
        judge_scores=dict(verdict.scores) if verdict else {},
        judge_reasons=dict(verdict.reasons) if verdict else {},
        judge_error=verdict.error if verdict else None,
        tool_count=len(trace.tool_calls),
        unique_tools=len(trace.tool_names()),
        repeats=trace.repeated_calls(),
        elapsed_s=round(trace.elapsed_s, 1),
        report=trace.final_text,
        tokens=dict(trace.tokens),
        cost_usd=agent_cost,
        subcalls=list(trace.subcalls),
        subcall_cost_usd=sub_cost,
        total_cost_usd=None if agent_cost is None else round(agent_cost + sub_cost, 6),
        judge_tokens=judge_tokens,
        judge_cost_usd=cost_usd(judge_tokens, judge_model) if judge_tokens else None,
    )


# --------------------------------------------------------------- persistence

# Longest list a saved tool result keeps. get_slate_weather and
# get_vegas_context return a row per game and per team, which is fine; this
# only bites if a tool ever returns the whole player board.
MAX_SAVED_ROWS = 40


def _cap(result):
    """Trim a tool result to something a JSON file can hold comfortably."""
    if not isinstance(result, dict):
        return result
    out = {}
    for k, v in result.items():
        if isinstance(v, list) and len(v) > MAX_SAVED_ROWS:
            out[k] = v[:MAX_SAVED_ROWS] + [f"... {len(v) - MAX_SAVED_ROWS} more"]
        else:
            out[k] = v
    return out


def save_run(results: list[ScenarioResult], traces: dict[str, Trace]) -> Path:
    """Write the whole suite run to disk.

    Two reasons, and the second is the one that matters. First, a scorecard
    printed to a terminal is gone as soon as you close it, and the question
    "did that prompt edit help?" needs yesterday's numbers. Second, align.py
    needs real saved runs to hand a human for scoring -- you cannot check
    that the judge agrees with you about reports you can no longer read.
    """
    RUNS_DIR.mkdir(exist_ok=True)
    path = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    payload = {
        "at": datetime.now().isoformat(timespec="seconds"),
        "results": [asdict(r) for r in results],
        "traces": {
            name: {
                "tool_calls": [[n, a] for n, a in t.tool_calls],
                # Results ARE saved, after initially being left out for size.
                # Grading grounding means asking "is this claim in the data
                # the agent was handed", and that is unanswerable without the
                # data. A replayed slate cannot be regenerated identically
                # from a saved lineup either, so this is the only record.
                # Each result is capped rather than the whole set dropped.
                "tool_results": [
                    [n, _cap(r)] for n, r in t.tool_results
                ],
                "final_text": t.final_text,
                "model_calls": t.model_calls,
                "lineup": t.lineup,
            }
            for name, t in traces.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def suite_quality(results: list[ScenarioResult]) -> float | None:
    scored = [r.quality for r in results if r.quality is not None]
    return round(statistics.mean(scored), 1) if scored else None


def by_scenario(results: list[ScenarioResult]) -> dict[str, list[ScenarioResult]]:
    """Group runs by scenario name, because --repeat produces several per name.

    Repeats are how flakiness becomes visible, so they must not be collapsed
    on the way in. They are averaged only where a single number is required
    -- comparing against a baseline -- and a scenario that passed twice and
    failed once is still a failing scenario, never a two-thirds pass.
    """
    grouped: dict[str, list[ScenarioResult]] = {}
    for r in results:
        grouped.setdefault(r.scenario, []).append(r)
    return grouped


def mean_quality(runs: list[ScenarioResult]) -> float | None:
    scored = [r.quality for r in runs if r.quality is not None]
    return round(statistics.mean(scored), 1) if scored else None


def load_baseline() -> dict | None:
    if not BASELINE.exists():
        return None
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def save_baseline(results: list[ScenarioResult]) -> None:
    """Freeze the current scores as the bar future runs must clear.

    Only ever called explicitly, with --save-baseline. A harness that
    silently rebaselines after every run cannot detect a regression: each
    bad result quietly becomes the new normal, and the suite reports green
    all the way down.
    """
    BASELINE.write_text(json.dumps({
        "at": datetime.now().isoformat(timespec="seconds"),
        "suite_quality": suite_quality(results),
        "scenarios": {
            name: {
                "quality": mean_quality(runs),
                "runs": len(runs),
                # Stored so the tolerance stays answerable rather than
                # remembered. Anyone asking "is a six-point drop real?" can
                # read the spread that produced this baseline instead of
                # guessing, and a scenario whose spread is wider than the
                # tolerance is telling you its quality score cannot gate.
                "spread": _spread(runs),
            }
            for name, runs in by_scenario(results).items()
        },
    }, indent=2), encoding="utf-8")


def _spread(runs: list[ScenarioResult]) -> dict | None:
    scored = [r.quality for r in runs if r.quality is not None]
    if len(scored) < 2:
        return None
    return {"min": min(scored), "max": max(scored),
            "range": round(max(scored) - min(scored), 1)}


def gate(results: list[ScenarioResult], baseline: dict | None) -> list[str]:
    """The quality gate. Returns the reasons to block; empty means ship.

    This is the function that turns evaluation from something you look at
    into something that stops a bad change, which is the whole point of
    wiring an eval suite into CI.
    """
    reasons = []
    grouped = by_scenario(results)

    for name, runs in grouped.items():
        bad = [r for r in runs if not r.checks_ok]
        if bad:
            reasons.append(
                f"{name}: programmatic checks failed in {len(bad)} of "
                f"{len(runs)} run(s)"
            )

    if not baseline:
        return reasons

    now = suite_quality(results)
    was = baseline.get("suite_quality")
    if now is not None and was is not None and now < was - QUALITY_TOLERANCE:
        reasons.append(
            f"suite quality {now} is more than {QUALITY_TOLERANCE} below "
            f"the baseline {was}"
        )

    for name, runs in grouped.items():
        # One run of a scenario says nothing about its quality -- see the
        # measured spread above. Gating on it would block good changes at
        # random, which teaches everyone to pass --no-judge.
        if len(runs) < MIN_RUNS_FOR_QUALITY_GATE:
            continue

        prior_entry = baseline.get("scenarios", {}).get(name) or {}
        prior = prior_entry.get("quality")

        # If this scenario's own recorded spread is wider than the tolerance,
        # its quality score cannot distinguish a regression from a rerun, so
        # it does not gate. Measured, not assumed: the two outage scenarios
        # come in at a 0-4 point spread and gate fine, while the scenarios
        # where the agent picks among six tools and writes a long report run
        # 29-59 points apart with nothing changed. Variance tracks how much
        # the agent has to decide.
        #
        # Those scenarios are not unguarded -- their programmatic checks are
        # still hard failures, and they still move the suite average. They
        # just do not get to block a deploy on their own noise.
        spread = (prior_entry.get("spread") or {}).get("range")
        if spread is not None and spread > QUALITY_TOLERANCE:
            continue

        now_q = mean_quality(runs)
        if now_q is not None and prior is not None and now_q < prior - QUALITY_TOLERANCE:
            reasons.append(f"{name}: quality {now_q} vs baseline {prior}")

    return reasons
