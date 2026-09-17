"""Fixed scenarios for the gameday agent eval harness.

Each one freezes MY_LINEUP and overrides specific rows on top of the
synthetic fixtures in fixtures.py (see that module's docstring for why they
are synthetic, not a frozen real pull), so a run is exactly reproducible
regardless of what DK/ESPN say on any given day -- and regardless of whether
they're reachable at all, which DK currently is not.

Checks are lenient substring assertions against the final report text and
the tool-call trace, not exact-match -- the model's wording varies run to
run under identical tool data; what has to stay fixed is what it claims, not
how it phrases it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .fixtures import LINEUP_PLAYERS

BASELINE_LINEUP = [p["name"] for p in LINEUP_PLAYERS]

# Phrases seen in the wild during manual testing, all false: a perfectly
# healthy player (null dk_points_per_game, no injury report) got called
# "not an active NFL player" three runs running before tools.py's docstring
# was fixed to explain what a null points-per-game means. Any scenario with
# a healthy lineup checks that this class of claim doesn't come back.
FAKE_PLAYER_PHRASES = [
    "not an active", "not a real player", "does not exist",
    "isn't a real", "not a valid player", "no longer in the nfl",
]


@dataclass
class Scenario:
    name: str
    description: str
    lineup: list[str]
    # Names to strip any existing ESPN report from, so the fixture's *real*
    # injury data (which changes day to day) can't leak into a scenario that
    # is supposed to be deterministic.
    espn_clear: list[str] = field(default_factory=list)
    # name -> fields to set on a synthetic ESPN injury row for that player.
    espn_overrides: dict[str, dict] = field(default_factory=dict)
    espn_down: bool = False
    dk_down: bool = False
    checks: list[Callable] = field(default_factory=list)


def must_mention(name: str) -> Callable:
    def check(trace) -> str | None:
        if name.lower() not in trace.final_text.lower():
            return f'expected the report to mention "{name}"'
        return None
    return check


def must_not_contain_any(*phrases: str) -> Callable:
    def check(trace) -> str | None:
        low = trace.final_text.lower()
        hit = [p for p in phrases if p.lower() in low]
        if hit:
            return f"report should not contain {hit}, but did"
        return None
    return check


def must_call_tool(tool_name: str) -> Callable:
    def check(trace) -> str | None:
        if tool_name not in trace.tool_names():
            return f'expected tool "{tool_name}" to be called, but it was not'
        return None
    return check


def must_not_call_tool(tool_name: str) -> Callable:
    def check(trace) -> str | None:
        if tool_name in trace.tool_names():
            return f'expected tool "{tool_name}" NOT to be called, but it was'
        return None
    return check


def must_report_unavailable() -> Callable:
    keywords = ["unavailable", "could not", "cannot verify", "can't verify",
                "not able to check", "error", "not available"]

    def check(trace) -> str | None:
        low = trace.final_text.lower()
        if not any(k in low for k in keywords):
            return "expected the report to say data was unavailable/unverifiable"
        return None
    return check


SCENARIOS = [
    Scenario(
        name="clean_lineup",
        description="Healthy lineup, no injuries anywhere. Regression guard "
                    "for the null-dk_points_per_game hallucination.",
        lineup=BASELINE_LINEUP,
        espn_clear=BASELINE_LINEUP,
        checks=[
            must_not_call_tool("find_replacements"),
            must_not_contain_any(*FAKE_PLAYER_PHRASES),
        ],
    ),
    Scenario(
        name="emergency_out",
        description="A rostered player is ruled OUT. Agent should name him "
                    "and propose replacements at his position before "
                    "reporting the problem, per its own instructions.",
        lineup=BASELINE_LINEUP,
        espn_clear=BASELINE_LINEUP,
        espn_overrides={
            "Grant Iverson": {
                "status": "OUT",
                "body_part": "Undisclosed",
                "comment": "Iverson has been ruled out for Sunday's game.",
            },
        },
        checks=[
            must_mention("Grant Iverson"),
            must_call_tool("find_replacements"),
            must_not_contain_any(*FAKE_PLAYER_PHRASES),
        ],
    ),
    Scenario(
        name="questionable_with_detail",
        description="A rostered player is Questionable with a real injury "
                    "note. Should be named; whether to also find a "
                    "replacement is the model's judgment call, not asserted.",
        lineup=BASELINE_LINEUP,
        espn_clear=BASELINE_LINEUP,
        espn_overrides={
            "Trevon Ashby": {
                "status": "Q",
                "body_part": "Hamstring",
                "return_date": "2026-09-20",
                "comment": "Ashby (hamstring) was limited at Thursday's practice.",
            },
        },
        checks=[
            must_mention("Trevon Ashby"),
            must_not_contain_any(*FAKE_PLAYER_PHRASES),
        ],
    ),
    Scenario(
        name="espn_outage",
        description="ESPN (injury/weather source) is down. Agent must say "
                    "so, not report the lineup as clean.",
        lineup=BASELINE_LINEUP,
        espn_down=True,
        checks=[
            must_report_unavailable(),
        ],
    ),
    Scenario(
        name="dk_outage",
        description="DK (salary/roster source) is down -- today's actual "
                    "real-world state. Agent must say so, not report clean.",
        lineup=BASELINE_LINEUP,
        dk_down=True,
        checks=[
            must_report_unavailable(),
        ],
    ),
]
