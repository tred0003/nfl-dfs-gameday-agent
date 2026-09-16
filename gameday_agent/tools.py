"""The five tools the agent can call.

Ported from src/agent.py with one deliberate change: every tool returns a dict
carrying a `status`, not a bare list.

The reason is that a bare list cannot tell the model apart two very different
situations. `get_lineup_status()` returning [] used to mean either "your lineup
is clean" or "MY_LINEUP was never configured, I have no idea who you roster" --
and the model, seeing an empty list, would confidently report the first. A
status field lets a tool say it failed, and the agent say so rather than invent
a clean bill of health.

Everything else about how tools work is unchanged from the raw-SDK build: the
function signature and the docstring below it ARE the specification the model
sees. Nothing in a function body reaches the model. Rename a parameter and you
have edited the prompt.
"""

from __future__ import annotations

import os

import pandas as pd

from predict_slate import normalize_name, vegas_for_slate

from .slate import my_lineup_names, slate


def _f(v) -> float | None:
    """NaN-safe float, since pandas NaN is not JSON and confuses the model."""
    return None if pd.isna(v) else float(v)


def _i(v) -> int | None:
    return None if pd.isna(v) else int(v)


def _s(v, default: str = "None") -> str:
    """NaN-safe string for status fields.

    `v or default` is NOT enough: pandas leaves missing values as float NaN,
    and NaN is truthy, so that expression passes the literal float nan through
    to the model. Caught on the first live slate run -- the Ravens DST had a
    missing news_status and the tool reported `news_flag: nan`.
    """
    return default if v is None or pd.isna(v) or v == "" else str(v)


def get_lineup_status() -> dict:
    """Get the user's current DFS lineup with each player's live availability.

    Returns one record per rostered player with position, team, opponent, salary,
    injury status (None/Q/D/OUT/IR) and whether DraftKings is flagging breaking
    news on them.
    """
    names = my_lineup_names()
    if not names:
        return {
            "status": "error",
            "message": "No lineup is configured (MY_LINEUP is unset), so the "
                       "user's roster is unknown. Say so; do not report the "
                       "lineup as clean.",
        }

    players, _ = slate()
    want = {normalize_name(pd.Series([n]))[0] for n in names}
    mine = players[players["jn"].isin(want)]

    found = {normalize_name(pd.Series([n]))[0] for n in mine["name"]}
    missing = [n for n in names if normalize_name(pd.Series([n]))[0] not in found]

    return {
        "status": "ok",
        "lineup": [
            {
                "name": r["name"],
                "position": r["position"],
                "team": r["team"],
                "game": r["game"],
                "salary": int(r["salary"]),
                "injury_status": _s(r["status"]),
                "news_flag": _s(r["news_status"]),
                "dk_points_per_game": _f(r["dk_fppg"]),
                "opponent_rank_vs_position": _i(r["dk_oprk"]),
            }
            for _, r in mine.iterrows()
        ],
        # A rostered player who is not on the slate at all is a real signal --
        # usually a name mismatch, occasionally a player on a different slate.
        "not_found_on_slate": missing,
    }


def get_slate_weather() -> dict:
    """Get weather for every game on the slate.

    `is_dome` matters: DraftKings reports the outdoor forecast even for covered
    stadiums, so a dome game showing rain is not actually affected.
    """
    _, games = slate()
    return {
        "status": "ok",
        "games": [
            {
                "game": g["game"],
                "venue": g["venue"],
                "is_dome": bool(g["is_dome"]),
                "conditions": g["weather_icon"],
            }
            for _, g in games.iterrows()
        ],
    }


def get_vegas_context() -> dict:
    """Get each team's Vegas implied point total for this week.

    Implied team total is how many points the betting market expects a team to
    score. It was the strongest non-form predictor in backtesting -- a player in
    a 28-point offense is in a materially better spot than one in a 19-point
    offense, regardless of talent.
    """
    season = int(os.environ.get("SEASON", "2026"))
    week = int(os.environ.get("WEEK", "1"))

    v = vegas_for_slate(season, week)
    v = v.dropna(subset=["implied_team_total"]).drop_duplicates("team")
    if v.empty:
        return {
            "status": "error",
            "message": f"No Vegas lines available for {season} week {week}.",
        }

    return {
        "status": "ok",
        "season": season,
        "week": week,
        "teams": [
            {
                "team": r["team"],
                "implied_team_total": round(float(r["implied_team_total"]), 2),
                "opponent_implied_total": round(float(r["implied_opp_total"]), 2),
            }
            for _, r in v.iterrows()
        ],
    }


def find_replacements(position: str, max_salary: int, min_salary: int = 3000) -> dict:
    """Find available players at a position within a salary range.

    Use this when a rostered player is OUT or Doubtful and needs replacing.
    Excludes anyone who is themselves OUT or on IR. Returns the top options by
    DraftKings points per game.
    """
    players, _ = slate()
    p = players[
        players["position"].eq(position.upper())
        & players["salary"].between(min_salary, max_salary)
        & ~players["status"].isin(["OUT", "IR"])
    ].copy()
    p = p.sort_values("dk_fppg", ascending=False).head(8)

    return {
        "status": "ok",
        "position": position.upper(),
        "salary_range": [min_salary, max_salary],
        "options": [
            {
                "name": r["name"],
                "team": r["team"],
                "game": r["game"],
                "salary": int(r["salary"]),
                "dk_points_per_game": _f(r["dk_fppg"]),
                "injury_status": _s(r["status"]),
            }
            for _, r in p.iterrows()
        ],
    }


def get_notable_scratches(min_salary: int = 6000) -> dict:
    """Get expensive players across the whole slate who are OUT, IR or Doubtful.

    These reshape the slate even if the user does not roster them -- their
    backups become viable and their teammates absorb the vacated workload.
    """
    players, _ = slate()
    p = players[
        players["status"].isin(["OUT", "IR", "D"]) & players["salary"].ge(min_salary)
    ]

    return {
        "status": "ok",
        "min_salary": min_salary,
        "scratches": [
            {
                "name": r["name"],
                "position": r["position"],
                "team": r["team"],
                "salary": int(r["salary"]),
                "injury_status": _s(r["status"]),
            }
            for _, r in p.nlargest(10, "salary").iterrows()
        ],
    }


TOOLS = [
    get_lineup_status,
    get_slate_weather,
    get_vegas_context,
    find_replacements,
    get_notable_scratches,
]
