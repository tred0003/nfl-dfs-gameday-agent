"""The five tools the agent can call.

Ported from src/agent.py with one deliberate change: every tool returns a dict
carrying a `status`, not a bare list.

The reason is that a bare list cannot tell the model apart two very different
situations. `get_lineup_status()` returning [] used to mean either "your lineup
is clean" or "MY_LINEUP was never configured, I have no idea who you roster" --
and the model, seeing an empty list, would confidently report the first. A
status field lets a tool say it failed, and the agent say so rather than invent
a clean bill of health.

Injury status and weather come from ESPN (espn_client.py), not DK, as of
September 2026 -- see espn_client.py's module docstring for why. DK's
draftables call (slate.py) is still consulted for salary and roster
structure, since nothing else has that. `_with_espn_status` is the seam:
every tool that needs to know if a player can play goes through it rather
than reading DK's own `status`/`news_status` columns directly.

Everything else about how tools work is unchanged from the raw-SDK build: the
function signature and the docstring below it ARE the specification the model
sees. Nothing in a function body reaches the model. Rename a parameter and you
have edited the prompt.
"""

from __future__ import annotations

import os

import pandas as pd
import requests

from predict_slate import normalize_name, vegas_for_slate

from .slate import espn_injuries, espn_scoreboard, my_lineup_names, slate


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


def _unavailable(e: Exception) -> dict:
    """A live-data fetch (DK or ESPN) failed -- both are unofficial, undocumented
    endpoints and have each been seen returning a transient 403 in testing.

    Returned instead of letting the exception crash the whole agent turn, so
    the model can say what could not be checked -- the same reasoning as the
    `status` field on every tool's happy-path return.
    """
    return {
        "status": "error",
        "message": f"Live slate/injury data is unavailable right now ({e}). "
                   "Do not report the lineup or slate as clean without it.",
    }


def _with_espn_status(players: pd.DataFrame) -> pd.DataFrame:
    """Left-join ESPN's injury report onto a DK players frame, by name.

    ESPN's `status` replaces DK's own `status`/`news_status` columns entirely
    -- it is fresher and, unlike DK's `news_status`, not 36%-of-the-slate
    noise (see espn_client.py). A player absent from ESPN's report gets
    status None, same meaning as ESPN's own "Active".
    """
    inj = espn_injuries()[["jn", "status", "body_part", "return_date", "comment"]]
    inj = inj.drop_duplicates("jn", keep="last")
    return players.drop(columns=["status", "news_status"], errors="ignore").merge(
        inj, on="jn", how="left"
    )


def _injury_detail(r) -> str | None:
    """Human-readable injury context, or None if there is nothing to say.

    Gated on an actual designation (status not None): ESPN attaches a
    `comment` to EVERY player who has ever been on a report, including
    healthy ones -- and for a healthy player that comment is a last-game
    recap blurb, not injury news. Surfacing it regardless of status was
    caught live-testing this: a perfectly healthy player came back with
    "caught 8 of 9 targets for 92 yards and two touchdowns" under
    `injury_detail`, which reads like the model should be worried about it.
    """
    if r.get("status") is None or pd.isna(r.get("status")):
        return None

    parts = []
    if r.get("body_part") is not None and not pd.isna(r.get("body_part")):
        parts.append(str(r["body_part"]))
    if r.get("return_date") is not None and not pd.isna(r.get("return_date")):
        parts.append(f"est. return {r['return_date']}")
    if r.get("comment") is not None and not pd.isna(r.get("comment")):
        parts.append(str(r["comment"]))
    return " -- ".join(parts) if parts else None


def get_lineup_status() -> dict:
    """Get the user's current DFS lineup with each player's live availability.

    Returns one record per rostered player with position, team, opponent,
    salary, injury status (None/Q/D/OUT/IR, from ESPN's injury report) and an
    `injury_detail` string (body part, estimated return date, and/or a
    sourced note) when ESPN has one -- None means there is nothing to report,
    not that it was not checked.

    `dk_points_per_game` can be null for a player DK has not accumulated
    enough games on yet (a rookie, an early-season week, a recent trade).
    Null there means "no average yet", nothing more -- it is not a sign the
    player is inactive, fake, or needs replacing. Use `injury_status` and
    `not_found_on_slate` to judge whether someone can actually play; a
    healthy player belongs in the report even with a null points-per-game.
    """
    names = my_lineup_names()
    if not names:
        return {
            "status": "error",
            "message": "No lineup is configured (MY_LINEUP is unset), so the "
                       "user's roster is unknown. Say so; do not report the "
                       "lineup as clean.",
        }

    try:
        players, _ = slate()
        players = _with_espn_status(players)
    except requests.RequestException as e:
        return _unavailable(e)
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
                "injury_detail": _injury_detail(r),
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
    """Get weather for this week's NFL games, from ESPN.

    `is_dome` matters: a dome game is not affected by outdoor conditions no
    matter what the forecast says.

    `weather_posted` matters just as much: ESPN does not publish a forecast
    until close to kickoff. When it is false, no forecast exists yet -- that
    is "unknown", not "clear skies". Say weather has not been posted rather
    than reporting a dome-free game as fine.
    """
    try:
        games = espn_scoreboard()
    except requests.RequestException as e:
        return _unavailable(e)
    return {
        "status": "ok",
        "games": [
            {
                "game": g["game"],
                "venue": g["venue"],
                "is_dome": bool(g["is_dome"]) if not pd.isna(g["is_dome"]) else None,
                "weather_posted": bool(g["weather_posted"]),
                "conditions": _s(g["weather_condition"], default="not posted"),
                "temperature_f": _f(g["weather_temp_f"]),
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


def find_replacements(position: str, max_salary: int, min_salary: int = 0) -> dict:
    """Find available players at a position within a salary range.

    Use this when a rostered player is OUT or Doubtful and needs replacing.
    Excludes anyone who is themselves OUT or on IR. Returns the top options by
    DraftKings points per game.

    `min_salary` defaults to 0, not some assumed floor -- DK's real minimum
    varies by position (DST as low as $2000, TE $2500 on a slate checked live)
    and is often below what a cheap OUT player like a min-salary TE actually
    costs. A higher default silently returned zero options for exactly the
    player who most needed one -- found by the eval harness's emergency_out
    scenario, not by guessing.
    """
    try:
        players, _ = slate()
        players = _with_espn_status(players)
    except requests.RequestException as e:
        return _unavailable(e)
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
    try:
        players, _ = slate()
        players = _with_espn_status(players)
    except requests.RequestException as e:
        return _unavailable(e)
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
                "injury_detail": _injury_detail(r),
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
