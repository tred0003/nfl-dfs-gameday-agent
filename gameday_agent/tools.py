"""The six tools the agent can call.

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

from predict_slate import TEAM_FIXES, normalize_name, vegas_for_slate

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


def _slate_teams(games: pd.DataFrame) -> set[str]:
    """Team codes playing on this slate, normalized to one spelling.

    The user plays the Sunday main slate only -- the 1pm and 4pm windows. A
    Thursday, Sunday-night or Monday-night game is not on it and its players
    cannot be rostered, so anything about those games is noise in a report
    whose whole job is to earn an interruption.

    Three sources spell teams three ways (ESPN `WSH`/`LAR`, DK `WAS`,
    nflverse `LA`), so everything goes through TEAM_FIXES before comparison.
    """
    return set(
        pd.concat([games["home_team"], games["away_team"]])
        .replace(TEAM_FIXES)
        .dropna()
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
    """Get weather for the games on this week's slate, from two sources.

    Only games the user can actually roster are returned -- the Sunday main
    slate, which is the 1pm and 4pm windows. Thursday, Sunday-night and
    Monday-night games are excluded because no player in them can be in the
    lineup, so their weather cannot matter.

    TWO INDEPENDENT SOURCES are reported per game and they do not always
    agree. Report a disagreement as a disagreement -- do not silently pick
    one and present it as settled.

    ESPN (`conditions`, `temperature_f`):
    - `conditions` is a one-word summary such as "Cloudy" or "Mostly sunny".
      There is NO precipitation probability in this feed, so "Cloudy" at a 5%
      chance of rain and "Cloudy" at a 60% chance are the identical string.
      A non-rain word means the label is not a rain word -- it does not mean
      rain has been ruled out.
    - `temperature_f` is the DAILY HIGH, not the temperature at kickoff. A
      1pm game on a day that turns severe by 4pm still reads as the high.

    DraftKings (`dk_weather_icon`): DK's own per-game forecast icon, e.g.
    "rain", "clear-day", "partly-cloudy-day". Coarser than ESPN -- no
    temperature, no timestamp, and it may be stale -- but it is a second
    opinion, and it has flagged rain on games ESPN labels merely cloudy. If
    it says rain and ESPN does not, say both and let the user decide.

    `is_dome` (ESPN) and `dk_is_dome` (DraftKings): a dome game is not
    affected by outdoor conditions no matter what either forecast says. These
    two disagree on retractable-roof stadiums. When they differ, the roof is
    operable and whether it is open on Sunday is genuinely unknown -- treat
    the weather as possibly relevant rather than assuming indoors.

    `weather_posted` distinguishes "no forecast exists yet" from "checked, no
    weather issue". When it is false that is "unknown", not "clear skies".

    Neither source reports wind. Do not comment on wind.
    """
    try:
        _, slate_games = slate()
        games = espn_scoreboard()
    except requests.RequestException as e:
        return _unavailable(e)

    games = games.copy()
    games["home_n"] = games["home_team"].replace(TEAM_FIXES)
    games["away_n"] = games["away_team"].replace(TEAM_FIXES)
    games = games[
        games["home_n"].isin(_slate_teams(slate_games))
        & games["away_n"].isin(_slate_teams(slate_games))
    ]

    # DK's own forecast, keyed by the same normalized team pair -- the two
    # feeds name games differently ("Green Bay Packers at New York Jets" vs
    # "GB @ NYJ"), so the team codes are the only reliable join key.
    dk = slate_games.copy()
    dk["home_n"] = dk["home_team"].replace(TEAM_FIXES)
    dk["away_n"] = dk["away_team"].replace(TEAM_FIXES)
    dk_by_pair = {
        (r["home_n"], r["away_n"]): (r.get("weather_icon"), r.get("is_dome"))
        for _, r in dk.iterrows()
    }

    out = []
    for _, g in games.iterrows():
        dk_icon, dk_dome = dk_by_pair.get((g["home_n"], g["away_n"]), (None, None))
        out.append({
            "game": g["game"],
            "venue": g["venue"],
            "is_dome": bool(g["is_dome"]) if not pd.isna(g["is_dome"]) else None,
            "dk_is_dome": None if dk_dome is None or pd.isna(dk_dome) else bool(dk_dome),
            "weather_posted": bool(g["weather_posted"]),
            "conditions": _s(g["weather_condition"], default="not posted"),
            "temperature_f": _f(g["weather_temp_f"]),
            "dk_weather_icon": _s(dk_icon, default="unavailable"),
        })

    return {"status": "ok", "games": out}


def get_vegas_context() -> dict:
    """Get the implied point total for each team playing on this slate.

    Implied team total is how many points the betting market expects a team to
    score. It was the strongest non-form predictor in backtesting -- a player in
    a 28-point offense is in a materially better spot than one in a 19-point
    offense, regardless of talent.

    Restricted to teams on the Sunday main slate, so a team playing Thursday,
    Sunday night or Monday night does not appear -- none of its players can be
    rostered.
    """
    season = int(os.environ.get("SEASON", "2026"))
    week = int(os.environ.get("WEEK", "1"))

    try:
        _, slate_games = slate()
    except requests.RequestException as e:
        return _unavailable(e)

    v = vegas_for_slate(season, week)
    v["team"] = v["team"].replace(TEAM_FIXES)
    v = v.dropna(subset=["implied_team_total"]).drop_duplicates("team")
    v = v[v["team"].isin(_slate_teams(slate_games))]
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


_cb_cache: dict = {}


def _cb_board():
    """Cornerback board, cached for the life of the process.

    Unlike the slate this has no short TTL. Grades are built from a full prior
    season plus whatever of the current one nflverse has published, and that
    does not change during a Sunday morning -- whereas a warm Agent Platform
    process would otherwise rebuild it, including two parquet downloads, on
    every tool call.
    """
    import time

    from cb_coverage import cb_board

    now = time.monotonic()
    if _cb_cache and now - _cb_cache["at"] < 3600:
        return _cb_cache["df"]
    df = cb_board()
    _cb_cache.update(df=df, at=now)
    return df


def get_cb_matchups() -> dict:
    """Get the cornerbacks each of the user's receivers and tight ends will face.

    `coverage_grade` runs 0-100 among full-time corners, higher meaning harder
    to throw on. It blends passer rating allowed, yards allowed per target, and
    how often quarterbacks throw his way per coverage snap. A grade of None
    means the player is a backup or has too little data, NOT that he is bad.

    THIS CANNOT TELL YOU WHO COVERS WHOM. No free data source publishes per-snap
    alignment, so there is no way to know whether a corner shadows a particular
    receiver or simply plays one side. `role` is inferred from how deep he is
    thrown at -- "boundary" corners see deeper targets than "slot" corners --
    and that is a tendency, not an assignment. Say "likely draws" or "may see",
    never "will be covered by".

    Grades come mostly from LAST season, because this season's defensive data
    lags by weeks. That is reasonable for corner talent, which is stable, but do
    not present it as current form.

    `injury_status` matters more than the grade. An elite corner who is OUT
    turns a hard matchup into a soft one, and the backup behind him is usually
    the weakest graded player on the list.
    """
    names = my_lineup_names()
    if not names:
        return {
            "status": "error",
            "message": "No lineup is configured (MY_LINEUP is unset), so there "
                       "is nothing to find matchups for.",
        }

    try:
        players, _ = slate()
        board = _cb_board()
    except (requests.RequestException, OSError, RuntimeError) as e:
        return _unavailable(e)

    want = {normalize_name(pd.Series([n]))[0] for n in names}
    mine = players[players["jn"].isin(want) & players["position"].isin(["WR", "TE"])]

    out = []
    for _, r in mine.iterrows():
        # "PIT @ NE" -- the opponent is whichever side is not his own team.
        sides = [t.strip() for t in str(r["game"]).replace("@", " ").split()]
        opp = next((t for t in sides if t != r["team"]), None)
        if opp is None:
            continue
        corners = board[board["team"].eq(opp)].head(4)
        out.append({
            "player": r["name"],
            "position": r["position"],
            "opponent": opp,
            "game": r["game"],
            "opponent_corners": [
                {
                    "name": c["pfr_player_name"],
                    "coverage_grade": _f(c["coverage_grade"]),
                    "role": _s(c["role"], default="unknown"),
                    "passer_rating_allowed": _f(c["passer_rating_allowed"]),
                    "yards_per_target": _f(c["yards_per_target"]),
                    "injury_status": _s(c["injury_status"]),
                    "snap_share": _f(c["snap_share"]),
                }
                for _, c in corners.iterrows()
            ],
        })

    return {
        "status": "ok",
        "matchups": out,
        "shadow_data_available": False,
        "note": "Grades are mostly prior-season. No alignment data exists, so "
                "which corner covers which receiver is unknown.",
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
    get_cb_matchups,
    get_slate_weather,
    get_vegas_context,
    find_replacements,
    get_notable_scratches,
]
