"""Sleeper + FantasyCalc ingestion for the dynasty strategy agent.

Everything that knows about either API's JSON shape lives in THIS file, same
contract as dk_client.py and espn_client.py -- the rest of the project consumes
tidy DataFrames.

Neither API needs auth. Sleeper's read endpoints are fully public; FantasyCalc's
values endpoint is open. There is no key to configure and no credential to leak,
which is why this module takes a league id and nothing else.

    python src/sleeper_client.py --league 1357813048248664064

WHY FANTASYCALC AND NOT KEEPTRADECUT: ids.parquet carries a ktc_id, so KTC was
the obvious route. FantasyCalc turned out strictly better for this job because
every one of its records carries `sleeperId` directly -- 421/421 when checked.
That removes the crosswalk, and with it the whole class of silent join failure
that name matching and float-formatted ids keep producing elsewhere in this
project. It also ships `redraftValue` beside the dynasty `value`, which is the
buy/sell signal itself rather than a proxy for it.

CACHING: Sleeper's /players/nfl dump is 14.7 MB and they ask that it be pulled
about once a day. It is cached to disk, not memory, because a Cloud Run job
starts cold every run -- an in-process TTL like slate.py's would re-download it
on every single invocation and never once serve a hit.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import requests

SLEEPER = "https://api.sleeper.app/v1"
FANTASYCALC = "https://api.fantasycalc.com/values/current"

ROOT = Path(__file__).resolve().parent.parent

TIMEOUT = 45

_cache_dir: Path | None = None


def cache_dir() -> Path:
    """Where the disk cache lives, resolved once and probed for writability.

    This used to be a bare ROOT / "data" / "sleeper". ROOT comes from
    __file__, so on a laptop that is nfl-dfs/data/sleeper -- fine. Deployed to
    Agent Engine the code is staged under /app, which is not promised to be
    writable, and a cache that cannot write is an exception on the first tool
    call rather than a slow one. Same class of break as the sys.path bootstrap
    in each agent's __init__: correct locally, fatal in the cloud.

    Order: SLEEPER_CACHE_DIR if set, then the repo's data dir, then the
    system temp dir, which always works and merely loses the cache between
    cold starts.
    """
    global _cache_dir
    if _cache_dir is not None:
        return _cache_dir

    candidates = [
        os.environ.get("SLEEPER_CACHE_DIR"),
        ROOT / "data" / "sleeper",
        Path(tempfile.gettempdir()) / "sleeper_cache",
    ]
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".writable"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError:
            continue
        _cache_dir = p
        return p

    raise RuntimeError("no writable cache directory found")

# The player dump changes slowly (names, teams, injury designations). A day is
# Sleeper's own suggested ceiling; injury status is NOT read from here for
# anything time-sensitive -- see fetch_rosters' note.
PLAYERS_TTL = 24 * 3600

# Dynasty values move on news, so a much shorter window than the player dump.
VALUES_TTL = 6 * 3600


def _cached(name: str, ttl: float, fetch) -> dict | list:
    """Read `name` from the disk cache, refetching when older than `ttl`.

    Disk rather than memory on purpose -- see the module docstring. A failed
    refetch falls back to the stale copy: a strategy memo built on
    yesterday's values is far more useful than no memo, and unlike injury
    status these numbers do not go dangerously wrong in a day.
    """
    path = cache_dir() / name

    if path.exists() and time.time() - path.stat().st_mtime < ttl:
        return json.loads(path.read_text(encoding="utf-8"))

    try:
        data = fetch()
    except requests.RequestException:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        raise

    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def fetch_players() -> pd.DataFrame:
    """Every NFL player Sleeper knows about, indexed by Sleeper's player_id.

    Includes Sleeper's own injury fields, which are keyed to the same
    player_id the roster endpoint returns -- so nothing in this pipeline ever
    matches a player by name.
    """
    raw = _cached(
        "players.json",
        PLAYERS_TTL,
        lambda: requests.get(f"{SLEEPER}/players/nfl", timeout=TIMEOUT).json(),
    )

    rows = []
    for pid, v in raw.items():
        if not v.get("active") or not v.get("team"):
            continue
        rows.append({
            "player_id": str(pid),
            "name": v.get("full_name") or f"{v.get('first_name','')} {v.get('last_name','')}".strip(),
            "position": v.get("position"),
            "team": v.get("team"),
            "age": v.get("age"),
            "years_exp": v.get("years_exp"),
            "injury_status": v.get("injury_status"),
            "injury_body_part": v.get("injury_body_part"),
            "practice_participation": v.get("practice_participation"),
            "depth_chart_order": v.get("depth_chart_order"),
            "search_rank": v.get("search_rank"),
        })
    return pd.DataFrame(rows)


def fetch_values(num_qbs: int = 2, num_teams: int = 12, ppr: float = 1) -> pd.DataFrame:
    """Dynasty and redraft trade values, keyed by Sleeper's player_id.

    `num_qbs=2` is what FantasyCalc calls a superflex league and it changes QB
    values enormously -- passing that wrong produces a valuation table that
    looks plausible and is wrong for every quarterback on the board.

    The returned `win_now_gap` is the whole point of this table:
    redraft minus dynasty, as a share of dynasty value. Strongly positive
    means the market pays more for this player THIS season than for his
    future -- a sell candidate in a league where the prize needs three
    championships. Strongly negative means the reverse: a buy.
    """
    raw = _cached(
        f"values_qb{num_qbs}_t{num_teams}.json",
        VALUES_TTL,
        lambda: requests.get(
            FANTASYCALC,
            params={"isDynasty": "true", "numQbs": num_qbs,
                    "numTeams": num_teams, "ppr": ppr},
            timeout=TIMEOUT,
        ).json(),
    )

    rows = []
    for r in raw:
        p = r.get("player") or {}
        sid = p.get("sleeperId")
        if not sid:
            continue
        dyn = r.get("value") or 0
        red = r.get("redraftValue") or 0
        rows.append({
            "player_id": str(sid),
            "value_name": p.get("name"),
            "dynasty_value": dyn,
            "redraft_value": red,
            # Guarded: a 0-value player would divide by zero, and there are
            # always a few at the bottom of the board.
            "win_now_gap": round((red - dyn) / dyn, 3) if dyn else None,
            "overall_rank": r.get("overallRank"),
            "position_rank": r.get("positionRank"),
            "trend_30day": r.get("trend30Day"),
        })
    return pd.DataFrame(rows)


def fetch_league(league_id: str) -> dict:
    """League settings: scoring, roster slots, size, status."""
    return requests.get(f"{SLEEPER}/league/{league_id}", timeout=TIMEOUT).json()


def fetch_rosters(league_id: str) -> pd.DataFrame:
    """One row per team: owner, record, points, and the player_ids they hold.

    Injury status is deliberately NOT joined here. The player dump is cached
    for a day, and a day-old injury designation is exactly the failure this
    project keeps finding -- a stale "Questionable" reported long after the
    downgrade. Strategy work tolerates day-old data; a start/sit decision
    does not, and should read Sleeper live rather than reuse this.
    """
    rosters = requests.get(f"{SLEEPER}/league/{league_id}/rosters", timeout=TIMEOUT).json()
    users = requests.get(f"{SLEEPER}/league/{league_id}/users", timeout=TIMEOUT).json()
    names = {u["user_id"]: u.get("display_name") for u in users}

    rows = []
    for r in rosters:
        s = r.get("settings") or {}
        rows.append({
            "roster_id": r.get("roster_id"),
            "owner_id": r.get("owner_id"),
            "owner": names.get(r.get("owner_id"), "unknown"),
            "wins": s.get("wins", 0),
            "losses": s.get("losses", 0),
            "points_for": float(s.get("fpts", 0)) + float(s.get("fpts_decimal", 0)) / 100,
            "players": [str(p) for p in (r.get("players") or [])],
            "starters": [str(p) for p in (r.get("starters") or [])],
        })
    return pd.DataFrame(rows)


def fetch_transactions(league_id: str, through_week: int) -> pd.DataFrame:
    """Every completed add/drop/trade up to `through_week`.

    Trades are the signal that matters: who is consolidating and who is
    breaking up. Sleeper returns one endpoint per week, so this walks them.
    """
    rows = []
    for wk in range(1, max(through_week, 1) + 1):
        try:
            data = requests.get(
                f"{SLEEPER}/league/{league_id}/transactions/{wk}", timeout=TIMEOUT
            ).json()
        except requests.RequestException:
            continue
        for t in data or []:
            if t.get("status") != "complete":
                continue
            rows.append({
                "week": wk,
                "type": t.get("type"),
                "roster_ids": t.get("roster_ids") or [],
                "adds": list((t.get("adds") or {}).keys()),
                "drops": list((t.get("drops") or {}).keys()),
                "draft_picks": t.get("draft_picks") or [],
                "created": t.get("created"),
            })
    return pd.DataFrame(rows)


def fetch_state() -> dict:
    """Current NFL season and week, per Sleeper."""
    return requests.get(f"{SLEEPER}/state/nfl", timeout=TIMEOUT).json()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Inspect a Sleeper league")
    p.add_argument("--league", required=True)
    a = p.parse_args()

    st = fetch_state()
    print(f"NFL state: {st.get('season')} week {st.get('week')}")

    lg = fetch_league(a.league)
    print(f"league: {lg.get('name')} ({lg.get('total_rosters')} teams, {lg.get('status')})")

    players = fetch_players()
    values = fetch_values()
    print(f"players: {len(players)}   values: {len(values)}")

    merged = players.merge(values, on="player_id", how="inner")
    print(f"join coverage: {len(merged)} players carry a dynasty value")

    rosters = fetch_rosters(a.league)
    print(rosters[["owner", "wins", "losses", "points_for"]].to_string(index=False))
