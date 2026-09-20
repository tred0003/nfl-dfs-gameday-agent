"""The five tools the dynasty strategy agent can call.

Same contract as gameday_agent/tools.py: every tool returns a dict carrying a
`status`, never a bare list, so a tool can say it failed instead of returning
an empty result the model reads as "nothing to report".

The function signature and the docstring below it ARE the specification the
model sees. Nothing in a function body reaches the model.

THE CENTRAL NUMBER IS `win_now_gap`. FantasyCalc publishes a dynasty value and
a redraft value for the same player. Their difference, normalised, says whether
the market pays more for this player's present or his future. In a league where
only first place pays and it takes three championships to get there, a
positive gap is a sell signal in a way that raw age is not -- age is a proxy
for that gap, and this is the gap itself.

IDP HAS NO VALUES. FantasyCalc covers offensive skill players only, so every
DL/LB on a roster comes back with dynasty_value None. That is reported
explicitly rather than silently dropped, because four of this league's
fourteen starting slots are IDP and a strategy memo that quietly ignores them
is describing a different team than the one being managed.
"""

from __future__ import annotations

import os

import pandas as pd
import requests

from sleeper_client import (
    fetch_league,
    fetch_players,
    fetch_rosters,
    fetch_state,
    fetch_transactions,
    fetch_values,
)

LEAGUE_ID = os.environ.get("SLEEPER_LEAGUE_ID", "")
MY_USER_ID = os.environ.get("SLEEPER_USER_ID", "")

# Below this, the player/value join has broken rather than merely thinned --
# see _board(). Chosen from an observed 390/421 on a healthy pull.
MIN_JOIN = 300


def _unavailable(e: Exception) -> dict:
    return {
        "status": "error",
        "message": f"Sleeper or FantasyCalc is unavailable right now ({e}). "
                   "Say what could not be checked; do not give strategy advice "
                   "as though the roster were fully known.",
    }


def _no_league() -> dict:
    return {
        "status": "error",
        "message": "SLEEPER_LEAGUE_ID is not configured, so the league is "
                   "unknown. Say so rather than answering generically.",
    }


def _n(v):
    """NaN-safe passthrough -- pandas NaN is not JSON and confuses the model."""
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else v


def _board() -> pd.DataFrame:
    """Every active player with age, injury flag and dynasty/redraft values.

    A LEFT join, deliberately. An inner join would silently delete every IDP
    player from the board, and the absence would be invisible to the model --
    the same bare-container failure this project keeps rediscovering. Left
    keeps them with a null value, so a tool can say "no value published".
    """
    players = fetch_players()
    values = fetch_values()
    board = players.merge(values, on="player_id", how="left")

    valued = int(board["dynasty_value"].notna().sum())
    board.attrs["valued"] = valued
    board.attrs["degraded"] = valued < MIN_JOIN
    return board


# Which positions may fill each starting slot in this league.
SLOT_ELIGIBILITY = {
    "QB": {"QB"},
    "RB": {"RB"},
    "WR": {"WR"},
    "TE": {"TE"},
    "FLEX": {"RB", "WR", "TE"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
}


def _starting_lineup_value(
    pool: list[tuple[float, str]], slots: list[str]
) -> tuple[int, dict]:
    """Best legal starting lineup this pool can field, and what fills each slot.

    WHY THIS EXISTS, and why it replaced ranking on the top five:

    Summing a roster's five best players is position-blind, and in a superflex
    league with a TE premium that is not a detail. The first live landscape
    ranked a manager first on top-five value off four elite RB and WR -- while
    his best quarterback was worth 2,719 in a format that starts TWO of them,
    and his best tight end 1,585 in a format that pays a bonus per TE catch.
    The user spotted it immediately from league knowledge; the metric never
    would have.

    Filling the league's ACTUAL slots prices scarcity automatically. A second
    quarterback you must start every week is worth what your second
    quarterback is worth, not what your fourth running back is worth.

    Greedy, strict slots before flex ones, which is why `slots` is sorted by
    how many positions may fill it. Greedy is not provably optimal for the
    flex tail, but it is what a manager actually does when setting a lineup
    and it never leaves a strict slot unfilled to feed a flex.

    IDP slots are skipped: FantasyCalc publishes no defensive values, so four
    of this league's fourteen starters are invisible to any of this.
    """
    pool = sorted(pool, reverse=True)
    used = [False] * len(pool)
    total, fills = 0, {}

    for slot in sorted(slots, key=lambda s: len(SLOT_ELIGIBILITY[s])):
        for i, (value, position) in enumerate(pool):
            if not used[i] and position in SLOT_ELIGIBILITY[slot]:
                used[i] = True
                total += value
                fills[slot] = int(value)
                break
    return int(total), fills


def _rows(board: pd.DataFrame, ids: list[str], starters: set[str]) -> list[dict]:
    b = board[board["player_id"].isin(ids)]
    out = [
        {
            "name": r["name"],
            "position": r["position"],
            "team": r["team"],
            "age": _n(r["age"]),
            "slot": "starter" if r["player_id"] in starters else "bench",
            "dynasty_value": _n(r["dynasty_value"]),
            "redraft_value": _n(r["redraft_value"]),
            "win_now_gap": _n(r["win_now_gap"]),
            "trend_30day": _n(r["trend_30day"]),
            "injury_status": _n(r["injury_status"]),
        }
        for _, r in b.iterrows()
    ]
    return sorted(out, key=lambda d: d["dynasty_value"] or -1, reverse=True)


def get_league_context() -> dict:
    """Get the current season and week, and how this league's assets work.

    CALL THIS FIRST. Without it there is no way to know what year it is, and a
    trade proposal naming the wrong draft year is worthless -- the first live
    run of this agent recommended trading for "2025 picks" during the 2026
    season, because nothing in the prompt said otherwise.

    `tradeable_pick_years` are the rookie-draft years that can actually change
    hands right now. Never name a year outside that list.

    `picks_traded_so_far` being zero means no pick has moved in this league's
    history. Picks are tradeable, but the market for them is untested -- treat
    a pick-based offer as a proposal that may need explaining to the other
    manager, not as a standard transaction.
    """
    if not LEAGUE_ID:
        return _no_league()
    try:
        state = fetch_state()
        league = fetch_league(LEAGUE_ID)
        traded = requests.get(
            f"https://api.sleeper.app/v1/league/{LEAGUE_ID}/traded_picks", timeout=30
        ).json()
    except requests.RequestException as e:
        return _unavailable(e)

    season = int(state.get("season") or 0)
    s = league.get("settings") or {}

    return {
        "status": "ok",
        "current_season": season,
        "current_week": state.get("week"),
        "league_name": league.get("name"),
        "teams": league.get("total_rosters"),
        "is_dynasty": s.get("type") == 2,
        # The startup was an auction; future rookie drafts are this many rounds.
        "rookie_draft_rounds": s.get("draft_rounds"),
        # A rebuild's main tool: young players parked off the active roster.
        "taxi_slots": s.get("taxi_slots"),
        "injured_reserve_slots": s.get("reserve_slots"),
        "tradeable_pick_years": [season + 1, season + 2, season + 3],
        "picks_traded_so_far": len(traded or []),
        "roster_slots": league.get("roster_positions"),
    }


def get_my_roster() -> dict:
    """Get the user's own roster with age, dynasty value and win-now gap.

    `win_now_gap` is redraft value minus dynasty value, as a fraction of
    dynasty value. Positive means the market values this player's current
    season above his future -- a sell candidate for a manager whose prize
    requires three championships. Negative means the opposite.

    `dynasty_value` is null for defensive players (DL/LB); FantasyCalc does
    not publish IDP values. Null means "not published", NOT "worthless" --
    several of those are among the best players on the roster. Judge IDP by
    age and role, and say plainly that they are unvalued.

    `trend_30day` is the change in dynasty value over thirty days: the timing
    signal for selling high or buying low.
    """
    if not LEAGUE_ID or not MY_USER_ID:
        return _no_league()
    try:
        board = _board()
        rosters = fetch_rosters(LEAGUE_ID)
    except requests.RequestException as e:
        return _unavailable(e)

    mine = rosters[rosters["owner_id"].eq(MY_USER_ID)]
    if mine.empty:
        return {
            "status": "error",
            "message": f"No roster in this league belongs to user {MY_USER_ID}.",
        }
    r = mine.iloc[0]
    rows = _rows(board, r["players"], set(r["starters"]))

    return {
        "status": "ok",
        "owner": r["owner"],
        "record": f"{r['wins']}-{r['losses']}",
        "points_for": round(r["points_for"], 1),
        "roster": rows,
        "unvalued_players": [x["name"] for x in rows if x["dynasty_value"] is None],
        "note": "Values degraded -- fewer players than expected carry one."
                if board.attrs["degraded"] else None,
    }


def get_league_landscape() -> dict:
    """Get every team's record, points, roster age and total dynasty value.

    This is how you tell a contender from a rebuilder, and therefore who
    wants to buy and who wants to sell. A team that is winning with an old
    roster is trying to win now and will pay for win-now players. A team that
    is losing with a young roster is doing what this prize structure rewards
    and will not sell its future cheaply.

    `avg_age` covers offensive skill players only, so it is comparable across
    teams regardless of how many defenders each one rosters.

    DO NOT RANK TEAMS BY `depth_value_sum`. It is a SUM, so it rewards
    rostering more players rather than better ones -- on the first live run
    the user led the league in total while holding 22 valued players against
    another team's 15, and ranked 7th of 12 once star power was compared.
    Reporting him as the league's most valuable roster was an artifact of
    roster size.

    RANK BY `starting_lineup_value`: the best legal lineup the roster can
    actually field, given this league's real slots. It is the only figure here
    that knows the league starts TWO quarterbacks and pays a TE premium.

    `top5_value` is reported for comparison but is position-blind and should
    not be the headline. On the first live run it ranked a manager first off
    four elite RB and WR while his best QB was worth 2,719 in a superflex
    format -- a roster that cannot fill its own lineup was being called the
    league's strongest.

    `slot_values` shows what fills each slot, which is where a thin position
    becomes visible. A low QB or SUPER_FLEX figure in this league is a
    structural weakness no amount of running back depth repairs.

    Use `avg_value_per_player` to judge quality per slot, and
    `depth_value_sum` only to describe overall depth. Never tell the user he
    "leads the league in value" -- that sentence is always about the sum and
    is always misleading.

    Every value here excludes IDP, since FantasyCalc publishes none. Two teams
    with similar numbers can differ a lot in defensive strength and none of
    these figures will show it.
    """
    if not LEAGUE_ID:
        return _no_league()
    try:
        board = _board()
        rosters = fetch_rosters(LEAGUE_ID)
        league = fetch_league(LEAGUE_ID)
    except requests.RequestException as e:
        return _unavailable(e)

    off = board[board["position"].isin(["QB", "RB", "WR", "TE"])]
    age = off.set_index("player_id")["age"]
    val = board.set_index("player_id")["dynasty_value"]
    pos = board.set_index("player_id")["position"]

    # The league's own starting slots, minus the IDP ones nothing can value.
    slots = [s for s in (league.get("roster_positions") or [])
             if s in SLOT_ELIGIBILITY]

    teams = []
    for _, r in rosters.iterrows():
        ages = [age.get(p) for p in r["players"] if pd.notna(age.get(p))]
        vals = [val.get(p) for p in r["players"] if pd.notna(val.get(p))]
        top = sorted(vals, reverse=True)[:5]
        pool = [(val.get(p), pos.get(p)) for p in r["players"]
                if pd.notna(val.get(p))]
        lineup_value, fills = _starting_lineup_value(pool, slots)
        teams.append({
            "owner": r["owner"],
            "record": f"{r['wins']}-{r['losses']}",
            "points_for": round(r["points_for"], 1),
            "avg_age": round(sum(ages) / len(ages), 1) if ages else None,
            # NOT named "total_dynasty_value" any more. Under that name the
            # model kept reporting the user as the league's most valuable
            # roster, which was an artifact of him rostering 22 valued players
            # to another team's 15. The field name is part of the prompt, so
            # the name now says what the number is: a depth sum, not a rank.
            "depth_value_sum": int(sum(vals)),
            "valued_players": len(vals),
            # The ranking number. Championship equity sits in stars, and unlike
            # the sum this cannot be inflated by rostering more bodies.
            "starting_lineup_value": lineup_value,
            "slot_values": fills,
            "top5_value": int(sum(top)),
            "avg_value_per_player": int(sum(vals) / len(vals)) if vals else None,
            "top5_share": round(sum(top) / sum(vals), 2) if vals else None,
            "is_you": r["owner_id"] == MY_USER_ID,
        })

    return {
        "status": "ok",
        # Sorted by star power, not by the sum -- so the ordering the model
        # reads first is already the one that means something.
        "teams": sorted(teams, key=lambda t: -t["starting_lineup_value"]),
    }


def get_team_roster(owner: str) -> dict:
    """Get another manager's roster, to find what they need and what they hold.

    Use this after get_league_landscape identifies a likely trade partner --
    a trade proposal needs to name real players on both sides.
    """
    if not LEAGUE_ID:
        return _no_league()
    try:
        board = _board()
        rosters = fetch_rosters(LEAGUE_ID)
    except requests.RequestException as e:
        return _unavailable(e)

    hit = rosters[rosters["owner"].str.lower().eq(owner.strip().lower())]
    if hit.empty:
        return {
            "status": "error",
            "message": f"No manager named {owner!r}. Known: "
                       f"{', '.join(sorted(rosters['owner']))}",
        }
    r = hit.iloc[0]
    return {
        "status": "ok",
        "owner": r["owner"],
        "record": f"{r['wins']}-{r['losses']}",
        "points_for": round(r["points_for"], 1),
        "roster": _rows(board, r["players"], set(r["starters"])),
    }


def get_market_movers(limit: int = 12) -> dict:
    """Get the biggest 30-day dynasty value movers among rostered players.

    Risers are sell-high windows; fallers are buy-low windows. Restricted to
    players actually rostered in this league, since a free agent nobody holds
    is not a trade opportunity.
    """
    if not LEAGUE_ID:
        return _no_league()
    try:
        board = _board()
        rosters = fetch_rosters(LEAGUE_ID)
    except requests.RequestException as e:
        return _unavailable(e)

    owner_of = {p: r["owner"] for _, r in rosters.iterrows() for p in r["players"]}
    b = board[board["player_id"].isin(owner_of) & board["trend_30day"].notna()].copy()
    b["owner"] = b["player_id"].map(owner_of)

    def pack(d):
        return [
            {
                "name": r["name"],
                "position": r["position"],
                "owner": r["owner"],
                "age": _n(r["age"]),
                "dynasty_value": _n(r["dynasty_value"]),
                "trend_30day": _n(r["trend_30day"]),
                "win_now_gap": _n(r["win_now_gap"]),
            }
            for _, r in d.iterrows()
        ]

    return {
        "status": "ok",
        "risers": pack(b.nlargest(limit, "trend_30day")),
        "fallers": pack(b.nsmallest(limit, "trend_30day")),
    }


def get_recent_transactions() -> dict:
    """Get completed trades and waiver activity so far this season.

    Trades reveal intent that standings do not: who is consolidating for a
    title run and who is breaking up. `draft_picks` on a trade shows whether
    picks are moving at all -- in a brand-new league they often are not, and
    an untested pick market is both an opportunity and a risk.
    """
    if not LEAGUE_ID:
        return _no_league()
    try:
        state = fetch_state()
        board = _board()
        rosters = fetch_rosters(LEAGUE_ID)
        tx = fetch_transactions(LEAGUE_ID, int(state.get("week") or 1))
    except requests.RequestException as e:
        return _unavailable(e)

    name = board.set_index("player_id")["name"]
    owner = dict(zip(rosters["roster_id"], rosters["owner"]))

    trades, adds = [], 0
    for _, t in tx.iterrows():
        if t["type"] != "trade":
            adds += 1
            continue
        trades.append({
            "week": int(t["week"]),
            "teams": [owner.get(i, str(i)) for i in t["roster_ids"]],
            "players_moved": [str(name.get(p, p)) for p in t["adds"]],
            "picks_moved": len(t["draft_picks"]),
        })

    return {
        "status": "ok",
        "current_week": state.get("week"),
        "trades": trades,
        "waiver_and_free_agent_moves": adds,
    }


TOOLS = [
    get_league_context,
    get_my_roster,
    get_league_landscape,
    get_team_roster,
    get_market_movers,
    get_recent_transactions,
]
