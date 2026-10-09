"""Rebuild a real past slate from local parquet, so the agent can be run on it.

The synthetic fixtures are right for regression tests: every row exists
because a check needs it. They are the wrong material for a human to sit and
grade, because nobody has an opinion about whether Grant Iverson was worth
starting. Grading needs reports about real games, where a bad call is
recognisable as a bad call.

Everything here comes from files already on disk -- no network, no DK pull,
nothing redistributed. 2021 is the one season where all three sources
overlap: `dk_salaries.parquet` stops after 2021 and `injuries.parquet` starts
at 2021.

ONE TOOL IS SWITCHED OFF IN REPLAY, and the reason matters more than the
mechanism. `cb_coverage.py` fetches CURRENT cornerback grades from nflverse
by design -- it says so in its own docstring. Pointed at a 2021 slate it
would grade 2021 receivers against the corners playing today, and produce
confident, specific, entirely fictional matchup verdicts. Feeding those to a
human as grading material would teach exactly the wrong lesson, so
`get_cb_matchups` is made to report itself unavailable instead. The agent
then has to say it could not check matchups, which is true.

So a replayed report is thinner than a live one. That is a real limitation of
the grading sample and it is stated here rather than discovered later.

AND A SECOND LIMITATION, WORSE THAN THE FIRST. Replay cannot produce the
agent's number-one case: a rostered player ruled OUT. In 2021 week 8 the
injury report lists 21 skill players as Out and NOT ONE of them appears in
that week's DK salary file -- DraftKings drops ruled-out players from the
board, so the archive only contains men who were available to roster. The
join is not broken; the rows do not exist.

Questionable players survive (15 on that slate), so replay still produces
real injury decisions. But a grading packet built only from replay would
contain no OUT case, no data outage and no injection attempt -- which is why
make_packet.py mixes replayed slates with the synthetic scenarios rather
than replacing them.

This is the same trap as a feature that backtests well and is null in the
live period: the file looks complete, and the rows you need are precisely
the ones missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predict_slate import normalize_name  # noqa: E402

RAW = ROOT / "data" / "raw"

# dk_salaries.parquet carries PFR-style three-letter codes in lower case.
# schedules.parquet and injuries.parquet use nflverse's. Nine disagree.
DK_TEAM_FIXES = {
    "GNB": "GB", "JAC": "JAX", "KAN": "KC", "LAR": "LA", "LVR": "LV",
    "NOR": "NO", "NWE": "NE", "SFO": "SF", "TAM": "TB",
}

# nflverse injury vocabulary -> the codes tools.py and the agent expect.
STATUS_MAP = {"Out": "OUT", "Doubtful": "D", "Questionable": "Q"}

# The user's contest is the Sunday 1pm/4pm windows only. Kickoff times in
# schedules.parquet are US Eastern, so this is a plain string compare.
MAIN_SLATE_START, MAIN_SLATE_END = "13:00", "16:30"

SEASON_WITH_FULL_COVERAGE = 2021


def _flip_name(s: pd.Series) -> pd.Series:
    """DK stores "Brady, Tom". Everything downstream wants "Tom Brady"."""
    parts = s.fillna("").str.split(",", n=1)
    return parts.apply(
        lambda p: f"{p[1].strip()} {p[0].strip()}" if len(p) == 2 else p[0].strip()
    )


def _fix_team(s: pd.Series) -> pd.Series:
    return s.fillna("").str.upper().replace(DK_TEAM_FIXES)


def available_weeks(season: int = SEASON_WITH_FULL_COVERAGE) -> list[int]:
    """Weeks where salaries, injuries and a betting line all exist."""
    dk = pd.read_parquet(RAW / "dk_salaries.parquet")
    inj = pd.read_parquet(RAW / "injuries.parquet")
    dk_w = set(dk[dk["season"] == season]["week"].unique())
    inj_w = set(inj[inj["season"] == season]["week"].dropna().astype(int).unique())
    return sorted(dk_w & inj_w)


def _games(season: int, week: int) -> pd.DataFrame:
    sched = pd.read_parquet(RAW / "schedules.parquet")
    g = sched[(sched["season"] == season) & (sched["week"] == week)].copy()
    g = g[g["weekday"] == "Sunday"]
    g = g[g["gametime"].between(MAIN_SLATE_START, MAIN_SLATE_END)]
    g["game"] = g["away_team"] + " @ " + g["home_team"]
    return g


def _dk_fppg(dk: pd.DataFrame, season: int, week: int) -> pd.Series:
    """Season-to-date average, from games BEFORE this one.

    Weeks at or after the slate are excluded deliberately. Including them
    would leak the result the agent is supposed to be reasoning ahead of,
    and week 1 legitimately has no average at all -- which is the null that
    the clean_lineup scenario exists to stop the agent panicking about.
    """
    prior = dk[(dk["season"] == season) & (dk["week"] < week)]
    return prior.groupby("name")["dk_points"].mean()


def _oprk(dk: pd.DataFrame, season: int, week: int) -> dict:
    """Opponent rank against each position, 1 = toughest, from prior weeks."""
    prior = dk[(dk["season"] == season) & (dk["week"] < week)].copy()
    if prior.empty:
        return {}
    prior["opp"] = _fix_team(prior["opponent"])
    allowed = prior.groupby(["opp", "position"])["dk_points"].mean().reset_index()
    allowed["rank"] = allowed.groupby("position")["dk_points"].rank(method="min")
    return {(r["opp"], r["position"]): int(r["rank"]) for _, r in allowed.iterrows()}


def replay_slate(season: int, week: int) -> dict:
    """Every frame the tools need, in the shape the live clients return."""
    dk_all = pd.read_parquet(RAW / "dk_salaries.parquet")
    # Flip ONCE, here, before anything groups or joins on the name. The first
    # version flipped only the slate rows and left the history in DK's
    # "Brady, Tom" form, so the season-to-date average joined on nothing and
    # every player came back with a null points-per-game. That is the exact
    # null the agent once called "not an active NFL player", so the bug was
    # actively degrading the reports being generated for grading.
    dk_all = dk_all.copy()
    dk_all["name"] = _flip_name(dk_all["name"])

    dk = dk_all[(dk_all["season"] == season) & (dk_all["week"] == week)].copy()
    if dk.empty:
        raise SystemExit(f"no DK salary rows for {season} week {week}")

    games = _games(season, week)
    if games.empty:
        raise SystemExit(f"no Sunday main-slate games for {season} week {week}")

    dk["team"] = _fix_team(dk["team"])
    dk["opp"] = _fix_team(dk["opponent"])
    # DK's historical file labels defences DEF; the agent's world says DST.
    dk["position"] = dk["position"].replace({"DEF": "DST"})

    on_slate = set(games["home_team"]) | set(games["away_team"])
    dk = dk[dk["team"].isin(on_slate) & dk["opp"].isin(on_slate)]

    pair = {}
    for _, g in games.iterrows():
        pair[g["home_team"]] = g["game"]
        pair[g["away_team"]] = g["game"]
    dk["game"] = dk["team"].map(pair)

    fppg = _dk_fppg(dk_all, season, week)
    oprk = _oprk(dk_all, season, week)
    dk["dk_fppg"] = dk["name"].map(fppg)
    dk["dk_oprk"] = [oprk.get((o, p)) for o, p in zip(dk["opp"], dk["position"])]

    players = dk[["name", "position", "team", "game", "salary",
                  "dk_fppg", "dk_oprk"]].copy()
    players["salary"] = players["salary"].fillna(0).astype(int)
    players["jn"] = normalize_name(players["name"])

    slate_games = games[["game", "home_team", "away_team"]].reset_index(drop=True)

    # ---- injuries, in the shape espn_injuries() returns
    inj_all = pd.read_parquet(RAW / "injuries.parquet")
    inj = inj_all[(inj_all["season"] == season) & (inj_all["week"] == week)].copy()
    inj["team"] = _fix_team(inj["team"])
    inj = inj[inj["team"].isin(on_slate)]
    inj = inj[inj["report_status"].isin(STATUS_MAP)]
    injuries = pd.DataFrame({
        "team": inj["team"],
        "name": inj["full_name"],
        "position": inj["position"],
        "status": inj["report_status"].map(STATUS_MAP),
        "body_part": inj["report_primary_injury"],
        "return_date": None,
        # The practice line is the closest thing this source has to ESPN's
        # free-text note, and it is the field the fenced-text handling in
        # tools.py applies to.
        "comment": inj["practice_status"],
        "report_date": None,
    })
    injuries["jn"] = normalize_name(injuries["name"])

    # ---- scoreboard, in the shape espn_scoreboard() returns
    scoreboard = pd.DataFrame({
        "game": games["away_team"] + " at " + games["home_team"],
        "home_team": games["home_team"],
        "away_team": games["away_team"],
        "start_time_utc": games["gameday"].astype(str) + "T" + games["gametime"].astype(str),
        "venue": games["stadium"],
        "is_dome": games["roof"].isin(["dome", "closed"]),
        "weather_posted": games["temp"].notna(),
        "weather_temp_f": games["temp"],
        # No condition string exists in this source, and WIND IS NOT PASSED
        # THROUGH even though the column exists: the agent's instruction says
        # no tool reports wind, and a replay that quietly started reporting it
        # would be testing a different agent than the one that is deployed.
        "weather_condition": None,
    }).reset_index(drop=True)

    # ---- vegas, in the shape vegas_for_slate() returns.
    # nflverse signs spread_line from the HOME team's perspective: positive
    # means the home team is favoured by that many points.
    vegas = pd.concat([
        pd.DataFrame({
            "team": games["home_team"],
            "implied_team_total": games["total_line"] / 2 + games["spread_line"] / 2,
        }),
        pd.DataFrame({
            "team": games["away_team"],
            "implied_team_total": games["total_line"] / 2 - games["spread_line"] / 2,
        }),
    ]).dropna(subset=["implied_team_total"]).reset_index(drop=True)

    return {"players": players, "games": slate_games, "injuries": injuries,
            "scoreboard": scoreboard, "vegas": vegas}


# --------------------------------------------------------------- lineup picks

SLOTS = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "DST"]
CAP = 50_000


def build_lineup(data: dict, seed: int = 0, want_injured: bool = False) -> list[str]:
    """An eight-man lineup a real person might plausibly have entered.

    Not optimised -- the optimiser is a separate part of this repo and the
    point here is a believable roster, not a good one. `seed` varies which
    tier gets taken so a packet of eight reports is not eight versions of the
    same lineup.

    With want_injured, one slot is deliberately filled by a player carrying a
    real designation from that week. A grading sample of nothing but clean
    slates would tell you nothing about how the agent handles the case that
    actually matters.
    """
    players = data["players"].copy()
    hurt = set(data["injuries"][data["injuries"]["status"].isin(["OUT", "D", "Q"])]["jn"])

    players = players[players["salary"] > 0].copy()
    # A player with no average yet is scored off his salary instead. Dropping
    # him would quietly exclude every rookie and every week-1 slate, and the
    # null average is itself something the agent has to handle.
    players["score"] = players["dk_fppg"].fillna(players["salary"] / 500)

    slots = list(SLOTS)
    picked, used, spent = [], set(), 0

    def take(row) -> None:
        nonlocal spent
        picked.append(row["name"])
        used.add(row["jn"])
        spent += int(row["salary"])

    if want_injured:
        pool = players[players["jn"].isin(hurt)
                       & players["position"].isin(["RB", "WR", "TE"])]
        pool = pool.sort_values("salary", ascending=False)
        if not pool.empty:
            r = pool.iloc[seed % min(len(pool), 6)]
            # Consume the slot he fills rather than adding a ninth man. The
            # first version prepended him and produced rosters with three
            # running backs and two receivers, which reads as a bug to
            # anyone who plays this game.
            if r["position"] in slots:
                slots.remove(r["position"])
            take(r)

    for i, slot in enumerate(slots):
        pool = players[(players["position"] == slot) & (~players["jn"].isin(used))]
        # Budget left after reserving DK's $2,500 minimum for each slot still
        # to come AFTER this one. Counting from `slots` rather than from
        # len(picked) is what makes this correct when a slot was consumed
        # above -- the earlier version was off by one and silently left a
        # receiver unfilled.
        still_to_come = len(slots) - i - 1
        affordable = pool[pool["salary"] <= CAP - spent - still_to_come * 2500]
        if affordable.empty:
            # Nothing fits once the remaining slots are reserved for. Relax
            # the reserve but NOT the cap -- falling straight back to the
            # cheapest player on the board put a lineup $200 over $50,000,
            # and a roster that could not legally be entered is the first
            # thing a DFS player notices and the last thing that should be
            # in material they are grading.
            affordable = pool[pool["salary"] <= CAP - spent]
        pool = affordable if not affordable.empty else pool.nsmallest(1, "salary")
        if pool.empty:
            continue
        pool = pool.sort_values("score", ascending=False)
        take(pool.iloc[seed % min(len(pool), 8)])

    return picked
