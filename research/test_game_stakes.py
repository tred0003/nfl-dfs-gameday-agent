"""Do stakes change who gets the ball? Rivalries, division games, primetime, tanking.

Hypotheses (his, from watching games):
  1. Division games and rivalries (McVay vs Shanahan) -- coaches feed their
     best players harder because they want that game more.
  2. Primetime -- same idea, bigger stage.
  3. A team out of the playoff race late in the season protects its young RB
     instead of putting miles on him.
  0. SANITY CHECK: top teams rest starters in the final week. That effect is
     known to be real -- if this method can't see it, trust nothing else here.

Measure: a player's share of his team's opportunities (targets + carries) in a
game, minus HIS OWN recent share coming in. Comparing each player to himself
removes "good players get the ball" from the answer.

Things that could fake an effect, and how they're handled:
  - game script: losing teams throw more and run less -> control for the
    Vegas spread (known before kickoff, so it's the expected script)
  - regression to the mean: a team's top player usually dips below his own
    average anyway -> every comparison is star-vs-star, never star-vs-everyone
  - late-season wear: compare late-season games to OTHER late-season games
Standard errors are clustered by game (teammates in one game aren't independent).

Run: python research/test_game_stakes.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
KEYS = ["player_id", "season", "week"]


def team_game_context() -> pd.DataFrame:
    """Per team-game: division, primetime, rivalry, and record coming in."""
    s = pd.read_parquet(RAW / "schedules.parquet")
    s = s[s["game_type"].eq("REG") & s["result"].notna()].copy()
    s["primetime"] = (~s["weekday"].eq("Sunday") | s["gametime"].ge("19:00")).astype(int)
    coaches = {"Sean McVay", "Kyle Shanahan"}
    s["mcvay_shanahan"] = (s["home_coach"].isin(coaches) & s["away_coach"].isin(coaches)
                           & s["home_coach"].ne(s["away_coach"])).astype(int)
    s["last_week"] = s.groupby("season")["week"].transform("max")
    sides = []
    for side, sign in (("home", 1), ("away", -1)):
        d = s[["game_id", "season", "week", "div_game", "primetime", "mcvay_shanahan", "last_week"]].copy()
        d["team"] = s[f"{side}_team"]
        d["win"] = np.sign(s["result"] * sign).map({1: 1.0, 0: 0.5, -1: 0.0})
        sides.append(d)
    t = pd.concat(sides).sort_values(["team", "season", "week"])
    g = t.groupby(["team", "season"])
    t["games_before"] = g.cumcount()
    t["win_pct_before"] = g["win"].transform(lambda x: x.shift(1).expanding().mean())
    return t


def player_games() -> pd.DataFrame:
    w = pd.read_parquet(RAW / "weekly.parquet")
    w = w[w["season_type"].eq("REG")].copy()
    w["opps"] = w["targets"].fillna(0) + w["carries"].fillna(0)
    tot = w.groupby(["recent_team", "season", "week"])[["opps", "carries"]].transform("sum")
    w["share"] = w["opps"] / tot["opps"].replace(0, np.nan)
    w["carry_share"] = w["carries"].fillna(0) / tot["carries"].replace(0, np.nan)

    m = pd.read_parquet(PROC / "model_table_v3.parquet",
                        columns=[*KEYS, "recent_team", "position", "dk_points", "dkp_ewm", "age", "team_spread"])
    m[["season", "week"]] = m[["season", "week"]].astype("int64")
    w[["season", "week"]] = w[["season", "week"]].astype("int64")
    d = m.merge(w[[*KEYS, "share", "carry_share"]], on=KEYS).sort_values(["player_id", "season", "week"])

    g = d.groupby(["player_id", "season"], sort=False)
    d["prior_share"] = g["share"].transform(lambda x: x.shift(1).ewm(halflife=3).mean())
    d["prior_carry_share"] = g["carry_share"].transform(lambda x: x.shift(1).ewm(halflife=3).mean())
    d["n_prior"] = g.cumcount()
    d = d[d["n_prior"] >= 3].copy()  # need a real baseline
    d["team_rank"] = d.groupby(["recent_team", "season", "week"])["prior_share"].rank(ascending=False, method="first")
    d["delta_share"] = (d["share"] - d["prior_share"]) * 100          # percentage points
    d["delta_carry_share"] = (d["carry_share"] - d["prior_carry_share"]) * 100
    d["delta_pts"] = d["dk_points"] - d["dkp_ewm"]

    ctx = team_game_context().rename(columns={"team": "recent_team"})
    d = d.merge(ctx, on=["recent_team", "season", "week"])
    d["late"] = d["week"] >= 12
    return d


def fit(formula: str, data: pd.DataFrame, term: str) -> str:
    r = smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["game_id"]})
    b, se = r.params[term], r.bse[term]
    flag = "  <- clear" if abs(b) > 2 * se else ""
    return f"{b:+.2f}  (95% range {b - 1.96 * se:+.2f} to {b + 1.96 * se:+.2f}){flag}"


def main() -> None:
    d = player_games()
    stars = d[d["team_rank"] <= 2]  # each team's top two opportunity players coming in
    print(f"{len(d):,} player-games 2016-2025 with 3+ prior games; {len(stars):,} are a team's top-2 players")
    print("effects in PERCENTAGE POINTS of team opportunities, each star vs his own recent share\n")

    # ---- 0. sanity check
    rest = stars[stars["week"].eq(stars["last_week"]) & stars["games_before"].ge(10)].copy()
    rest["top_team"] = (rest["win_pct_before"] >= 0.70).astype(int)
    print("0. SANITY CHECK -- final week, teams .700+ vs everyone else (should be clearly negative)")
    print(f"   opportunity share: {fit('delta_share ~ top_team + team_spread', rest, 'top_team')}")
    print(f"   DK points:         {fit('delta_pts ~ top_team + team_spread', rest, 'top_team')}")
    print(f"   ({int(rest['top_team'].sum())} star-games on top teams)\n")

    # drop final-week rest games from everything else so they don't pollute it
    base = stars[~(stars["week"].eq(stars["last_week"]) & stars["win_pct_before"].ge(0.70))]

    print("1. DIVISION GAMES (vs non-division, same weeks, spread controlled)")
    print(f"   opportunity share: {fit('delta_share ~ div_game + team_spread + C(week)', base, 'div_game')}")
    print(f"   DK points:         {fit('delta_pts ~ div_game + team_spread + C(week)', base, 'div_game')}")

    print("\n2. PRIMETIME (Thu/Sat/Mon or Sunday night)")
    print(f"   opportunity share: {fit('delta_share ~ primetime + team_spread + C(week)', base, 'primetime')}")
    print(f"   DK points:         {fit('delta_pts ~ primetime + team_spread + C(week)', base, 'primetime')}")

    ls = base[base["recent_team"].isin(["LA", "SF"]) & base["season"].ge(2017)]
    n_games = ls.loc[ls["mcvay_shanahan"].eq(1), "game_id"].nunique()
    print(f"\n3. McVAY vs SHANAHAN -- Rams/49ers stars in those {n_games} games vs their other games")
    print(f"   opportunity share: {fit('delta_share ~ mcvay_shanahan + team_spread', ls, 'mcvay_shanahan')}")
    print(f"   DK points:         {fit('delta_pts ~ mcvay_shanahan + team_spread', ls, 'mcvay_shanahan')}")
    div_only = ls[ls["div_game"].eq(1)]
    print(f"   vs their OTHER division games only: "
          f"{fit('delta_share ~ mcvay_shanahan + team_spread', div_only, 'mcvay_shanahan')}")

    # ---- 4. tanking and young RBs
    rb = d[d["position"].astype(str).eq("RB") & d["late"] & d["prior_carry_share"].ge(0.40)
           & d["games_before"].ge(10) & d["week"].lt(d["last_week"])].copy()
    rb["stakes"] = pd.cut(rb["win_pct_before"], [-0.01, 0.35, 0.55, 1.01],
                          labels=["out of it (<=.350)", "middle", "contender (.550+)"])
    rb["age_group"] = pd.cut(rb["age"], [0, 24.99, 26.99, 60], labels=["young (<25)", "25-26", "veteran (27+)"])
    print("\n4. LATE SEASON (weeks 12+, before final week): lead RBs (40%+ of carries coming in)")
    print("   change in CARRY share vs his own recent share, by team situation and age:")
    tab = rb.groupby(["stakes", "age_group"]).agg(games=("delta_carry_share", "size"),
                                                   carry_share_change=("delta_carry_share", "mean"),
                                                   dk_pts_change=("delta_pts", "mean")).round(2)
    print(tab.to_string())
    rb["out_of_it"] = rb["win_pct_before"].le(0.35).astype(int)
    rb["young"] = rb["age"].lt(25).astype(int)
    print("\n   the actual question -- do out-of-it teams cut YOUNG backs more than other teams do?")
    print(f"   (young x out of it, spread controlled): "
          f"{fit('delta_carry_share ~ out_of_it * young + team_spread', rb, 'out_of_it:young')}")
    print(f"   out-of-it teams, all lead RBs:          "
          f"{fit('delta_carry_share ~ out_of_it + young + team_spread', rb, 'out_of_it')}")

    late_season_playing_time(d)


def late_season_playing_time(d: pd.DataFrame) -> None:
    """The blind spot above: a player who sits entirely has NO row, so "shut him
    down" never shows up as a lower share. Count games actually played instead."""
    w = pd.read_parquet(RAW / "weekly.parquet")
    w = w[w["season_type"].eq("REG")].copy()
    w[["season", "week"]] = w[["season", "week"]].astype("int64")
    tot = w.groupby(["recent_team", "season", "week"])["carries"].transform("sum")
    w["cs"] = w["carries"].fillna(0) / tot.replace(0, np.nan)
    early = w[w["position"].eq("RB") & w["week"].le(11)]
    lead = (early.groupby(["player_id", "season", "recent_team"])
                 .agg(cs=("cs", "mean"), g=("cs", "size")).reset_index())
    lead = lead[lead["cs"].ge(0.40) & lead["g"].ge(6)]  # lead back through week 11

    ctx = team_game_context()
    played = set(zip(w["player_id"], w["season"], w["week"], w["recent_team"]))
    ages = d[["player_id", "season", "age"]].drop_duplicates(["player_id", "season"])
    rows = []
    for r in lead.itertuples():
        tg = ctx[ctx["team"].eq(r.recent_team) & ctx["season"].eq(r.season)]
        if tg.empty:
            continue
        window = tg[tg["week"].ge(12) & tg["week"].lt(tg["last_week"])].sort_values("week")
        if window.empty:
            continue
        n = sum((r.player_id, r.season, wk, r.recent_team) in played for wk in window["week"])
        rows.append({"player_id": r.player_id, "season": r.season,
                     "win_pct": window["win_pct_before"].iloc[0], "pct_played": n / len(window) * 100})
    x = pd.DataFrame(rows).merge(ages, on=["player_id", "season"]).dropna(subset=["age"])
    x["stakes"] = pd.cut(x["win_pct"], [-0.01, 0.35, 0.55, 1.01], labels=["out of it", "middle", "contender"])
    x["age_group"] = pd.cut(x["age"], [0, 24.99, 26.99, 60], labels=["young (<25)", "25-26", "veteran (27+)"])
    print("\n5. LATE-SEASON PLAYING TIME -- lead RBs through week 11: % of weeks 12+ games played")
    print(x.groupby(["stakes", "age_group"]).agg(rb_seasons=("pct_played", "size"),
                                                  pct_games_played=("pct_played", "mean")).round(1).to_string())
    x["out_of_it"] = x["win_pct"].le(0.35).astype(int)
    x["young"] = x["age"].lt(25).astype(int)
    r = smf.ols("pct_played ~ out_of_it * young", data=x).fit(cov_type="HC1")
    for k in ("out_of_it", "out_of_it:young"):
        b, se = r.params[k], r.bse[k]
        print(f"   {k:<16} {b:+.1f} pts of games (95% range {b - 1.96 * se:+.1f} to {b + 1.96 * se:+.1f})")


if __name__ == "__main__":
    main()
