"""Lineup backtest: what would each projection source actually have scored?

The scoreboard that matters in the end. For every week 2021-2024:
  1. Build the DK main-slate player pool (Sunday games before the night game).
  2. Hand the optimizer each source's projections and let it build a lineup.
  3. Score that lineup on ACTUAL DK points.

Sources:  FantasyData projection | XGBoost v3 | XGBoost base | salary line
Rules:    "current" -- src/optimize.py as it stands (forced QB stack, $49k floor)
          "loose"   -- no forced stack, no salary floor
          Both rules were once doubted on the strength of "hindsight-optimal
          lineups stacked only 53%". That reasoning is a base-rate error: a
          position-legal lineup built with no stacking rule still stacks ~20%
          of the time by chance, so 53% (62% on 2021-24) is roughly 3x chance
          and is evidence FOR the rule. The floor, separately, never binds.
Plus the hindsight-perfect lineup each week, for scale.

FAIRNESS -- READ THIS BEFORE QUOTING ANY NUMBER BELOW. The pool contains only
players who recorded a stat that week. This was described here as a leak that
affects every source "equally". Measured on 2026-09-16, that is wrong, and the
bias runs in the model's favour:

  - 39% of DK-priced main-slate skill players never record a stat and are
    silently removed.
  - About 13.6 players per week averaged 12+ DK points over their previous five
    games and did not play. FantasyData projects those at a MEDIAN OF 0.00 --
    its process knows who is inactive.
  - The model keys on prior form (dkp_blended, dkp_ewm, dkp_r5 are its top
    features) and has no equivalent mechanism, so it would project them high and
    roster them.

So the filter shields this model from exactly the failure mode the commercial
feed is best at avoiding, and the real gap to FantasyData is wider than the
numbers here show. The root cause is upstream: features.py builds a training row
only for players who recorded a stat, so the model has never seen a single
non-playing player and cannot have learned to recognise one.

Compare sources to each other with that caveat in mind, and never to real
contest scores.

Defenses have no model: every source uses FantasyData's DST projection, and DST
actual points are scored from play-by-play (approximate DK scoring).

Run: python src/backtest.py            (2021-2024, ~10-15 min -> research/backtest_lineups.csv)
     python src/backtest.py --final    (2025 only -> research/backtest_lineups_2025.csv)
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from optimize import optimize  # noqa: E402
from train import load_table, walk_forward  # noqa: E402

RAW = ROOT / "data" / "raw"
SEASONS = [2021, 2022, 2023, 2024]
KEYS = ["player_id", "season", "week"]
SOURCES = {"FantasyData": "fd_projection", "v3": "v3", "v3 no-salary": "v3_nosal",
           "base": "base", "salary line": "salary_implied_pts"}
RULES = {
    "current": dict(stack_qb=True, min_salary=49_000),
    "loose": dict(stack_qb=False, min_salary=0),
}
TEAM_TO_NFLVERSE = {"LAR": "LA"}


# ------------------------------------------------------------------ inputs
def fantasydata_week_files() -> pd.DataFrame:
    """Every FantasyData row 2021-2024: salary + projection, all positions."""
    frames = []
    for f in glob.glob(str(RAW / "fantasydata" / "projections" / "dk_2021_*.csv")):
        season, week = map(int, re.findall(r"dk_(\d+)_wk(\d+)", f)[0])
        d = pd.read_csv(f).rename(columns={"pos_slate": "position", "fpts": "fd_projection"})
        frames.append(d.assign(season=season, week=week))
    for f in glob.glob(str(RAW / "fantasydata" / "salaries" / "*.csv")):
        season, week = map(int, re.findall(r"dk_(\d+)_wk(\d+)", f)[0])
        if season not in SEASONS:
            continue
        d = pd.read_csv(f).rename(columns={"pos": "position", "salary_draftkings": "salary",
                                           "fpts_draftkings": "fd_projection"})
        frames.append(d.assign(season=season, week=week))
    fd = pd.concat(frames, ignore_index=True)
    fd = fd[fd["position"].isin(["QB", "RB", "WR", "TE", "DST"])]
    for c in ("team", "opp"):
        fd[c] = fd[c].replace(TEAM_TO_NFLVERSE)
    return fd[["id", "player", "position", "team", "opp", "salary", "fd_projection", "season", "week"]]


def main_slate_games() -> pd.DataFrame:
    """Team -> game for Sunday games that kick off before the night game."""
    s = pd.read_parquet(RAW / "schedules.parquet")
    s = s[s["season"].isin(SEASONS) & s["game_type"].eq("REG")
          & s["weekday"].eq("Sunday") & s["gametime"].lt("19:00")]
    sides = [s.assign(team=s["home_team"]), s.assign(team=s["away_team"])]
    return pd.concat(sides)[["season", "week", "team", "game_id"]]


def dst_actual_points() -> pd.DataFrame:
    """Approximate DK DST scoring from play-by-play."""
    rows = []
    for season in SEASONS:
        p = pd.read_parquet(RAW / "pbp" / f"play_by_play_{season}.parquet")
        p = p[p["season_type"].eq("REG")]
        final = p.groupby("game_id").agg(week=("week", "first"), home=("home_team", "first"),
                                         away=("away_team", "first"),
                                         hs=("total_home_score", "max"), as_=("total_away_score", "max"))
        for gid, g in final.iterrows():
            plays = p[p["game_id"].eq(gid)]
            for team, allowed in ((g["home"], g["as_"]), (g["away"], g["hs"])):
                d = plays[plays["defteam"].eq(team)]
                blocked = (d["field_goal_result"].eq("blocked") | d["punt_blocked"].eq(1)
                           | d["extra_point_result"].eq("blocked")).sum()
                return_tds = (plays["td_team"].eq(team) & plays["posteam"].ne(team)).sum()
                pa = (10 if allowed == 0 else 7 if allowed <= 6 else 4 if allowed <= 13 else
                      1 if allowed <= 20 else 0 if allowed <= 27 else -1 if allowed <= 34 else -4)
                pts = (d["sack"].fillna(0).sum() + 2 * d["interception"].fillna(0).sum()
                       + 2 * d["fumble_lost"].fillna(0).sum() + 2 * d["safety"].fillna(0).sum()
                       + 2 * blocked + 6 * return_tds + pa)
                rows.append({"season": season, "week": int(g["week"]), "team": team, "dst_points": pts})
    return pd.DataFrame(rows)


# Every feature that carries DraftKings' own pricing, directly or not.
#
# The obvious four are not enough. `value_*` is literally dk_points minus
# salary_implied_pts, so it is salary wearing a hat. And `dkp_blended` -- the
# single most important feature in the model at 16.8% of total SHAP -- falls
# back to salary_implied_pts whenever prior-season data is thin
# (features_carry.py:176), so it smuggles pricing in for exactly the players
# where pricing matters most: rookies and role-changers.
#
# Dropping dkp_blended costs the top feature, but its non-salary half survives
# as `ps_dkp_pg`, which stays. Leave any of these in and the experiment is fake.
SALARY_DERIVED = {
    "salary", "salary_implied_pts", "salary_chg_1w", "salary_chg_3w",
    "salary_pos_pct", "value_last", "value_r3", "value_streak", "dkp_blended",
}


def model_predictions() -> pd.DataFrame:
    """Walk-forward predictions from each model variant, keyed by player-week.

    `v3_nosal` is the same model and the same rows as `v3` with pricing removed
    from the inputs. The point is not accuracy -- it will score worse, because
    salary encodes real information (beat reports, late news) that none of the
    other features have. The point is INDEPENDENCE: a projection built without
    the price can disagree with the price, and disagreement is the only thing
    that can be mispriced. Whether that independence is worth the lost accuracy
    is not arguable from Spearman; it is settled by lineup points below.
    """
    out = None
    specs = (("v3", "model_table_v3", False),
             ("base", "model_table", False),
             ("v3_nosal", "model_table_v3", True))

    for name, table, drop_salary in specs:
        df, feats = load_table(table)
        if drop_salary:
            feats = [f for f in feats if f not in SALARY_DERIVED]
        with contextlib.redirect_stdout(io.StringIO()):
            p, _ = walk_forward(df, feats, with_ceiling=False, seasons=SEASONS)
        keep = [*KEYS, "pred_mean"] + (["dk_points", "salary_implied_pts"] if name == "v3" else [])
        p = p[keep].rename(columns={"pred_mean": name})
        out = p if out is None else out.merge(p, on=KEYS)
    return out


def build_pools() -> pd.DataFrame:
    fd = fantasydata_week_files()
    ids = pd.read_parquet(RAW / "ids.parquet")[["fantasy_data_id", "gsis_id"]].dropna()
    ids["id"] = ids["fantasy_data_id"].astype(int).astype(str)
    fd["id"] = fd["id"].astype(str)

    games = main_slate_games()
    fd = fd.merge(games, on=["season", "week", "team"])  # main slate only

    skill = fd[fd["position"].ne("DST")].merge(ids[["id", "gsis_id"]], on="id")
    skill = skill.rename(columns={"gsis_id": "player_id"}).merge(model_predictions(), on=KEYS)

    dst = fd[fd["position"].eq("DST")].merge(dst_actual_points(), on=["season", "week", "team"])
    dst = dst.rename(columns={"dst_points": "dk_points"})
    for col in ("v3", "v3_nosal", "base", "salary_implied_pts"):
        dst[col] = dst["fd_projection"]  # no DST model: everyone uses FantasyData's

    pool = pd.concat([skill, dst], ignore_index=True)
    pool = pool.rename(columns={"player": "name", "opp": "opp_team", "game_id": "game"})
    pool = pool.dropna(subset=["salary", "dk_points", "fd_projection", "v3", "v3_nosal",
                               "base", "salary_implied_pts"])
    return pool.drop_duplicates(["season", "week", "name", "team", "position"])


# ------------------------------------------------------------------ run
def lineup_row(lu: pd.DataFrame, proj_col: str) -> dict:
    qb = lu[lu["position"].eq("QB")]
    stacked = bool(len(qb)) and lu[lu["position"].isin(["WR", "TE"])]["team"].eq(qb["team"].iloc[0]).any()
    return {"actual": lu["dk_points"].sum(), "projected": lu[proj_col].sum(),
            "salary": lu["salary"].sum(), "stacked": int(stacked),
            "players": "; ".join(lu["name"])}


def main() -> None:
    print("building pools (training models, scoring defenses)...", flush=True)
    pools = build_pools()
    weeks = pools.groupby(["season", "week"])
    print(f"{weeks.ngroups} weeks, avg pool {len(pools) / weeks.ngroups:.0f} players\n", flush=True)

    rows = []
    for i, ((season, week), pool) in enumerate(weeks, 1):
        if pool["position"].eq("DST").sum() == 0 or pool["position"].eq("QB").sum() == 0:
            continue
        best = optimize(pool, "dk_points", max_per_team=8, stack_qb=False, min_salary=0,
                        block_dst_vs_offense=False)
        if best:
            rows.append({"season": season, "week": week, "source": "PERFECT (hindsight)",
                         "rules": "-", **lineup_row(best[0], "dk_points")})
        for src, col in SOURCES.items():
            for rule, kw in RULES.items():
                lu = optimize(pool, col, **kw)
                if lu:
                    rows.append({"season": season, "week": week, "source": src, "rules": rule,
                                 **lineup_row(lu[0], col)})
        if i % 12 == 0:
            print(f"  {i}/{weeks.ngroups} weeks", flush=True)

    r = pd.DataFrame(rows)
    name = "backtest_lineups_2025.csv" if SEASONS == [2025] else "backtest_lineups.csv"
    r.to_csv(ROOT / "research" / name, index=False)
    report(r)


def report(r: pd.DataFrame) -> None:
    # grabbed before the PERFECT rows are filtered out below
    perfect_stack = r[r["source"].str.startswith("PERFECT")]["stacked"].mean() * 100
    perfect = r[r["source"].str.startswith("PERFECT")][["season", "week", "actual"]]
    r = r[~r["source"].str.startswith("PERFECT")].merge(
        perfect.rename(columns={"actual": "perfect"}), on=["season", "week"])
    r["pct_of_perfect"] = r["actual"] / r["perfect"] * 100
    perfect = perfect["actual"]
    n = r[["season", "week"]].drop_duplicates().shape[0]

    print("\n" + "=" * 72)
    print(f"ACTUAL LINEUP POINTS -- {n} main-slate weeks, {SEASONS[0]}-{SEASONS[-1]}")
    print("=" * 72)
    t = r.pivot_table(index="source", columns="rules", values="actual", aggfunc="mean")
    t["avg_projected (current)"] = r[r["rules"].eq("current")].groupby("source")["projected"].mean()
    t["% of perfect (current)"] = r[r["rules"].eq("current")].groupby("source")["pct_of_perfect"].mean()
    print(t.sort_values("current", ascending=False).round(1).to_string())
    print(f"\nperfect lineup averaged {perfect.mean():.1f}")

    print("\nby season (current rules):")
    print(r[r["rules"].eq("current")].pivot_table(index="source", columns="season", values="actual",
                                                  aggfunc="mean").round(1).to_string())

    cur = r[r["rules"].eq("current")].pivot_table(index=["season", "week"], columns="source", values="actual")
    diff = cur["v3"] - cur["FantasyData"]
    print(f"\nv3 vs FantasyData, week by week (current rules): v3 higher in {(diff > 0).mean() * 100:.0f}% "
          f"of weeks | avg margin {diff.mean():+.1f} | median {diff.median():+.1f}")

    # Printed as CURRENT MINUS LOOSE, so a positive number means the rules help.
    # The old orientation was loose-minus-current, which reads as "the stack cost
    # you points" to anyone who does not check the direction -- and it did fool a
    # reader on 2026-09-16 into proposing the rule be dropped. It gains points.
    print("\nDoes forcing the QB stack help? (POSITIVE = forcing the stack scored MORE)")
    both = r.pivot_table(index=["season", "week", "source"], columns="rules", values="actual")
    effect = (both["current"] - both["loose"]).groupby("source").agg(
        ["mean", lambda s: (s > 0).mean() * 100])
    effect.columns = ["avg gain from stacking", "% weeks stacking better"]
    print(effect.round(1).to_string())

    # BASE-RATE WARNING, because the raw percentage below invites the wrong read.
    # A position-legal lineup built with no stacking rule at all still stacks
    # about 20% of the time by accident (measured: 19.7% over 72,000 simulated
    # random legal lineups). So "optimal lineups stacked 53%" is not evidence
    # that stacking is optional -- it is 2.7x the rate you would get from chance.
    stacked = r[r["rules"].eq("loose")].groupby("source")["stacked"].mean().mul(100).round(0)
    print(f"\nloose lineups that stacked anyway: {stacked.to_dict()}")
    print(f"hindsight-perfect lineups that stacked: {perfect_stack:.0f}%  "
          f"(vs ~20% base rate for a random legal lineup -- so ~{perfect_stack/19.7:.1f}x chance)")

    # Same trap, same direction: the salary floor looks like a constraint that
    # costs points. Measured, it never binds -- with the floor removed, v3 and
    # FantasyData still spend over $49,000 in 100% of weeks.
    under = (r[r["rules"].eq("loose")].assign(u=lambda d: d["salary"] < 49_000)
              .groupby("source")["u"].mean().mul(100).round(1))
    print(f"\nwith the salary floor OFF, % of lineups under $49,000: {under.to_dict()}")
    print("  (0% means the floor never binds -- it is a no-op, not a cost)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="2025 only, trained on 2016-2024")
    if ap.parse_args().final:
        SEASONS = [2025]
    main()
