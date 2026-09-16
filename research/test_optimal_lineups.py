"""What did the perfect lineup look like? Hindsight-optimal DK lineups, 2016-2021.

For every week, solve for the best possible lineups using ACTUAL points, with
none of our optimizer's construction rules switched on (no forced QB stack, no
team cap, no minimum spend). Then measure what those lineups had in common.

This tests the rules hard-coded in src/optimize.py -- which were assumptions,
never checked:
  - always stack the QB with one of his pass catchers
  - at most 4 players from one team
  - spend at least $49,000
  - never play a defense against your own skill players

Data: RotoGuru (salary + actual points in one row, defenses included).
Caveat: RotoGuru prices the whole week (Thursday-Monday), so these lineups
could use games the Sunday main slate leaves out. Hindsight lineups also
describe what WON, not what was predictable -- this is about lineup shape,
not player picks.

Run: python research/test_optimal_lineups.py      (~5-10 minutes)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from optimize import optimize  # noqa: E402

TOP_N = 5  # best 5 distinct lineups per week, not just the single best


def weekly_pools() -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "data" / "raw" / "dk_salaries.parquet")
    d = d[d["salary"].gt(0) & d["dk_points"].notna()].copy()
    d["position"] = d["position"].replace({"DEF": "DST"})
    d = d[d["position"].isin(["QB", "RB", "WR", "TE", "DST"])]
    d["team"] = d["team"].str.upper()
    d["opp_team"] = d["opponent"].str.upper()
    d["game"] = [" v ".join(sorted(p)) for p in zip(d["team"], d["opp_team"])]
    return d


def describe(lu: pd.DataFrame) -> dict:
    qb = lu[lu["position"].eq("QB")].iloc[0]
    catchers = lu[lu["position"].isin(["WR", "TE"])]
    same_team = catchers["team"].eq(qb["team"]).sum()
    skill = lu[lu["position"].ne("DST")]
    dst = lu[lu["position"].eq("DST")].iloc[0]
    per_team = skill["team"].value_counts()
    per_game = lu["game"].value_counts()
    flex_pool = lu[lu["position"].isin(["RB", "WR", "TE"])]["position"].value_counts()
    flex = next(p for p, n in [("RB", flex_pool.get("RB", 0) - 2),
                               ("WR", flex_pool.get("WR", 0) - 3),
                               ("TE", flex_pool.get("TE", 0) - 1)] if n > 0)
    return {
        "points": lu["dk_points"].sum(),
        "salary_used": lu["salary"].sum(),
        "qb_stacked": int(same_team >= 1),
        "qb_double_stack": int(same_team >= 2),
        "bring_back": int(skill[skill["team"].eq(qb["opp_team"])].shape[0] >= 1),
        "max_from_one_team": int(per_team.max()),
        "max_from_one_game": int(per_game.max()),
        "dst_vs_own_player": int(skill["team"].eq(dst["opp_team"]).any()),
        "players_under_4k": int(lu["salary"].lt(4000).sum()),
        "players_8k_plus": int(lu["salary"].ge(8000).sum()),
        "flex": flex,
        "qb_salary": qb["salary"], "dst_salary": dst["salary"],
    }


def main() -> None:
    pools = weekly_pools()
    rows = []
    weeks = pools.groupby(["season", "week"])
    for i, ((season, week), pool) in enumerate(weeks, 1):
        lineups = optimize(pool, proj_col="dk_points", n_lineups=TOP_N,
                           max_per_team=8, stack_qb=False, min_salary=0,
                           min_differ=2, block_dst_vs_offense=False)
        for rank, lu in enumerate(lineups, 1):
            rows.append({"season": season, "week": week, "rank": rank, **describe(lu)})
        if i % 20 == 0:
            print(f"  solved {i}/{weeks.ngroups} weeks", flush=True)

    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "research" / "optimal_lineups_2016_2021.csv", index=False)
    n = r[["season", "week"]].drop_duplicates().shape[0]
    print(f"\n{len(r)} hindsight lineups ({TOP_N} per week, {n} weeks)\n")

    pct = lambda c: f"{r[c].mean() * 100:5.1f}%"  # noqa: E731
    print("RULE CHECKS vs src/optimize.py")
    print(f"  QB stacked with 1+ of his WR/TE ...... {pct('qb_stacked')}   (optimizer FORCES 100%)")
    print(f"  QB double stack (2+ catchers) ........ {pct('qb_double_stack')}")
    print(f"  bring-back (player from QB's opponent) {pct('bring_back')}")
    print(f"  more than 4 skill players, one team .. {(r['max_from_one_team'] > 4).mean() * 100:5.1f}%   (optimizer BLOCKS)")
    print(f"  DST facing one of its own lineup's players {pct('dst_vs_own_player')}   (optimizer BLOCKS)")
    print(f"  spent under $49,000 .................. {(r['salary_used'] < 49000).mean() * 100:5.1f}%   (optimizer BLOCKS)")

    print("\nSHAPE")
    print(f"  avg points {r['points'].mean():.1f} | avg salary ${r['salary_used'].mean():,.0f}")
    print(f"  max players from one game: {r['max_from_one_game'].value_counts(normalize=True).sort_index().mul(100).round(1).to_dict()}")
    print(f"  players under $4k per lineup: {r['players_under_4k'].mean():.2f} | $8k+: {r['players_8k_plus'].mean():.2f}")
    print(f"  FLEX position: {r['flex'].value_counts(normalize=True).mul(100).round(1).to_dict()}")
    print(f"  avg QB salary ${r['qb_salary'].mean():,.0f} | avg DST salary ${r['dst_salary'].mean():,.0f}")
    print("\nby season (stack rate, avg points):")
    print(r.groupby("season").agg(qb_stacked=("qb_stacked", "mean"), points=("points", "mean"),
                                  under_4k=("players_under_4k", "mean")).round(2).to_string())


if __name__ == "__main__":
    main()
