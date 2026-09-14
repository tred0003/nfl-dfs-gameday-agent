"""DraftKings NFL Classic lineup optimizer -- integer linear program.

Deliberately NOT machine learning. Given projections, the best lineup under the
cap is a solved problem with an exact answer, and it should be reproducible to
the cent. Putting an LLM or a heuristic search here would make it slower, less
accurate, and impossible to debug. Knowing where the model stops and the solver
starts is the point.

DK Classic roster: QB, RB, RB, WR, WR, WR, TE, FLEX (RB/WR/TE), DST -- 9 players,
$50,000 cap, and players from at least two different games.

Run: python src/optimize.py --slate data/slates/dg151307/players_<ts>.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pulp

SALARY_CAP = 50_000
ROSTER_SIZE = 9

# (min, max) counts once FLEX is accounted for
POSITION_LIMITS = {
    "QB": (1, 1),
    "RB": (2, 3),
    "WR": (3, 4),
    "TE": (1, 2),
    "DST": (1, 1),
}
FLEX_TOTAL = 7          # RB + WR + TE always sums to 7


def optimize(
    pool: pd.DataFrame,
    proj_col: str = "projection",
    n_lineups: int = 1,
    max_per_team: int = 4,
    stack_qb: bool = True,
    min_salary: int = 49_000,
    min_differ: int = 3,
    block_dst_vs_offense: bool = True,
) -> list[pd.DataFrame]:
    """Return the top `n_lineups` lineups, each differing from the others.

    max_per_team  -- caps correlated downside from one team busting
    stack_qb      -- force >=1 pass catcher from the QB's team (QB and his
                     receivers score together; a lineup that ignores that
                     leaves tournament equity on the table)
    min_salary    -- leaving money unspent is almost always a mistake
    min_differ    -- how many players each lineup must differ by. At 1 the
                     solver returns near-identical lineups that share every
                     correlated risk, which defeats the point of entering more
                     than one.
    block_dst_vs_offense -- never roster a defense against your own skill
                     players. You would be betting the offense scores a lot AND
                     the defense facing it does well. Those cannot both happen.
    """
    pool = pool.reset_index(drop=True)
    valid = pool[proj_col].notna() & pool["salary"].gt(0)
    pool = pool[valid].reset_index(drop=True)

    lineups: list[pd.DataFrame] = []
    previous: list[list[int]] = []

    for _ in range(n_lineups):
        prob = pulp.LpProblem("dk_lineup", pulp.LpMaximize)
        x = {i: pulp.LpVariable(f"p{i}", cat="Binary") for i in pool.index}

        prob += pulp.lpSum(pool.loc[i, proj_col] * x[i] for i in pool.index)

        prob += pulp.lpSum(x.values()) == ROSTER_SIZE
        prob += pulp.lpSum(pool.loc[i, "salary"] * x[i] for i in pool.index) <= SALARY_CAP
        prob += pulp.lpSum(pool.loc[i, "salary"] * x[i] for i in pool.index) >= min_salary

        for pos, (lo, hi) in POSITION_LIMITS.items():
            idx = pool.index[pool["position"].eq(pos)]
            prob += pulp.lpSum(x[i] for i in idx) >= lo
            prob += pulp.lpSum(x[i] for i in idx) <= hi

        flex_idx = pool.index[pool["position"].isin(["RB", "WR", "TE"])]
        prob += pulp.lpSum(x[i] for i in flex_idx) == FLEX_TOTAL

        for team in pool["team"].dropna().unique():
            idx = pool.index[pool["team"].eq(team)]
            prob += pulp.lpSum(x[i] for i in idx) <= max_per_team

        # DK requires players from at least two different games
        games = pool["game"].dropna().unique()
        if len(games) > 1:
            for g in games:
                idx = pool.index[pool["game"].eq(g)]
                prob += pulp.lpSum(x[i] for i in idx) <= ROSTER_SIZE - 1

        if stack_qb:
            for team in pool["team"].dropna().unique():
                qbs = pool.index[pool["team"].eq(team) & pool["position"].eq("QB")]
                catchers = pool.index[pool["team"].eq(team)
                                      & pool["position"].isin(["WR", "TE"])]
                if len(qbs) and len(catchers):
                    # if the QB is rostered, at least one of his catchers must be
                    prob += pulp.lpSum(x[i] for i in catchers) >= pulp.lpSum(x[i] for i in qbs)

        if block_dst_vs_offense and "opp_team" in pool.columns:
            for team in pool["team"].dropna().unique():
                dsts = pool.index[pool["team"].eq(team) & pool["position"].eq("DST")]
                if not len(dsts):
                    continue
                opp = pool.loc[dsts, "opp_team"].iloc[0]
                foes = pool.index[pool["team"].eq(opp)
                                  & pool["position"].ne("DST")]
                for j in foes:
                    # DST and an opposing skill player cannot both be rostered
                    prob += pulp.lpSum(x[i] for i in dsts) + x[j] <= 1

        # force each new lineup to differ from every previous one
        for prev in previous:
            prob += pulp.lpSum(x[i] for i in prev) <= ROSTER_SIZE - min_differ

        status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[status] != "Optimal":
            break

        chosen = [i for i in pool.index if x[i].value() == 1]
        previous.append(chosen)
        lineups.append(pool.loc[chosen].sort_values(
            "position", key=lambda s: s.map(
                {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "DST": 4})))

    return lineups


def show(lineup: pd.DataFrame, proj_col: str = "projection", n: int = 1) -> None:
    cols = [c for c in ["name", "position", "team", "game", "salary", proj_col,
                        "status"] if c in lineup.columns]
    print(f"\n--- lineup {n} ---")
    print(lineup[cols].to_string(index=False))
    print(f"    salary ${lineup['salary'].sum():,} / ${SALARY_CAP:,}"
          f"   |   projected {lineup[proj_col].sum():.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--slate", required=True, help="parquet with projections")
    p.add_argument("--proj-col", default="projection")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--max-per-team", type=int, default=4)
    p.add_argument("--min-differ", type=int, default=3)
    p.add_argument("--no-stack", action="store_true")
    p.add_argument("--allow-dst-vs-offense", action="store_true")
    a = p.parse_args()

    pool = pd.read_parquet(a.slate)
    outs = optimize(pool, a.proj_col, a.n, a.max_per_team, not a.no_stack,
                    min_differ=a.min_differ,
                    block_dst_vs_offense=not a.allow_dst_vs_offense)
    if not outs:
        raise SystemExit("no feasible lineup -- check the pool and constraints")
    for i, lu in enumerate(outs, 1):
        show(lu, a.proj_col, i)
