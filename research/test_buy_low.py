"""THE decisive test: does 'buy low after a bad game' work on a VALUE basis?

FEATURES.md 9.1 showed the rule fails on raw points -- post-bad-game WRs score
slightly less, not more. But DFS is a salary-cap game, and DK cuts the price
after a dud. The rule only needs the price to fall FARTHER than the production.

Uses RotoGuru's 2016-2021 archive, which carries salary and actual DK points in
the same row -- no join, no name matching.

Run: python research/test_buy_low.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def prep() -> pd.DataFrame:
    d = pd.read_parquet(RAW / "dk_salaries.parquet")
    d = d[d["position"].isin(["QB", "RB", "WR", "TE"]) & d["salary"].gt(0)].copy()
    d = d.sort_values(["name", "season", "week"]).reset_index(drop=True)

    g = d.groupby(["name", "season"], sort=False)
    d["pts_r4"] = g["dk_points"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=3).mean())
    d["pts_lag1"] = g["dk_points"].transform(lambda s: s.shift(1))
    d["sal_lag1"] = g["salary"].transform(lambda s: s.shift(1))
    d["games_so_far"] = g.cumcount()

    d["sal_chg_pct"] = (d["salary"] - d["sal_lag1"]) / d["sal_lag1"] * 100
    d["value"] = d["dk_points"] / (d["salary"] / 1000)   # DK points per $1,000

    # What does the market normally deliver at this price? Bucket by salary and
    # take the mean -- the baseline any pick has to beat.
    d["sal_bucket"] = (d["salary"] // 500) * 500
    d["price_baseline"] = d.groupby(["season", "position", "sal_bucket"])[
        "dk_points"].transform("mean")
    d["vs_price"] = d["dk_points"] - d["price_baseline"]
    return d


def report(d: pd.DataFrame, pos: str, min_sal: int) -> None:
    x = d[
        d["position"].eq(pos)
        & d["sal_lag1"].ge(min_sal)      # was an expensive, high-volume guy
        & d["games_so_far"].ge(4)
        & d["pts_r4"].gt(0)
        & d["pts_lag1"].notna()
        & d["sal_chg_pct"].notna()
    ].copy()

    x["group"] = np.where(x["pts_lag1"] < 0.6 * x["pts_r4"],
                          "after BAD game", "control")

    out = x.groupby("group").agg(
        n=("dk_points", "size"),
        salary_change_pct=("sal_chg_pct", "mean"),
        this_wk_salary=("salary", "mean"),
        this_wk_points=("dk_points", "mean"),
        value_per_1k=("value", "mean"),
        vs_price_baseline=("vs_price", "mean"),
    ).round(2)

    print(f"\n  {pos}, prior salary >= ${min_sal:,}")
    print(out.to_string())

    if len(out) == 2:
        bad, ctl = out.loc["after BAD game"], out.loc["control"]
        pts_gap = (bad.this_wk_points / ctl.this_wk_points - 1) * 100
        val_gap = (bad.value_per_1k / ctl.value_per_1k - 1) * 100
        print(f"    points: {pts_gap:+.1f}%   |   salary cut: {bad.salary_change_pct:+.1f}% "
              f"(control {ctl.salary_change_pct:+.1f}%)   |   VALUE: {val_gap:+.1f}%")


if __name__ == "__main__":
    d = prep()
    print(f"{len(d):,} priced player-weeks, {d.season.min()}-{d.season.max()}")
    print("=" * 78)
    print("BUY LOW: production, price, and value after a bad game")
    print("=" * 78)

    for pos, sal in [("WR", 6000), ("WR", 6500), ("WR", 7000),
                     ("RB", 6000), ("RB", 7000), ("TE", 5000)]:
        report(d, pos, sal)

    print()
    print("=" * 78)
    print("How hard does DK actually cut the price? (WR, prior salary >= $6,500)")
    print("=" * 78)
    x = d[d.position.eq("WR") & d.sal_lag1.ge(6500) & d.games_so_far.ge(4)
          & d.pts_r4.gt(0) & d.sal_chg_pct.notna()].copy()
    x["dud_severity"] = pd.cut(
        x["pts_lag1"] / x["pts_r4"],
        [-0.01, 0.3, 0.5, 0.7, 1.0, 99],
        labels=["<30% of form", "30-50%", "50-70%", "70-100%", "beat form"],
    )
    print(x.groupby("dud_severity", observed=True).agg(
        n=("salary", "size"),
        salary_change_pct=("sal_chg_pct", "mean"),
        next_points=("dk_points", "mean"),
        value_per_1k=("value", "mean"),
        vs_price_baseline=("vs_price", "mean"),
    ).round(2).to_string())
