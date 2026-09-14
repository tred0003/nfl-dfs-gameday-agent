"""Backtest the user's own DFS heuristics against 2016-2024.

Not modeling -- just asking whether each rule shows up in nine seasons of data,
and if so, in what form. A rule that survives becomes a feature; a rule that
dies teaches us something more useful than a rule we never checked.

Run: python research/test_heuristics.py
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
from scoring import add_dk_points  # noqa: E402

RAW = ROOT / "data" / "raw"


def load() -> pd.DataFrame:
    w = pd.read_parquet(RAW / "weekly.parquet")
    w = w[w["season_type"].eq("REG")].copy()
    w = add_dk_points(w)
    return w.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def add_shifted_baselines(w: pd.DataFrame) -> pd.DataFrame:
    """Prior-form features. Every one is shifted so week N sees only < N."""
    g = w.groupby(["player_id", "season"], sort=False)

    for col, name in [("dk_points", "dkp"), ("target_share", "tgt"), ("targets", "tgts")]:
        w[f"{name}_r4"] = g[col].transform(lambda s: s.shift(1).rolling(4, min_periods=3).mean())
        w[f"{name}_lag1"] = g[col].transform(lambda s: s.shift(1))

    w["games_so_far"] = g.cumcount()
    return w


# ---------------------------------------------------------------- H1
def h1_bounce_back(w: pd.DataFrame) -> None:
    """'A high-volume WR with 1-2 bad games is due -- the QB force-feeds him.'

    The critical split: did the VOLUME stay high while the points dropped
    (bad luck, should regress up), or did the volume drop too (role change,
    should not)? Same 'bad game' to the naked eye, opposite meaning.
    """
    print("=" * 78)
    print("H1: WR bounce-back after a bad game")
    print("=" * 78)

    d = w[
        w["position"].eq("WR")
        & w["tgt_r4"].ge(0.18)          # established WR1/WR2 role
        & w["games_so_far"].ge(4)
        & w["dkp_r4"].gt(0)
        & w["tgt_lag1"].notna()
    ].copy()

    bad = d["dkp_lag1"] < 0.6 * d["dkp_r4"]        # scored well under own baseline
    held = d["tgt_lag1"] >= 0.9 * d["tgt_r4"]      # but kept his targets

    d["group"] = np.select(
        [bad & held, bad & ~held, ~bad],
        ["BAD GAME, volume held", "BAD GAME, volume dropped", "no bad game (control)"],
        default="other",
    )

    # Outcome: this week's points vs. the player's own prior baseline.
    d["vs_baseline"] = d["dk_points"] - d["dkp_r4"]

    out = d.groupby("group").agg(
        n=("dk_points", "size"),
        mean_pts=("dk_points", "mean"),
        mean_vs_baseline=("vs_baseline", "mean"),
        pct_20plus=("dk_points", lambda s: (s >= 20).mean() * 100),
        next_tgt_share=("target_share", "mean"),
    ).round(2)
    print(out.to_string())

    a = d.loc[d["group"].eq("BAD GAME, volume held"), "vs_baseline"]
    b = d.loc[d["group"].eq("BAD GAME, volume dropped"), "vs_baseline"]
    print(f"\n  volume-held minus volume-dropped: {a.mean() - b.mean():+.2f} DK pts")
    print(f"  (n={len(a)} vs n={len(b)})")


# ---------------------------------------------------------------- H2 / H3
def h2_inexperienced_qb(w: pd.DataFrame) -> None:
    """'Rookie/backup QBs force-feed the WR1 -- and probably run more.'"""
    print()
    print("=" * 78)
    print("H2/H3: inexperienced QB -> WR1 usage, and team run rate")
    print("=" * 78)

    s = pd.read_parquet(RAW / "schedules.parquet")
    s = s[s["game_type"].eq("REG")]

    starts = pd.concat([
        s[["season", "week", "home_team", "home_qb_id"]].rename(
            columns={"home_team": "team", "home_qb_id": "qb_id"}),
        s[["season", "week", "away_team", "away_qb_id"]].rename(
            columns={"away_team": "team", "away_qb_id": "qb_id"}),
    ]).dropna(subset=["qb_id"]).sort_values(["season", "week"])

    # Career starts BEFORE this game (within our window -- so early-window
    # veterans look inexperienced; we drop 2016-17 to blunt that).
    starts["prior_starts"] = starts.groupby("qb_id").cumcount()
    starts = starts[starts["season"] >= 2018]
    starts["green_qb"] = starts["prior_starts"] <= 5

    # WR1 = highest prior-form target share on the team that week
    wr = w[w["position"].eq("WR") & w["tgt_r4"].notna()].copy()
    wr["wr_rank"] = wr.groupby(["recent_team", "season", "week"])["tgt_r4"].rank(
        ascending=False, method="first")
    wr1 = wr[wr["wr_rank"].eq(1)].rename(columns={"recent_team": "team"})

    m = wr1.merge(starts, on=["season", "week", "team"], how="inner")
    print("\n  WR1 production by QB experience:")
    print(m.groupby("green_qb").agg(
        n=("dk_points", "size"),
        wr1_target_share=("target_share", "mean"),
        wr1_targets=("targets", "mean"),
        wr1_dk_points=("dk_points", "mean"),
    ).round(3).to_string())

    # Team run rate: carries vs pass attempts, aggregated from player rows
    team = w.groupby(["recent_team", "season", "week"]).agg(
        carries=("carries", "sum"), attempts=("attempts", "sum")
    ).reset_index().rename(columns={"recent_team": "team"})
    team["run_rate"] = team["carries"] / (team["carries"] + team["attempts"])

    tm = team.merge(starts, on=["season", "week", "team"], how="inner")
    print("\n  Team run rate by QB experience:")
    print(tm.groupby("green_qb").agg(
        n=("run_rate", "size"),
        run_rate=("run_rate", "mean"),
        carries=("carries", "mean"),
    ).round(3).to_string())


# ---------------------------------------------------------------- H4
def h4_weather(w: pd.DataFrame) -> None:
    """'Rain means more RB carries and production.'

    No precipitation column exists in nflverse schedules -- only temp/wind/roof.
    Wind is the closest available proxy, so that is what gets tested here.
    """
    print()
    print("=" * 78)
    print("H4: bad weather -> RB usage  (WIND proxy; no precip data available)")
    print("=" * 78)

    s = pd.read_parquet(RAW / "schedules.parquet")
    s = s[s["game_type"].eq("REG")]
    g = pd.concat([
        s[["season", "week", "home_team", "wind", "temp", "roof"]].rename(
            columns={"home_team": "team"}),
        s[["season", "week", "away_team", "wind", "temp", "roof"]].rename(
            columns={"away_team": "team"}),
    ])

    team = w.groupby(["recent_team", "season", "week"]).agg(
        carries=("carries", "sum"), attempts=("attempts", "sum")
    ).reset_index().rename(columns={"recent_team": "team"})
    team["run_rate"] = team["carries"] / (team["carries"] + team["attempts"])

    m = team.merge(g, on=["season", "week", "team"], how="inner")
    m = m[m["roof"].isin(["outdoors", "open"])].dropna(subset=["wind"])
    m["wind_band"] = pd.cut(m["wind"], [-1, 5, 10, 15, 100],
                            labels=["0-5", "6-10", "11-15", "15+"])

    print(m.groupby("wind_band", observed=True).agg(
        n=("run_rate", "size"),
        run_rate=("run_rate", "mean"),
        carries=("carries", "mean"),
        pass_att=("attempts", "mean"),
    ).round(3).to_string())

    rb = w[w["position"].eq("RB")].merge(
        g.rename(columns={"team": "recent_team"}),
        on=["recent_team", "season", "week"], how="inner")
    rb = rb[rb["roof"].isin(["outdoors", "open"])].dropna(subset=["wind"])
    rb["windy"] = rb["wind"] > 15
    print("\n  RB points, windy (>15mph) vs not:")
    print(rb.groupby("windy").agg(
        n=("dk_points", "size"),
        rb_dk_points=("dk_points", "mean"),
        rb_carries=("carries", "mean"),
    ).round(2).to_string())


if __name__ == "__main__":
    w = add_shifted_baselines(load())
    print(f"loaded {len(w):,} player-weeks, {w.season.min()}-{w.season.max()}\n")
    h1_bounce_back(w)
    h2_inexperienced_qb(w)
    h4_weather(w)
