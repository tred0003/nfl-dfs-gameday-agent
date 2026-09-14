"""Does RB production fall off late in the season, and does age drive it?

Two competing stories:
  (a) THE WALL -- accumulated carries wear a back down, worse with age.
  (b) PACING -- teams manage older backs on purpose, so decline is a coaching
      decision rather than a physical one.

They make different predictions. If it is the wall, EFFICIENCY (yards per carry)
should fall. If it is pacing, USAGE (carries) should fall while efficiency holds.
Splitting production into those two pieces is what separates them.

Run: python research/test_rb_age_wall.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from scoring import add_dk_points  # noqa: E402

RAW = ROOT / "data" / "raw"
EARLY, LATE = (1, 6), (12, 17)


def load_rbs() -> pd.DataFrame:
    w = pd.read_parquet(RAW / "weekly.parquet")
    w = w[w["season_type"].eq("REG") & w["position"].eq("RB")].copy()
    w = add_dk_points(w)

    years = sorted(w["season"].unique().tolist())
    r = nfl.import_seasonal_rosters(years)
    r = (r.dropna(subset=["age"])
           .sort_values("age")
           .drop_duplicates(subset=["player_id", "season"], keep="first")
           [["player_id", "season", "age", "years_exp"]])

    w = w.merge(r, on=["player_id", "season"], how="inner")
    w["ypc"] = w["rushing_yards"] / w["carries"].replace(0, pd.NA)
    return w.sort_values(["player_id", "season", "week"])


def early_late(w: pd.DataFrame) -> pd.DataFrame:
    """Per RB-season: early-window vs late-window production, usage, efficiency."""
    e = w[w["week"].between(*EARLY)]
    l = w[w["week"].between(*LATE)]

    agg = {"dk_points": "mean", "carries": "mean", "targets": "mean",
           "rushing_yards": "sum", "ypc": "mean"}
    E = e.groupby(["player_id", "season"]).agg(agg)
    L = l.groupby(["player_id", "season"]).agg(agg)
    games_e = e.groupby(["player_id", "season"]).size().rename("g_early")
    games_l = l.groupby(["player_id", "season"]).size().rename("g_late")

    d = E.join(L, lsuffix="_early", rsuffix="_late").join(games_e).join(games_l)
    d = d[(d["g_early"] >= 4) & (d["g_late"] >= 3)]          # real workloads only
    d = d[d["carries_early"] >= 8]                            # actual RBs, not scatbacks

    age = w.groupby(["player_id", "season"])["age"].first()
    cum = e.groupby(["player_id", "season"])["carries"].sum().rename("early_carry_load")
    d = d.join(age).join(cum)

    d["pts_delta"] = d["dk_points_late"] - d["dk_points_early"]
    d["pts_delta_pct"] = d["pts_delta"] / d["dk_points_early"] * 100
    d["carry_delta"] = d["carries_late"] - d["carries_early"]
    d["ypc_delta"] = d["ypc_late"] - d["ypc_early"]
    return d.reset_index()


def main() -> None:
    w = load_rbs()
    d = early_late(w)
    print(f"{len(d):,} RB-seasons with a real early AND late workload "
          f"({w.season.min()}-{w.season.max()})\n")

    d["age_band"] = pd.cut(d["age"], [0, 24, 26, 28, 99],
                           labels=["<=24", "25-26", "27-28", "29+"])

    print("=" * 78)
    print("LATE-SEASON CHANGE BY AGE  (weeks 12-17 vs weeks 1-6, same player-season)")
    print("=" * 78)
    print(d.groupby("age_band", observed=True).agg(
        n=("pts_delta", "size"),
        pts_early=("dk_points_early", "mean"),
        pts_late=("dk_points_late", "mean"),
        pts_change_pct=("pts_delta_pct", "mean"),
        carries_early=("carries_early", "mean"),
        carries_late=("carries_late", "mean"),
        ypc_change=("ypc_delta", "mean"),
    ).round(2).to_string())

    print()
    print("=" * 78)
    print("THE WALL: does a heavy early workload cause the late drop?")
    print("=" * 78)
    d["load_band"] = pd.qcut(d["early_carry_load"], 4,
                             labels=["lightest 25%", "2nd", "3rd", "heaviest 25%"])
    print(d.groupby("load_band", observed=True).agg(
        n=("pts_delta", "size"),
        early_carries_total=("early_carry_load", "mean"),
        pts_change_pct=("pts_delta_pct", "mean"),
        carry_change=("carry_delta", "mean"),
        ypc_change=("ypc_delta", "mean"),
    ).round(2).to_string())

    print()
    print("=" * 78)
    print("OLD + HEAVILY USED -- the combination the theory actually predicts")
    print("=" * 78)
    d["old"] = d["age"] >= 27
    d["heavy"] = d["early_carry_load"] >= d["early_carry_load"].median()
    print(d.groupby(["old", "heavy"], observed=True).agg(
        n=("pts_delta", "size"),
        pts_change_pct=("pts_delta_pct", "mean"),
        carry_change=("carry_delta", "mean"),
        ypc_change=("ypc_delta", "mean"),
    ).round(2).to_string())


if __name__ == "__main__":
    main()
