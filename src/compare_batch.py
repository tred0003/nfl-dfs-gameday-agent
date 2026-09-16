"""Do the feature batches help? Scored against the bars that actually matter.

Trains the same XGBoost settings once per model table with walk-forward
validation, then grades every model -- plus DK salary and FantasyData's paid
projection -- on IDENTICAL player-weeks. Same rows matters: a model graded on
easier rows can look better without being better.

  base -- original features                      (model_table)
  v2   -- + injuries, routes run, red zone         (model_table_v2)
  v3   -- + salary, value momentum, carry-over     (model_table_v3)

Two scoreboards:
  1. RANKING   -- does it order each week's players by points? Bar: FantasyData.
  2. VALUE     -- does it find players who beat their SALARY? That is what a
                  contest actually pays for. "Points above salary" = actual
                  points minus what a typical player at that price scores.

Settings are held fixed so any change comes from the new data.

Run: python src/compare_batch.py            (development seasons 2021-2024)
     python src/compare_batch.py --final    (2025 only: the untouched final exam)
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
from train import TEST_SEASONS, load_table, slate_spearman, walk_forward  # noqa: E402

RAW = ROOT / "data" / "raw"
KEYS = ["player_id", "season", "week"]
TABLES = {"base": "model_table", "v2": "model_table_v2", "v3": "model_table_v3"}
PICKS_PER_POSITION = 3   # value scoreboard: top 3 QB, RB, WR, TE each week
BEAT_BY = 6


def fantasydata_benchmark() -> pd.DataFrame:
    """FantasyData salary + projection per player-week, keyed to NFL player IDs."""
    ids = pd.read_parquet(RAW / "ids.parquet")[["fantasy_data_id", "gsis_id"]].dropna()
    ids["fantasy_data_id"] = ids["fantasy_data_id"].astype(int)
    frames = []
    for f in glob.glob(str(RAW / "fantasydata" / "salaries" / "*.csv")):
        season, week = map(int, re.findall(r"dk_(\d+)_wk(\d+)", f)[0])
        frames.append(pd.read_csv(f).assign(season=season, week=week))
    fd = pd.concat(frames)
    fd = fd[fd["pos"].isin(["QB", "RB", "WR", "TE"])]
    fd["id"] = pd.to_numeric(fd["id"], errors="coerce")  # DST rows use team codes
    fd = fd.dropna(subset=["id"]).astype({"id": int})
    fd = fd.merge(ids, left_on="id", right_on="fantasy_data_id")
    return (fd.rename(columns={"gsis_id": "player_id", "salary_draftkings": "salary",
                               "fpts_draftkings": "fd_projection"})
              [["player_id", "season", "week", "salary", "fd_projection"]]
              .drop_duplicates(KEYS))


def _table(scores: dict, seasons) -> pd.DataFrame:
    return pd.DataFrame(scores, index=[str(s) for s in seasons]).T.round(4)


def main(test_seasons: list[int]) -> None:
    preds, importances, features = {}, {}, {}
    for name, table in TABLES.items():
        df, feats = load_table(table)
        print(f"training {name:<4} ({len(feats)} features)...")
        with contextlib.redirect_stdout(io.StringIO()):
            p, imp = walk_forward(df, feats, with_ceiling=False, seasons=test_seasons)
        preds[name] = p
        importances[name] = pd.DataFrame(imp).mean(axis=1)
        features[name] = feats

    both = preds["v3"][[*KEYS, "position", "dk_points", "salary_implied_pts", "dkp_ewm", "pred_mean"]]
    both = both.rename(columns={"pred_mean": "v3"})
    for name in ("base", "v2"):
        both = both.merge(preds[name][[*KEYS, "pred_mean"]].rename(columns={"pred_mean": name}), on=KEYS)
    both["position"] = both["position"].astype(str)
    models = list(TABLES)

    # ------------------------------------------------ 1. ranking, all test rows
    print("\n" + "=" * 72)
    print(f"1. RANKING -- all test rows {test_seasons[0]}-{test_seasons[-1]} (within-week rank correlation)")
    print("=" * 72)
    seasons = [*test_seasons, "all"]
    pick = lambda s: both if s == "all" else both[both["season"].eq(s)]  # noqa: E731
    print(_table({m: [slate_spearman(pick(s), m) for s in seasons] for m in models}, seasons).to_string())

    print("\nby position:")
    print(pd.DataFrame({m: {p: slate_spearman(both[both["position"].eq(p)], m) for p in ["QB", "RB", "WR", "TE"]}
                        for m in models}).T.round(4).to_string())

    print("\nweeks 1-4 vs later, and players with no game yet this season:")
    early = both[both["week"] <= 4]
    fresh = both[both["dkp_ewm"].isna()]
    print(pd.DataFrame({m: {"wk 1-4": slate_spearman(early, m),
                            "wk 5+": slate_spearman(both[both["week"] > 4], m),
                            "no game yet": slate_spearman(fresh, m)} for m in models}).T.round(4).to_string())

    # ------------------------------------------------ vs the market
    bench = both.merge(fantasydata_benchmark(), on=KEYS).dropna(subset=["salary_implied_pts"])
    cols = {"fd_projection": "FantasyData", "v3": "v3", "v2": "v2", "base": "base", "salary": "DK salary"}
    print("\n" + "=" * 72)
    bench_seasons = sorted(bench["season"].unique())
    print(f"1b. RANKING vs THE MARKET -- {bench_seasons[0]}-{bench_seasons[-1]}, {len(bench):,} identical player-weeks")
    print("=" * 72)
    seasons = [*bench_seasons, "all"]
    pickb = lambda s: bench if s == "all" else bench[bench["season"].eq(s)]  # noqa: E731
    print(_table({lab: [slate_spearman(pickb(s), c) for s in seasons] for c, lab in cols.items()},
                 seasons).to_string())
    bench_early = bench[bench["week"] <= 4]
    print("\nweeks 1-4 only:",
          "  ".join(f"{lab} {slate_spearman(bench_early, c):.4f}" for c, lab in cols.items()))

    # ------------------------------------------------ 2. value
    print("\n" + "=" * 72)
    print(f"2. VALUE -- does it find players who beat their salary? ({bench_seasons[0]}-{bench_seasons[-1]})")
    print("=" * 72)
    b = bench.copy()
    b["actual_value"] = b["dk_points"] - b["salary_implied_pts"]
    b["beat"] = b["actual_value"] >= BEAT_BY
    sources = {"FantasyData": "fd_projection", "v3": "v3", "v2": "v2", "base": "base"}
    rows = []
    for lab, c in sources.items():
        b["pred_value"] = b[c] - b["salary_implied_pts"]
        rank = slate_spearman(b.assign(dk_points=b["actual_value"]), "pred_value")
        top = (b.sort_values("pred_value", ascending=False)
                .groupby(["season", "week", "position"]).head(PICKS_PER_POSITION))
        rows.append({"source": lab, "value_rank_corr": rank,
                     "picks": len(top),
                     "pts_above_salary_per_pick": top["actual_value"].mean(),
                     f"beat_by_{BEAT_BY}+_%": top["beat"].mean() * 100})
    rows.append({"source": "every player (baseline)", "value_rank_corr": np.nan, "picks": len(b),
                 "pts_above_salary_per_pick": b["actual_value"].mean(),
                 f"beat_by_{BEAT_BY}+_%": b["beat"].mean() * 100})
    print(pd.DataFrame(rows).set_index("source").round(3).to_string())
    print(f"\n  picks = top {PICKS_PER_POSITION} per position per week by (projection - salary line)")

    # ------------------------------------------------ what v3 leaned on
    new = [f for f in features["v3"] if f not in features["v2"]]
    imp = importances["v3"]
    print("\nbatch 3 features, share of v3's work (%):")
    print((imp[new].sort_values(ascending=False) * 100).round(2).to_string())
    print(f"  total from batch 3: {imp[new].sum() * 100:.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true", help="grade 2025 only, trained on 2016-2024")
    main([2025] if ap.parse_args().final else TEST_SEASONS)
