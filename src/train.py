"""Train and evaluate the projection model with walk-forward validation.

This is where the ML actually starts. Three things make it modeling rather than
data plumbing:

1. WALK-FORWARD VALIDATION. Train on seasons < S, test on S, step S forward.
   Standard k-fold would shuffle 2024 games into the training set and grade the
   model on a season it had already seen. The score would be excellent and the
   model would lose money.

2. THE METRIC IS RANKING, NOT ACCURACY. A lineup optimizer never uses the
   predicted number -- it uses the ORDER. Being wrong by 4 points on everyone
   costs nothing if the ordering holds. So the headline metric is Spearman rank
   correlation computed WITHIN each week's slate, which is the decision the
   model is actually asked to make. RMSE is reported as a secondary check.

3. BASELINES IT HAS TO BEAT. A model is only as good as the dumb thing it
   replaces. Three of them here: the player's own rolling average, a Vegas-only
   model, and season-to-date mean. Beating none of them means the features add
   nothing.

Run: python src/train.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"

TARGET = "dk_points"
TEST_SEASONS = [2021, 2022, 2023, 2024]

PARAMS = dict(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,
    reg_lambda=2.0,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)


# ------------------------------------------------------------------- metrics
def slate_spearman(df: pd.DataFrame, pred_col: str) -> float:
    """Mean within-week rank correlation -- the metric that matches the task.

    Pooling all weeks would let the model score well just by knowing that good
    players outscore bad ones. Ranking WITHIN a slate is the real question.
    """
    rhos = []
    for _, wk in df.groupby(["season", "week"]):
        if len(wk) < 10:
            continue
        rho = spearmanr(wk[pred_col], wk[TARGET]).statistic
        if np.isfinite(rho):
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else np.nan


def evaluate(df: pd.DataFrame, pred_col: str, label: str) -> dict:
    d = df.dropna(subset=[pred_col, TARGET])
    return {
        "model": label,
        "n": len(d),
        "slate_spearman": round(slate_spearman(d, pred_col), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(d[TARGET], d[pred_col]))), 3),
        "mae": round(float(mean_absolute_error(d[TARGET], d[pred_col])), 3),
    }


# ----------------------------------------------------------------- training
def walk_forward(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict]:
    """Train on everything before season S, predict S. Repeat."""
    preds, importances = [], {}

    for season in TEST_SEASONS:
        tr = df[df["season"] < season]
        te = df[df["season"] == season]
        if tr.empty or te.empty:
            continue

        Xtr, ytr = tr[features], tr[TARGET]
        Xte = te[features]

        mean_model = xgb.XGBRegressor(objective="reg:squarederror", **PARAMS)
        mean_model.fit(Xtr, ytr, verbose=False)

        # Tournaments pay the tail, not the average. An 85th-percentile model
        # answers "how high can this player go", which is the actual GPP
        # question -- a different target, not just a different threshold.
        ceil_model = xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=0.85, **PARAMS
        )
        ceil_model.fit(Xtr, ytr, verbose=False)

        out = te.copy()
        out["pred_mean"] = mean_model.predict(Xte)
        out["pred_ceiling"] = ceil_model.predict(Xte)
        preds.append(out)

        importances[season] = dict(
            zip(features, mean_model.feature_importances_.astype(float))
        )
        print(f"  season {season}: trained on {len(tr):,}, predicted {len(te):,}")

    return pd.concat(preds, ignore_index=True), importances


def add_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """The dumb models the ML has to beat."""
    d = df.copy()
    # 1. the player's own recent form
    d["base_rolling"] = d["dkp_r5"].fillna(d["dkp_r3"])
    # 2. Vegas only -- no player history at all
    d["base_vegas"] = d["implied_team_total"]
    # 3. flat league average
    d["base_constant"] = d[TARGET].mean()
    return d


def main(save: bool) -> None:
    df = pd.read_parquet(PROC / "model_table.parquet")
    features = [c for c in df.columns if c not in {
        "player_id", "player_display_name", "position", "recent_team",
        "opponent_team", "season", "week", TARGET,
    }]
    # position is categorical and genuinely predictive -- one-hot it
    df = pd.concat([df, pd.get_dummies(df["position"], prefix="pos")], axis=1)
    features += [c for c in df.columns if c.startswith("pos_")]

    print(f"model table {df.shape[0]:,} rows, {len(features)} features")
    print(f"walk-forward over seasons {TEST_SEASONS}\n")

    preds, importances = walk_forward(df, features)
    preds = add_baselines(preds)

    rows = [
        evaluate(preds, "pred_mean", "XGBoost (mean)"),
        evaluate(preds, "pred_ceiling", "XGBoost (85th pct)"),
        evaluate(preds, "base_rolling", "baseline: rolling avg"),
        evaluate(preds, "base_vegas", "baseline: Vegas only"),
        evaluate(preds, "base_constant", "baseline: constant"),
    ]
    res = pd.DataFrame(rows).set_index("model")
    print("\n" + "=" * 72)
    print("HELD-OUT PERFORMANCE (2021-2024, never trained on)")
    print("=" * 72)
    print(res.to_string())

    print("\nby position (XGBoost mean, slate-rank correlation):")
    for pos in ["QB", "RB", "WR", "TE"]:
        sub = preds[preds["position"].astype(str).eq(pos)]
        if len(sub) > 100:
            print(f"  {pos}: {slate_spearman(sub, 'pred_mean'):.4f}  (n={len(sub):,})")

    imp = pd.DataFrame(importances).mean(axis=1).sort_values(ascending=False)
    print("\ntop 15 features:")
    print((imp.head(15) * 100).round(2).to_string())

    if save:
        MODELS.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(MODELS / "walk_forward_preds.parquet", index=False)
        (MODELS / "metrics.json").write_text(json.dumps(rows, indent=2))
        print(f"\nsaved predictions + metrics to {MODELS}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-save", action="store_true")
    a = p.parse_args()
    main(save=not a.no_save)
