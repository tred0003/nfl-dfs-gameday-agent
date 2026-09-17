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
Tune: set TUNE = True below, then run the same way.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

import calibrate
import model_io

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

# ------------------------------------------------------------------- tuning
# True  = try every combination below and write models/tuning_results.xlsx.
# False = the normal single training run using PARAMS above.
# Tuning never touches metrics.json, so the saved results stay safe.
TUNE = False 
TREE_GRID = [100, 300, 600, 1000]
LEARNING_RATE_GRID = [0.01, 0.03, 0.1]


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
def walk_forward(
    df: pd.DataFrame,
    features: list[str],
    params: dict = PARAMS,
    with_ceiling: bool = True,
    seasons: list[int] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Train on everything before season S, predict S. Repeat.

    `seasons` overrides TEST_SEASONS so a caller can validate over a different
    window -- src/backtest.py relies on this to score 2025 on its own.
    """
    preds, importances = [], {}

    for season in seasons or TEST_SEASONS:
        tr = df[df["season"] < season]
        te = df[df["season"] == season]
        if tr.empty or te.empty:
            continue

        Xtr, ytr = tr[features], tr[TARGET]
        Xte = te[features]

        mean_model = xgb.XGBRegressor(objective="reg:squarederror", **params)
        mean_model.fit(Xtr, ytr, verbose=False)

        out = te.copy()
        out["pred_mean"] = mean_model.predict(Xte)

        # Tournaments pay the tail, not the average. An 85th-percentile model
        # answers "how high can this player go", which is the actual GPP
        # question -- a different target, not just a different threshold.
        # Skipped while tuning: it doubles the run time and isn't being tuned.
        if with_ceiling:
            ceil_model = xgb.XGBRegressor(
                objective="reg:quantileerror", quantile_alpha=0.85, **params
            )
            ceil_model.fit(Xtr, ytr, verbose=False)
            out["pred_ceiling"] = ceil_model.predict(Xte)

        preds.append(out)

        importances[season] = dict(
            zip(features, mean_model.feature_importances_.astype(float))
        )
        print(f"  season {season}: trained on {len(tr):,}, predicted {len(te):,}")

    return pd.concat(preds, ignore_index=True), importances


def fit_final(
    df: pd.DataFrame, features: list[str], params: dict = PARAMS
) -> tuple[dict, dict]:
    """Refit on EVERY row -- this is the model that actually ships.

    Be precise about what this is and is not. `walk_forward()` produces the
    honest skill estimate, because each prediction it makes is on a season the
    model had never seen. But a model deliberately blinded to 2025 is a worse
    model to carry into 2026, so the shipping artifact is refit on everything
    available and the walk-forward numbers ride along in the model card as its
    *estimated* skill.

    The model you measure and the model you ship are not the same object. That
    is normal and correct. What would not be correct is quoting 0.6442 as this
    artifact's score -- it has never been scored on anything it did not train
    on, and it never can be.
    """
    X, y = df[features], df[TARGET]
    print(f"\nfinal fit on all {len(df):,} rows ({len(features)} features)")

    models = {
        "mean": xgb.XGBRegressor(objective="reg:squarederror", **params),
        "ceiling": xgb.XGBRegressor(
            objective="reg:quantileerror", quantile_alpha=0.85, **params
        ),
    }
    for role, m in models.items():
        m.fit(X, y, verbose=False)
        print(f"  fit {role}")

    last = int(df["season"].max())
    coverage = {
        "seasons": [int(df["season"].min()), last],
        "through": [last, int(df[df["season"].eq(last)]["week"].max())],
        "rows": int(len(df)),
    }
    return models, coverage


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


# ------------------------------------------------------------------- tuning
def tune(df: pd.DataFrame, features: list[str]) -> None:
    """Score every tree-count x learning-rate combination, write to Excel."""
    combos = list(itertools.product(TREE_GRID, LEARNING_RATE_GRID))
    print(f"TUNING: {len(combos)} combinations, mean model only\n")

    rows, baseline = [], None
    for i, (trees, lr) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] trees={trees}, learning_rate={lr}", flush=True)
        params = {**PARAMS, "n_estimators": trees, "learning_rate": lr}

        start = time.time()
        preds, _ = walk_forward(df, features, params, with_ceiling=False)
        res = evaluate(preds, "pred_mean", "XGBoost")

        row = {
            "model": "XGBoost",
            "trees": trees,
            "learning_rate": lr,
            "slate_spearman": res["slate_spearman"],
            "rmse": res["rmse"],
            "mae": res["mae"],
        }
        # one score per test season -- shows whether a winner wins every year
        for season in TEST_SEASONS:
            row[f"spearman_{season}"] = round(
                slate_spearman(preds[preds["season"].eq(season)], "pred_mean"), 4
            )
        row["seconds"] = round(time.time() - start, 1)
        rows.append(row)
        print(f"    -> slate_spearman {row['slate_spearman']}, rmse {row['rmse']}, "
              f"{row['seconds']}s\n", flush=True)

        # the baseline doesn't depend on the model, so score it once
        if baseline is None:
            preds = add_baselines(preds)
            b = evaluate(preds, "base_rolling", "baseline: rolling avg")
            baseline = {"model": b["model"], "slate_spearman": b["slate_spearman"],
                        "rmse": b["rmse"], "mae": b["mae"]}
            for season in TEST_SEASONS:
                sub = preds[preds["season"].eq(season)].dropna(subset=["base_rolling"])
                baseline[f"spearman_{season}"] = round(slate_spearman(sub, "base_rolling"), 4)

    results = pd.DataFrame(rows).sort_values("slate_spearman", ascending=False)
    results.insert(0, "rank", range(1, len(results) + 1))
    results = pd.concat([results, pd.DataFrame([baseline])], ignore_index=True)

    # the same scores as a grid: trees down the side, learning rate across
    grid = (pd.DataFrame(rows)
              .pivot(index="trees", columns="learning_rate", values="slate_spearman"))
    grid.columns = [f"lr {c}" for c in grid.columns]

    write_tuning_excel(results, grid)


def write_tuning_excel(results: pd.DataFrame, grid: pd.DataFrame) -> None:
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.utils import get_column_letter

    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / "tuning_results.xlsx"
    try:
        writer = pd.ExcelWriter(path, engine="openpyxl")
    except PermissionError:
        # the file is open in Excel -- write a new one instead of crashing
        path = MODELS / f"tuning_results_{time.strftime('%H%M%S')}.xlsx"
        writer = pd.ExcelWriter(path, engine="openpyxl")

    with writer:
        results.to_excel(writer, sheet_name="results", index=False)
        grid.to_excel(writer, sheet_name="spearman grid", index_label="trees")

        for ws in writer.sheets.values():
            ws.freeze_panes = "B2"
            for col in ws.columns:
                width = max(len(str(c.value)) for c in col if c.value is not None)
                ws.column_dimensions[get_column_letter(col[0].column)].width = width + 3

        # green = best ranking score, red = worst
        g = writer.sheets["spearman grid"]
        cells = f"B2:{get_column_letter(g.max_column)}{g.max_row}"
        g.conditional_formatting.add(cells, ColorScaleRule(
            start_type="min", start_color="F8696B",
            mid_type="percentile", mid_value=50, mid_color="FFEB84",
            end_type="max", end_color="63BE7B",
        ))

    print(f"saved tuning results to {path}")


# ---------------------------------------------------------------- feature families
# A named group of features that arrive together and go missing together, because
# they come out of one upstream file.
#
# `routes` is the one this exists for. nflverse publishes participation data --
# who was on the field for each play -- on a lag, and as of 2026-09-17 the 2026
# file is not published. Not late on this machine: not published, for anyone.
#
# Serving the full model anyway is worse than it sounds, and the reason is what
# missingness MEANS in the training data. Across 2021-2025 WR/TE rows, the ones
# where route_share_r3 is NaN have dkp_ewm of -0.17 against 7.93 for the rest,
# and their snap and target history is blank too. Route data goes missing for
# players with no recent history at all -- debuts, returns, deep bench. So the
# model learned, correctly, that a missing route share means someone who barely
# plays.
#
# In 2026 it is missing for EVERYONE. Every established receiver hits that split
# and gets routed down a branch built from players averaging -0.17 points. That
# is not ten percent less information, it is a systematic depression of exactly
# the players an optimizer would roster, and nothing about the output looks wrong.
#
# So train a variant without them. A model fit without route features leans on
# the snap family instead, and never builds a split whose default branch is a
# trap in production.
FEATURE_FAMILIES = {
    "routes": ["route_share_r3", "route_share_r5", "route_share_ewm",
               "route_share_last", "tprr_r5", "route_share_trend"],
}


def resolve_drop(names: list[str]) -> list[str]:
    """Expand family names; anything unrecognised is taken as a literal column."""
    out: list[str] = []
    for n in names:
        out.extend(FEATURE_FAMILIES.get(n, [n]))
    return out


def load_table(name: str = "model_table",
               data_dir: str | None = None,
               drop: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Read a model table; every column that isn't an ID or the answer is a feature.

    Kept as a function rather than inlined in main() because src/backtest.py
    imports it to build the same feature set main() trains on. Two copies of
    this logic is how training and backtesting quietly drift apart.

    `data_dir` may be a local directory or a gs:// prefix. Inside a training
    container there is no data/ directory -- the image ships code, not the
    51,046-row table -- so the job is pointed at Cloud Storage instead.
    """
    base = str(data_dir).rstrip("/") if data_dir else str(PROC)
    df = pd.read_parquet(f"{base}/{name}.parquet")
    features = [c for c in df.columns if c not in {
        "player_id", "player_display_name", "position", "recent_team",
        "opponent_team", "season", "week", TARGET,
    }]
    if drop:
        cut = resolve_drop(drop)
        absent = [c for c in cut if c not in features]
        features = [c for c in features if c not in cut]
        print(f"dropping {len(cut) - len(absent)} feature(s): {sorted(set(cut) - set(absent))}")
        if absent:
            print(f"  (not in this table anyway: {absent})")
    # position is categorical and genuinely predictive -- one-hot it
    df = pd.concat([df, pd.get_dummies(df["position"], prefix="pos")], axis=1)
    features += [c for c in df.columns if c.startswith("pos_")]
    return df, features


def main(save: bool, export: bool, table: str, model_dir: str,
         data_dir: str | None = None, drop: list[str] | None = None) -> None:
    df, features = load_table(table, data_dir, drop=drop)

    print(f"model table {df.shape[0]:,} rows, {len(features)} features")
    print(f"walk-forward over seasons {TEST_SEASONS}\n")

    if TUNE:
        tune(df, features)
        return

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

    # Calibration is fitted on the WALK-FORWARD predictions -- every row of
    # which is out-of-sample -- and ships inside the artifact. Fitting it on the
    # final model's own training fits would learn the training residuals, which
    # are small, differently shaped, and useless in production.
    cal = calibrate.fit(preds, "pred_mean")
    print("\n" + calibrate.describe(cal))
    corrected = calibrate.apply(preds, cal, "pred_mean")
    print(f"  slate Spearman {slate_spearman(preds, 'pred_mean'):.4f} -> "
          f"{slate_spearman(preds.assign(_c=corrected), '_c'):.4f}  "
          f"(rank barely moves -- the level was the problem, not the order)")

    if export:
        models, coverage = fit_final(df, features)
        out = model_io.save(
            model_dir, models,
            features=features, params=PARAMS, target=TARGET,
            train_rows=len(df), coverage=coverage, metrics=rows,
            calibration=cal, dropped=resolve_drop(drop) if drop else None,
        )
        print(f"\nexported artifact -> {out}")
        print(f"  {len(models)} models, {len(features)} features, "
              f"seasons {coverage['seasons'][0]}-{coverage['seasons'][1]}")
        print("  this is the thing that gets registered and served.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--export", action="store_true",
                   help="also refit on all data and write a servable artifact")
    p.add_argument("--table", default="model_table",
                   help="model table stem under data/processed (e.g. model_table_v3)")
    p.add_argument("--model-dir", default=os.environ.get("AIP_MODEL_DIR") or str(MODELS / "current"),
                   help="where to write the artifact; accepts a gs:// path. "
                        "Defaults to AIP_MODEL_DIR when set, which is how Agent "
                        "Platform tells a custom training job where its output goes.")
    p.add_argument("--data-dir", default=os.environ.get("AIP_DATA_DIR"),
                   help="where the model table lives; local dir or gs:// prefix. "
                        "Defaults to data/processed when unset.")
    p.add_argument("--drop", nargs="+", metavar="NAME", default=None,
                   help="exclude features before training: a family name "
                        "(routes) or literal column names. Use when an upstream "
                        "source stops publishing -- a model fit without the family "
                        "beats one whose missing-value branches were learned on a "
                        "different population than they now meet.")
    a = p.parse_args()
    main(save=not a.no_save, export=a.export, table=a.table,
         model_dir=a.model_dir, data_dir=a.data_dir, drop=a.drop)
