"""What is the model actually using -- globally, and per position.

`train.py` prints ONE importance list: `feature_importances_` from the pooled
model, averaged across walk-forward seasons. That list has two problems, and
this script exists to fix both.

PROBLEM 1: IT CANNOT BE SPLIT BY POSITION.
`feature_importances_` is computed from tree structure -- the total gain each
feature accumulated across every split in all 600 trees. It is a property of
the model as a whole. The trees are shared across positions, so there is no
"WR share" of a split to extract. One fitted model, one list, forever.

The fix is SHAP, which attributes per PREDICTION instead of per model: for each
player-week it returns how many points each feature added or subtracted from
that specific projection. Those rows carry a position, so they can be grouped.
The per-position view comes out of the POOLED model -- it never needed separate
models, it needed a better attribution method.

This adds no dependency. XGBoost computes exact tree SHAP natively via
`pred_contribs=True`; the `shap` package is only needed for its plots.

PROBLEM 2: CORRELATED FEATURES SPLIT THEIR OWN CREDIT.
This table has dkp_r3, dkp_r5, dkp_r8, dkp_ewm, dkp_lag1 and dkp_blended --
six measurements of "how has he been scoring lately", all highly correlated.
When features are correlated, importance is divided among them semi-arbitrarily
based on which one the tree happened to split on first. `dkp_ewm` topping the
list at 29.84 does not mean the EWM window is special; delete it and `dkp_r5`
would absorb most of that number with barely any change in accuracy.

So the honest unit of analysis is the FAMILY, not the column. The rollup below
groups columns that measure the same concept at different windows and sums
them, which answers the question you actually have -- "how much of this model
is recent scoring?" -- instead of an accounting artifact.

A THIRD THING WORTH KNOWING: importance says what the model USES, not what is
TRUE. A feature ranks high if it is predictive, if it leaks, or if it proxies
for something unmeasured. Importance is a debugging tool, not evidence.

Run:
  python research/test_feature_importance.py
  python research/test_feature_importance.py --table model_table_v3 --top 20
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import feature_labels  # noqa: E402
from train import PARAMS, TARGET  # noqa: E402

warnings.filterwarnings("ignore")

POSITIONS = ["QB", "RB", "WR", "TE"]
ID_COLS = {"player_id", "player_display_name", "position", "recent_team",
           "opponent_team", "season", "week", TARGET}

# Columns that measure the same thing over a different window collapse to one
# family. Everything else is its own family, so nothing is silently merged.
WINDOW_SUFFIX = re.compile(r"_(r\d+|ewm|lag\d+|last|trend|pg)$")


def family(name: str) -> str:
    return WINDOW_SUFFIX.sub("", name)


def load(table: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(ROOT / "data" / "processed" / f"{table}.parquet")
    df = df[df["position"].isin(POSITIONS)].copy()
    feats = [c for c in df.columns if c not in ID_COLS]
    dummies = pd.get_dummies(df["position"], prefix="pos")
    df = pd.concat([df, dummies], axis=1)
    return df, feats + list(dummies.columns)


def walk_forward_shap(df: pd.DataFrame, feats: list[str],
                      seasons: list[int]) -> tuple[pd.DataFrame, pd.Series]:
    """Attribution on HELD-OUT rows only, same folds train.py validates on.

    SHAP on the training rows would describe how the model memorised its own
    data. Computing it on the predicted season instead describes how the model
    behaves on slates it has never seen, which is the only behaviour that
    matters on Sunday.
    """
    shap_parts, meta_parts, gains = [], [], []

    for season in seasons:
        tr, te = df[df["season"] < season], df[df["season"] == season]
        if tr.empty or te.empty:
            continue

        m = xgb.XGBRegressor(objective="reg:squarederror", **PARAMS)
        m.fit(tr[feats], tr[TARGET], verbose=False)
        gains.append(pd.Series(m.feature_importances_.astype(float), index=feats))

        booster = m.get_booster()
        # (n_rows, n_features + 1) -- the trailing column is the base value
        contribs = booster.predict(xgb.DMatrix(te[feats]), pred_contribs=True)
        shap_parts.append(pd.DataFrame(contribs[:, :-1], columns=feats))
        meta_parts.append(te[["position", "season", "week"]].reset_index(drop=True))
        print(f"  {season}: {len(te):,} held-out rows attributed", flush=True)

    shap = pd.concat(shap_parts, ignore_index=True)
    shap[["position", "season", "week"]] = pd.concat(meta_parts, ignore_index=True)
    return shap, pd.concat(gains, axis=1).mean(axis=1)


def mean_abs(shap: pd.DataFrame, feats: list[str],
             mask: pd.Series | None = None) -> pd.Series:
    """Mean |SHAP| -- average points of influence, in DK points.

    Absolute value on purpose: a feature that pushes some players up and others
    down is doing a lot of work, and signed averaging would cancel it to zero.
    """
    d = shap if mask is None else shap[mask]
    # cast to float64: SHAP contributions come back float32, and a float32 17.8
    # prints as 17.799999 once it is rounded and rendered
    return d[feats].abs().mean().astype("float64").sort_values(ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="model_table_v3")
    ap.add_argument("--seasons", type=int, nargs="+",
                    default=[2021, 2022, 2023, 2024, 2025])
    ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args()

    df, feats = load(a.table)
    print(f"{a.table}: {len(df):,} rows, {len(feats)} features")
    print(f"attributing over held-out seasons {a.seasons}")
    shap, gain = walk_forward_shap(df, feats, a.seasons)

    # ---------------------------------------------------------------- global
    sh = mean_abs(shap, feats)
    cmp = pd.DataFrame({
        "shap_pts": sh.round(3),
        "shap_rank": sh.rank(ascending=False).astype(int),
        "gain_rank": gain.rank(ascending=False).astype(int),
    })
    # build from a dict of Series and pandas aligns on index, which lands
    # alphabetical -- sort back to the ranking this table exists to show
    # build from a dict of Series and pandas aligns on index, which lands
    # alphabetical -- sort back to the ranking this table exists to show
    cmp["moved"] = cmp["gain_rank"] - cmp["shap_rank"]
    cmp = cmp.sort_values("shap_rank")
    cmp.index = feature_labels.relabel(cmp.index)
    print("\n" + "=" * 78)
    print("GLOBAL -- SHAP vs gain importance")
    print("  shap_pts = average DK points this feature moves a projection")
    print("  moved    = how far it climbed when measured honestly (+) or fell (-)")
    print("=" * 78)
    print(cmp.head(a.top).to_string())

    # ---------------------------------------------------------------- family
    fam = sh.groupby(family).sum().sort_values(ascending=False)
    share = (fam / fam.sum() * 100).round(1)
    print("\n" + "=" * 78)
    print("BY FAMILY -- correlated columns summed into the concept they measure")
    print("  This is the number to quote. Six dkp_* columns splitting credit")
    print("  between themselves is an accounting artifact, not a finding.")
    print("=" * 78)
    fam_tbl = pd.DataFrame({"shap_pts": fam.round(3), "pct": share.round(1)}).head(a.top)
    fam_tbl.index = feature_labels.relabel(fam_tbl.index)
    print(fam_tbl.to_string())

    # ---------------------------------------------------------- per position
    print("\n" + "=" * 78)
    print(f"PER POSITION -- top {a.top} families by mean |SHAP|")
    print("=" * 78)
    per_pos = {}
    for pos in POSITIONS:
        s = mean_abs(shap, feats, shap["position"].eq(pos)).groupby(family).sum()
        per_pos[pos] = (s / s.sum() * 100).round(1)
    pos_tbl = (pd.DataFrame(per_pos).fillna(0)
                 .sort_values(POSITIONS[0], ascending=False).head(a.top).round(1))
    pos_tbl.index = feature_labels.relabel(pos_tbl.index)
    print(pos_tbl.to_string())

    print("\n  (columns are % of that position's total attribution, so they")
    print("   compare WITHIN a position -- read down a column, not across a row)")

    # ------------------------------------------------------------------ dead
    print("\n" + "=" * 78)
    print("DEAD WEIGHT -- features moving a position's projection < 0.01 pts")
    print("  The bake-off hand-mapped which features to drop per position.")
    print("  This measures it instead. Agreement validates that map; a feature")
    print("  dead here but kept there is one the hand-map missed.")
    print("=" * 78)
    for pos in POSITIONS:
        s = mean_abs(shap, feats, shap["position"].eq(pos))
        dead = s[s < 0.01]
        print(f"\n  {pos}: {len(dead)} of {len(feats)} features inert")
        for name in dead.index[:14]:
            print(f"    - {feature_labels.label(name)}")
        if len(dead) > 14:
            print(f"    ... and {len(dead) - 14} more")

    unlabeled = feature_labels.missing(feats)
    if unlabeled:
        print(f"\n  note: {len(unlabeled)} column(s) have no plain-English name yet "
              f"-- add them to src/feature_labels.py: {', '.join(unlabeled)}")

    out = ROOT / "research" / "shap_values.parquet"
    shap.to_parquet(out, index=False)
    print(f"\nper-row attributions -> {out}")


if __name__ == "__main__":
    main()
