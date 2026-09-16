"""Bake-off: one pooled model vs. one model per position vs. a blend of both.

THE ARGUMENT THIS SETTLES
-------------------------
The current model pools every skill player into one XGBoost fit and hands it
`pos_QB` / `pos_RB` / ... one-hots. The case for splitting it into four models:

  1. A THIRD OF THE FEATURE SPACE IS DEAD for any given player. 24 of the 79
     features are receiving-only, 8 are rushing-only. They are not NaN -- they
     are ZEROS -- so they survive every null check and still get drawn by
     `colsample_bytree=0.8`, diluting every tree. For a QB, roughly 24 of the
     63 features each tree sees are constants.

  2. SOME OF IT IS WORSE THAN DEAD. QB `route_share_ewm` averages 0.875, not
     0.0 -- QBs are on the field for most pass plays, so the column is garbage
     WITH STRUCTURE. A tree can split on it and learn something that means
     nothing.

  3. IT UNLOCKS THE NGS DATA. `data/raw/ngs/` has separation, cushion, aDOT,
     rush-yards-over-expected, time-to-throw. All of it is position-exclusive
     by construction and cannot go in a pooled table without padding zeros for
     three quarters of the rows.

The case against splitting:

  4. IT SUBTRACTS DATA, IT DOES NOT ADD ANY. The pooled model already trains on
     all 20,824 WR rows. A WR-only model sees the same 20,824 -- it just gives
     up the 30,222 non-WR rows it was borrowing shared structure from. WR can
     afford that. QB, at 6,258 rows, might not.

  5. CALIBRATION. Four independent models each fit their own scale, and nothing
     forces them to agree with each other. The optimizer compares a TE's number
     against a WR's number at the FLEX slot and against every position under
     the salary cap. A TE model running 1.5 points hot over-drafts TEs every
     week -- and WITHIN-POSITION RANKING WOULD LOOK PERFECT THE WHOLE TIME,
     because adding a constant to every TE does not change their order.

So this script scores two things, not one. Spearman answers "is the order
right". Mean residual answers "is the number right". Only the second one can
see failure mode 5, which is why it is printed next to every architecture.

THE THREE ARCHITECTURES
-----------------------
  A. pooled       -- one model, all rows, all features + position one-hots
  B. per-position -- four models, each on its own rows and its own features
  C. blend        -- w * B + (1-w) * A, with w fit PER POSITION on held-out data

C is not splitting the difference. The pooled model is the anchor that holds
the common scale (fixing 5), the position model adds the specialisation
(harvesting 1-3), and `w` is not a guess -- it is fitted, so the data says
where specialisation actually helps. A WR weight near 0.8 and a QB weight near
0.3 would be the whole argument answered in two numbers.

DELIBERATELY HELD FIXED: hyperparameters. Every architecture uses the same
PARAMS. A per-position model probably wants different ones -- `min_child_weight
=20` is far more restrictive on 6,258 QB rows than on 51,046 pooled rows -- but
tuning per position in the same experiment that changes the architecture would
confound the two. Settle the architecture first, then tune the winner.

NOT INCLUDED: DST. There are zero DST rows in the model table; defenses need a
team-level table that does not exist yet. See the note printed at the end.

Run:
  python research/test_position_models.py
  python research/test_position_models.py --table model_table_v3 --seasons 2021 2022 2023 2024 2025
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from train import PARAMS, TARGET, slate_spearman  # noqa: E402

warnings.filterwarnings("ignore")

POSITIONS = ["QB", "RB", "WR", "TE"]
ID_COLS = {"player_id", "player_display_name", "position", "recent_team",
           "opponent_team", "season", "week", TARGET}

# ---------------------------------------------------------------------------
# Which features belong to which position.
#
# Hand-mapped on purpose rather than inferred from variance. A variance rule
# would drop the clean zeros (QB `tgt_share_ewm` = 0.000) but KEEP the harmful
# one (QB `route_share_ewm` = 0.875, which varies), and the whole point is to
# remove columns that are meaningless for the position, not merely constant.
RECEIVING = {
    "tgt_share_r3", "tgt_share_r5", "tgt_share_ewm", "ay_share_r5", "wopr_r5",
    "targets_r3", "targets_r5", "rec_r5", "targets_per_snap_r5",
    "route_share_r3", "route_share_r5", "route_share_ewm", "route_share_last",
    "tprr_r5", "route_share_trend", "rz_tgt_share_r5", "ez_tgt_share_r5",
    "rz_tgt_r5", "healthy_tgt_rank", "vacated_tgt_share", "new_vacated_tgt_share",
    "ps_targets_pg", "ps_tgt_share", "ps_ay_share",
}
RUSHING = {
    "carries_r3", "carries_r5", "i10_carry_share_r5", "i5_carry_share_r5",
    "i10_carries_r5", "vacated_carry_share", "ps_carries_pg", "spread_x_rb",
}

# What each position actually does. QBs rush (carries_r5 averages 3.35 for them
# and QB rushing is worth real DK points), so they keep the rushing block.
# RBs catch passes (tgt_share_ewm 0.075), so they keep both blocks.
DROP_FOR = {
    "QB": RECEIVING,
    "RB": set(),
    "WR": RUSHING,
    "TE": RUSHING,
}


def load(table: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(ROOT / "data" / "processed" / f"{table}.parquet")
    df = df[df["position"].isin(POSITIONS)].copy()
    feats = [c for c in df.columns if c not in ID_COLS]
    dummies = pd.get_dummies(df["position"], prefix="pos")
    df = pd.concat([df, dummies], axis=1)
    pooled_feats = feats + list(dummies.columns)
    return df, pooled_feats


def position_features(base_feats: list[str], pos: str) -> list[str]:
    """Per-position feature list: drop the irrelevant block and the one-hots.

    The one-hots go because they are constant inside a single-position model --
    a column that never varies can never be split on, so it is pure overhead in
    every `colsample_bytree` draw.
    """
    drop = DROP_FOR[pos]
    return [f for f in base_feats if f not in drop and not f.startswith("pos_")]


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, feats: list[str]) -> np.ndarray:
    m = xgb.XGBRegressor(objective="reg:squarederror", **PARAMS)
    m.fit(tr[feats], tr[TARGET], verbose=False)
    return m.predict(te[feats])


# ---------------------------------------------------------------------- A + B
def walk_forward(df: pd.DataFrame, pooled_feats: list[str],
                 seasons: list[int]) -> pd.DataFrame:
    """Train on everything before season S, predict S, for both architectures."""
    out = []
    for season in seasons:
        tr, te = df[df["season"] < season], df[df["season"] == season].copy()
        if tr.empty or te.empty:
            continue

        # A: pooled -- every row, every feature
        te["pred_pooled"] = fit_predict(tr, te, pooled_feats)

        # B: per position -- its own rows, its own features
        te["pred_pos"] = np.nan
        for pos in POSITIONS:
            trp, tep = tr[tr["position"].eq(pos)], te[te["position"].eq(pos)]
            if trp.empty or tep.empty:
                continue
            feats = position_features(pooled_feats, pos)
            te.loc[tep.index, "pred_pos"] = fit_predict(trp, tep, feats)

        n_pos = {p: len(tr[tr["position"].eq(p)]) for p in POSITIONS}
        print(f"  {season}: pooled trained on {len(tr):,} | per-pos "
              + ", ".join(f"{p} {n:,}" for p, n in n_pos.items()), flush=True)
        out.append(te)

    return pd.concat(out, ignore_index=True)


# -------------------------------------------------------------------------- C
W_GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)


def _blend_score(sub: pd.DataFrame, w: float) -> float:
    b = w * sub["pred_pos"] + (1 - w) * sub["pred_pooled"]
    return slate_spearman(sub.assign(_b=b), "_b")


def add_blend(preds: pd.DataFrame, seasons: list[int]) -> tuple[pd.DataFrame, dict]:
    """Blend with w fitted per position on EARLIER held-out seasons only.

    The honest version. `w` for season S is chosen using out-of-sample
    predictions from test seasons before S -- never from S itself. The first
    test season has no prior held-out data, so it falls back to 0.5, which is
    an assumption, not a result, and is labelled as such in the output.
    """
    preds = preds.copy()
    preds["pred_blend"] = np.nan
    chosen: dict = {}

    for season in seasons:
        prior = preds[preds["season"] < season]
        for pos in POSITIONS:
            mask = preds["season"].eq(season) & preds["position"].eq(pos)
            if not mask.any():
                continue
            hist = prior[prior["position"].eq(pos)].dropna(
                subset=["pred_pos", "pred_pooled", TARGET])

            if len(hist) < 500:
                w = 0.5  # no evidence yet
            else:
                scores = {float(x): _blend_score(hist, x) for x in W_GRID}
                w = max(scores, key=scores.get)

            chosen.setdefault(pos, {})[season] = w
            preds.loc[mask, "pred_blend"] = (
                w * preds.loc[mask, "pred_pos"]
                + (1 - w) * preds.loc[mask, "pred_pooled"]
            )
    return preds, chosen


# --------------------------------------------------------------------- scoring
def score(preds: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """Within-position rank quality AND calibration, side by side.

    `spearman` is the order question. `bias` -- mean(predicted - actual) -- is
    the number question, and it is the only column here that can see a model
    running systematically hot or cold. A per-position model with a great
    Spearman and a +1.5 bias will quietly cost money every week.
    """
    rows = []
    cols = {"pooled": "pred_pooled", "per-position": "pred_pos", "blend": "pred_blend"}

    for pos in POSITIONS:
        sub = preds[preds["position"].eq(pos)]
        for label, col in cols.items():
            d = sub.dropna(subset=[col, TARGET])
            if d.empty:
                continue
            rows.append({
                "position": pos, "n": len(d), "model": label,
                "spearman": round(slate_spearman(d, col), 4),
                "bias": round(float((d[col] - d[TARGET]).mean()), 3),
                "mae": round(float((d[col] - d[TARGET]).abs().mean()), 3),
            })

    # cross-position: the whole slate ranked together, which is what the
    # optimizer is really doing when it spends the cap across positions.
    for label, col in cols.items():
        d = preds.dropna(subset=[col, TARGET])
        if d.empty:
            continue
        rows.append({
            "position": "ALL (cross-pos)", "n": len(d), "model": label,
            "spearman": round(slate_spearman(d, col), 4),
            "bias": round(float((d[col] - d[TARGET]).mean()), 3),
            "mae": round(float((d[col] - d[TARGET]).abs().mean()), 3),
        })

    return pd.DataFrame(rows)


def blend_curve(preds: pd.DataFrame) -> pd.DataFrame:
    """Spearman across the whole w range, per position -- a diagnostic, not a result.

    Scored on the test seasons themselves, so these numbers are optimistic and
    must not be quoted as performance. They are here to show the SHAPE: a flat
    curve means the architecture barely matters for that position; a curve
    peaking hard at w=1.0 means the position model genuinely dominates.
    """
    rows = []
    for pos in POSITIONS:
        sub = preds[preds["position"].eq(pos)].dropna(
            subset=["pred_pos", "pred_pooled", TARGET])
        if sub.empty:
            continue
        rows.append({"position": pos,
                     **{f"w={w}": round(_blend_score(sub, w), 4) for w in W_GRID}})
    return pd.DataFrame(rows).set_index("position")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="model_table_v3")
    ap.add_argument("--seasons", type=int, nargs="+",
                    default=[2021, 2022, 2023, 2024, 2025])
    a = ap.parse_args()

    df, pooled_feats = load(a.table)
    print(f"{a.table}: {len(df):,} rows, {len(pooled_feats)} pooled features")
    for pos in POSITIONS:
        f = position_features(pooled_feats, pos)
        print(f"  {pos}: {len(df[df.position.eq(pos)]):,} rows, {len(f)} features "
              f"({len(pooled_feats) - len(f)} dropped)")

    print(f"\nwalk-forward over {a.seasons}")
    preds = walk_forward(df, pooled_feats, a.seasons)
    preds, chosen = add_blend(preds, a.seasons)

    print("\n" + "=" * 78)
    print("HELD-OUT: rank quality (spearman) AND calibration (bias)")
    print("=" * 78)
    tbl = score(preds, a.seasons)
    print(tbl.pivot(index="position", columns="model",
                    values=["spearman", "bias"]).round(4).to_string())

    print("\nfitted blend weight per position (w=1 is pure position model)")
    print("  season 1 is 0.5 by fallback -- no prior held-out data, not a result")
    print(pd.DataFrame(chosen).T.to_string())

    print("\nblend curve -- IN-SAMPLE ON TEST, shape only, do not quote as performance")
    print(blend_curve(preds).to_string())

    out = ROOT / "research" / "position_model_preds.parquet"
    preds.to_parquet(out, index=False)
    print(f"\npredictions -> {out}")
    print("\nNOT TESTED HERE: DST. The model table has zero DST rows -- defenses")
    print("are team-level, not player-level, and need their own table. That is a")
    print("missing model, not a missing split, and it is still the biggest hole.")


if __name__ == "__main__":
    main()
