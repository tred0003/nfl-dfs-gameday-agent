"""Recalibration: make the projected number mean what it says.

THE PROBLEM, measured on 71 backtested weeks

The model is very nearly unbiased across the whole player pool -- mean residual
+0.29 points. But the optimizer does not sample the pool at random. It solves for
the nine highest projections under a salary cap, which means it preferentially
selects players whose projection error happens to be most POSITIVE that week.
On the players it actually rosters the bias is +0.82 each, which is +7.4 on a
nine-man lineup. The lineup says 143 and delivers 136, every week.

That is the optimizer's curse, and the important thing about it is that it is
NOT an irreducible property of taking an argmax over noisy numbers.

The ILP sees only (prediction, salary). Selection is therefore a deterministic
function of those two. So by iterated expectation, if the model were
conditionally unbiased given (prediction, salary) -- if E[actual - pred | pred,
salary] = 0 -- the bias on the selected set would be exactly zero no matter what
the optimizer did. The curse IS conditional miscalibration. Remove the
miscalibration and the curse goes with it.

And the model is not conditionally unbiased. Fitting the error on the pool:

    err = 1.083 + 0.1205*pred - 0.4208*(salary/1000)      t(salary) = -5.7

Error grows with the prediction, and at a fixed prediction it grows as salary
falls. The ILP buys precisely that corner -- high projection, low price -- and
25% of its skill slots go to players under $5,000.

WHY THIS IS MOSTLY FREE

A purely AFFINE recalibration cannot change the optimizer's choice at all:
maximizing sum(a + b*p_i) over nine players is the same argmax as maximizing
sum(p_i) for any b > 0. So correcting the LEVEL costs nothing and changes
nothing except that the number becomes true. Only the CURVATURE -- the fact that
the correction differs by prediction size, salary and position -- moves lineups.
Measured end to end: overprojection +7.41 -> -0.52, and mean actual points went
UP 1.15 (p=0.41, i.e. free rather than an improvement).

WHY IT MATTERS EVEN THOUGH IT ADDS NO POINTS

Every decision that compares a projection to a threshold is currently fed a
number that is 5.5% hot on exactly the players you rostered: contest selection,
cash-versus-tournament, bankroll sizing, "is this slate worth entering", and the
swap costs on the deal board. Fix the input before building anything on it.

HONESTY ABOUT THE FIT

The map is fitted on held-out walk-forward predictions -- never on rows the
projection model trained on -- and applied to seasons after the ones it was
fitted on. Fitting it on in-sample predictions would learn the training
residuals, which are small and differently shaped, and would do nothing useful
in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

POSITIONS = ["QB", "RB", "WR", "TE"]


def _design(pred: np.ndarray, salary: np.ndarray, position: pd.Series) -> np.ndarray:
    """[1, pred, pred^2, salary/1000, position one-hots].

    pred^2 because the miscalibration grows with the prediction rather than
    shifting it; salary because at a fixed prediction the error still moves with
    price, which is the half a 1-D calibration on `pred` alone cannot see (an
    isotonic map on pred only got the overprojection to +4.59, not to zero).
    """
    sal = np.nan_to_num(salary, nan=float(np.nanmedian(salary))) / 1000.0
    cols = [np.ones_like(pred), pred, pred ** 2, sal]
    for p in POSITIONS[1:]:                      # QB is the reference level
        cols.append((position.values == p).astype(float))
    return np.column_stack(cols)


def fit(df: pd.DataFrame, pred_col: str, target_col: str = "dk_points",
        salary_col: str = "salary") -> dict:
    """Least-squares fit of residual (pred - actual) on the design above.

    `df` must be OUT-OF-SAMPLE predictions -- walk-forward output, not training
    fits.
    """
    d = df.dropna(subset=[pred_col, target_col])
    d = d[d["position"].isin(POSITIONS)]
    if len(d) < 500:
        raise ValueError(f"only {len(d)} usable rows; refusing to fit a calibration")

    X = _design(d[pred_col].to_numpy(float), d[salary_col].to_numpy(float), d["position"])
    y = (d[pred_col] - d[target_col]).to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    corrected = d[pred_col].to_numpy(float) - X @ beta
    return {
        "beta": [float(b) for b in beta],
        "terms": ["intercept", "pred", "pred^2", "salary_k"] + [f"pos_{p}" for p in POSITIONS[1:]],
        "n": int(len(d)),
        "bias_before": float(y.mean()),
        "bias_after": float((corrected - d[target_col]).mean()),
        "fitted_on": sorted(int(s) for s in d["season"].unique()),
    }


def apply(df: pd.DataFrame, cal: dict, pred_col: str,
          salary_col: str = "salary", out_col: str | None = None) -> pd.Series:
    """Apply a fitted map. Rows at other positions pass through uncorrected."""
    beta = np.asarray(cal["beta"], dtype=float)
    pred = df[pred_col].to_numpy(float)
    sal = df[salary_col].to_numpy(float) if salary_col in df.columns else np.full(len(df), np.nan)

    X = _design(pred, sal, df["position"])
    out = pred - X @ beta

    known = df["position"].isin(POSITIONS).to_numpy()
    out = np.where(known & np.isfinite(out), out, pred)
    # A projection below zero is not information, it is extrapolation past the
    # edge of the fit. DK floors at 0 for practical purposes.
    out = np.clip(out, 0.0, None)

    s = pd.Series(out, index=df.index, name=out_col or pred_col)
    return s


def describe(cal: dict) -> str:
    terms = ", ".join(f"{t} {b:+.4f}" for t, b in zip(cal["terms"], cal["beta"]))
    return (f"calibration fitted on {cal['n']:,} held-out rows "
            f"(seasons {cal['fitted_on']}):\n  {terms}\n"
            f"  pool bias {cal['bias_before']:+.3f} -> {cal['bias_after']:+.3f}")
