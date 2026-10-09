"""Walk-forward training for the sequence model, graded against XGBoost.

Grading is deliberately identical to src/train.py -- same folds, same metric,
same baselines -- because the only question this project is asking is whether
a network that reads raw games beats a gradient-boosted model that reads
hand-built rolling averages. Any difference in how the two are scored would
make the answer meaningless.

  folds    train on seasons < S, predict S, for S in 2021..2024
  metric   mean within-week Spearman rank correlation. The optimizer consumes
           the ORDER of the projections, never the numbers, so ranking within
           a slate is the decision actually being made. RMSE rides along as a
           secondary check.
  floor    the player's own rolling average. A model that cannot beat it has
           learned nothing worth shipping.

THE EPOCH-COUNT PROBLEM, AND WHY THERE ARE TWO FITS PER FOLD
------------------------------------------------------------
XGBoost trains on every season before S. A network needs a validation set to
know when to stop, and taking one out of the training seasons would hand
XGBoost a data advantage that had nothing to do with either model's quality.

So each fold fits twice:

  1. train on seasons < S-1, validate on S-1, early-stop, record the best epoch
  2. throw that model away, rebuild, and fit on ALL seasons < S for exactly
     that many epochs

Pass 1 only ever decides a number: how long to train. Pass 2 is the model that
gets graded, and it saw precisely the data XGBoost saw. Season S is untouched
by both passes.

Run:  .venv-tf/Scripts/python tf_sequence/train_tf.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import spearmanr
from tensorflow import keras

from common import OUT, PROC, TARGET, TEST_SEASONS, TFR
from model import adapt_arrays, build_model
from pipeline import load_spec, make_dataset, steps_for

MODEL_INPUTS = ("seq", "seq_mask", "static", "pos")


def strip(features, label):
    """Keep only what the model declares as an input.

    The records also carry season and week so predictions can be grouped into
    slates at grading time. Keras 3 rejects a dict input containing keys the
    model never declared, so they are dropped here and recovered separately.
    """
    return {k: features[k] for k in MODEL_INPUTS}, label


def predict(model, ds):
    """model.predict, minus one alarming and entirely false warning.

    Keras 3 reuses its training EpochIterator for predict(), so when a finite
    dataset ends it emits "Your input ran out of data; interrupting training."
    during inference, where there is no training to interrupt and nothing is
    dropped -- verified: predict returns exactly spec['counts'][season] rows.

    Suppressed narrowly, by message, around this one call. Silencing it
    globally would also hide the real version of this warning, which fires
    during fit() and means an epoch was genuinely truncated. That one is a
    correctness bug and must stay loud. The caller asserts the row count
    regardless, so this is cosmetic on top of a real check.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*ran out of data.*",
                                category=UserWarning)
        return model.predict(ds.map(strip), verbose=0)


def slate_spearman(df: pd.DataFrame, pred_col: str, target: str = TARGET) -> float:
    """Mean within-week rank correlation. Copied in spirit from src/train.py.

    Pooling every week into one correlation would reward a model for knowing
    that good players outscore bad ones -- true, and useless, because you pick
    a lineup from one slate at a time.
    """
    rhos = []
    for _, wk in df.groupby(["season", "week"]):
        if len(wk) < 10:
            continue
        rho = spearmanr(wk[pred_col], wk[target]).statistic
        if np.isfinite(rho):
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else float("nan")


def collect(ds) -> pd.DataFrame:
    """Pull season/week/pos/label out of an un-shuffled dataset, in order."""
    rows = {"season": [], "week": [], "pos": [], TARGET: []}
    for feat, label in ds:
        rows["season"].append(feat["season"].numpy().reshape(-1))
        rows["week"].append(feat["week"].numpy().reshape(-1))
        rows["pos"].append(feat["pos"].numpy().reshape(-1))
        rows[TARGET].append(label.numpy().reshape(-1))
    return pd.DataFrame({k: np.concatenate(v) for k, v in rows.items()})


def fit_once(spec, train_seasons, val_seasons, epochs, batch_size, aug_kwargs,
             patience, tfr_dir, seed, verbose):
    """One fit. Returns (model, history). val_seasons=None means no early stop."""
    train_raw = make_dataset(train_seasons, spec, batch_size=batch_size,
                             training=True, tfr_dir=tfr_dir,
                             aug_kwargs=aug_kwargs, repeat=True, seed=seed)
    seq_adapt, static_adapt = adapt_arrays(
        make_dataset(train_seasons, spec, batch_size=batch_size, training=False,
                     tfr_dir=tfr_dir, augment=False)
    )
    keras.utils.set_random_seed(seed)
    model = build_model(spec, seq_adapt, static_adapt)

    # Explicit, because the dataset repeats forever now. See pipeline.steps_for
    # for why guessing was worse than it looked.
    kw = {"steps_per_epoch": steps_for(train_seasons, spec, batch_size)}
    if val_seasons:
        kw["validation_data"] = make_dataset(
            val_seasons, spec, batch_size=batch_size, training=False,
            tfr_dir=tfr_dir, augment=False).map(strip)
        kw["validation_steps"] = steps_for(val_seasons, spec, batch_size)
        kw["callbacks"] = [keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True)]

    hist = model.fit(train_raw.map(strip), epochs=epochs, verbose=verbose,
                     shuffle=False, **kw)
    return model, hist


def main(epochs: int, batch_size: int, patience: int, tfr_dir: Path,
         seed: int, save: bool, verbose: int,
         xgb_spearman: float | None = None, xgb_rmse: float | None = None) -> None:
    spec = load_spec(tfr_dir)
    all_seasons = sorted(int(s) for s in spec["counts"])
    print(f"spec: seq {spec['seq_len']}x{spec['n_seq']}, static {spec['static_width']}, "
          f"seasons {all_seasons[0]}-{all_seasons[-1]}")

    aug_kwargs = {"drop": 2, "jitter": 0.05, "crop": 2, "extra_cost": 0}
    print(f"augmentation (applied fresh every epoch, inside tf.data): {aug_kwargs}\n")

    folds = []
    for S in TEST_SEASONS:
        train_seasons = [s for s in all_seasons if s < S]
        if len(train_seasons) < 3:
            continue
        inner, val = train_seasons[:-1], [train_seasons[-1]]
        t0 = time.time()

        print(f"season {S}: train {train_seasons[0]}-{train_seasons[-1]} "
              f"({sum(spec['counts'][str(s)] for s in train_seasons):,} rows)")

        # pass 1 -- decide the epoch count only
        _, hist = fit_once(spec, inner, val, epochs, batch_size, aug_kwargs,
                           patience, tfr_dir, seed, verbose)
        best = int(np.argmin(hist.history["val_loss"])) + 1
        print(f"  pass 1: validated on {val[0]}, best epoch {best}/{len(hist.history['val_loss'])}")

        # pass 2 -- the model that gets graded, on the same data XGBoost saw
        model, _ = fit_once(spec, train_seasons, None, best, batch_size,
                            aug_kwargs, patience, tfr_dir, seed, verbose)

        test = make_dataset([S], spec, batch_size=batch_size, training=False,
                            tfr_dir=tfr_dir, augment=False)
        meta = collect(test)
        meta["pred"] = predict(model, test).reshape(-1)
        meta["fold"] = S
        # Every prediction must line up with the row it came from, or the
        # metric is measuring a shuffle. Cheap to assert, expensive to miss.
        assert len(meta) == spec["counts"][str(S)], (
            f"{len(meta)} predictions vs {spec['counts'][str(S)]} records")
        folds.append(meta)

        print(f"  pass 2: {best} epochs on {len(train_seasons)} seasons  "
              f"-> slate Spearman {slate_spearman(meta, 'pred'):.4f}  "
              f"({time.time() - t0:.0f}s)\n")

    preds = pd.concat(folds, ignore_index=True)

    # ---------------------------------------------------------------- grading
    print("=" * 74)
    print(f"HELD-OUT PERFORMANCE ({TEST_SEASONS[0]}-{TEST_SEASONS[-1]}, never trained on)")
    print("=" * 74)

    rmse = float(np.sqrt(np.mean((preds["pred"] - preds[TARGET]) ** 2)))
    rows = [{"model": "TF sequence (GRU)", "n": len(preds),
             "slate_spearman": round(slate_spearman(preds, "pred"), 4),
             "rmse": round(rmse, 3)}]

    # the floor, recomputed here rather than trusted from another file
    raw = pd.read_parquet(PROC / f"{spec['table']}.parquet")
    raw = raw[raw["season"].isin(TEST_SEASONS) & raw["position"].isin(spec["positions"])]
    if len(raw) != len(preds):
        print(f"  NOTE: parquet has {len(raw):,} rows for these seasons vs "
              f"{len(preds):,} predictions -- baselines are computed on the "
              f"parquet's own rows, not joined to the predictions.")
    base = raw.assign(base_rolling=raw["dkp_r5"].fillna(raw["dkp_r3"])).dropna(
        subset=["base_rolling"])
    rows.append({"model": "baseline: rolling avg", "n": len(base),
                 "slate_spearman": round(slate_spearman(base, "base_rolling"), 4),
                 "rmse": round(float(np.sqrt(np.mean(
                     (base["base_rolling"] - base[TARGET]) ** 2))), 3)})

    # XGBoost's numbers, IF they were produced on this same feature table.
    #
    # Read this before trusting the comparison. models/metrics.json does not
    # record which table produced it, and src/train.py defaults to
    # --table model_table while this model is built from model_table_v3. Those
    # are different feature sets: v3 carries 83 features including dkp_blended,
    # which alone is 26% of XGBoost's importance. Comparing against a stale
    # metrics.json understated XGBoost by 0.045 Spearman -- enough to turn a
    # loss into an apparent win, which is exactly what it did the first time
    # this script was run.
    #
    # So the number is not read automatically any more. Generate the matching
    # one deliberately and pass it in.
    if xgb_spearman is not None:
        rows.append({"model": f"XGBoost (mean) on {spec['table']}",
                     "n": len(preds), "slate_spearman": round(xgb_spearman, 4),
                     "rmse": xgb_rmse if xgb_rmse else float("nan")})
    else:
        print(f"  NO XGBOOST COLUMN. models/metrics.json does not record which\n"
              f"  table it came from, so it is not safe to compare against.\n"
              f"  For the honest number, run:\n"
              f"      python src/train.py --table {spec['table']} --no-save\n"
              f"  then re-run this with --xgb-spearman <value> --xgb-rmse <value>.\n")

    print(pd.DataFrame(rows).set_index("model").to_string())

    print("\nby fold (slate-rank correlation):")
    for S in sorted(preds["fold"].unique()):
        sub = preds[preds["fold"].eq(S)]
        print(f"  {S}: {slate_spearman(sub, 'pred'):.4f}  (n={len(sub):,})")

    print("\nby position:")
    for i, p in enumerate(spec["positions"]):
        sub = preds[preds["pos"].eq(i)]
        if len(sub) > 100:
            print(f"  {p}: {slate_spearman(sub, 'pred'):.4f}  (n={len(sub):,})")

    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(OUT / "walk_forward_preds.parquet", index=False)
        (OUT / "metrics.json").write_text(json.dumps(rows, indent=2))
        print(f"\nsaved predictions + metrics to {OUT}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60, help="cap for pass 1")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--tfr-dir", default=str(TFR))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--xgb-spearman", type=float, default=None,
                   help="XGBoost's slate Spearman ON THIS SAME TABLE, from "
                        "`python src/train.py --table <table> --no-save`. "
                        "Omitted means no comparison row, on purpose.")
    p.add_argument("--xgb-rmse", type=float, default=None)
    a = p.parse_args()
    tf.get_logger().setLevel("ERROR")
    main(a.epochs, a.batch_size, a.patience, Path(a.tfr_dir), a.seed,
         not a.no_save, a.verbose, a.xgb_spearman, a.xgb_rmse)
