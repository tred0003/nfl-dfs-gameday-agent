"""Model artifacts: turn a fitted model into a file, and read it back safely.

Nothing in this project used to save a model. `train.py` fit models inside
`walk_forward()` and let them fall out of scope; `predict_week.py` refit from
scratch on every run. That is fine while the model is a research object. It
stops being fine the moment you want to do anything a production ML system
does, for three separate reasons:

1. YOU CANNOT REGISTER A LOCAL VARIABLE. Model Registry, batch prediction,
   endpoints and drift monitoring every one of them take a *stored artifact*
   as their input. There is nothing to upload until there is a file.

2. YOU CANNOT AUDIT WHAT YOU DID NOT KEEP. Refitting at predict time means the
   model that picked last week's lineup is gone. When a week goes badly the
   question is always "was the model wrong, or was the data wrong" -- and that
   is unanswerable without the exact artifact that produced the projection.

3. TRAINING/SERVING SKEW LIVES IN THE FEATURE LIST. Refitting also means the
   feature set is whatever columns happened to be in the DataFrame that day.
   A slate with no QB in it produces no `pos_QB` dummy; a new feature added to
   `features_extra.py` silently joins the model mid-season. The model card
   pins the exact feature *order* at training time, and `align()` forces every
   future slate into that order -- missing columns become NaN, which XGBoost
   handles natively, and surprise columns are dropped loudly rather than
   shifting every downstream column by one.

Format is XGBoost's native `.ubj`, not pickle. Pickle serialises the Python
object graph, so an artifact written by this laptop's xgboost/sklearn can fail
to unpickle inside a training container built on different versions. `.ubj` is
just the trees and is portable across both.
"""

from __future__ import annotations

import json
import platform
import subprocess
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import xgboost as xgb

CARD_NAME = "model_card.json"


# --------------------------------------------------------------- gs:// paths
# A custom training job on Agent Platform gets its output location as the
# AIP_MODEL_DIR environment variable, and that value is always a Cloud Storage
# URI -- the job's local disk is destroyed when the container exits, so an
# artifact written locally does not survive the run that produced it.
#
# xgboost's save_model/load_model only take local filesystem paths, so the
# pattern here is write-local-then-copy rather than streaming. That is also
# what the prebuilt training containers do.

def _is_gcs(path) -> bool:
    return str(path).startswith("gs://")


def _split_uri(uri: str) -> tuple[str, str]:
    bucket, _, prefix = str(uri)[len("gs://"):].partition("/")
    return bucket, prefix.strip("/")


def _upload_dir(local: Path, uri: str) -> None:
    from google.cloud import storage

    bucket_name, prefix = _split_uri(uri)
    bucket = storage.Client().bucket(bucket_name)
    for f in sorted(local.iterdir()):
        if f.is_file():
            bucket.blob(f"{prefix}/{f.name}" if prefix else f.name).upload_from_filename(f)


def _download_dir(uri: str, local: Path) -> None:
    from google.cloud import storage

    bucket_name, prefix = _split_uri(uri)
    client = storage.Client()
    local.mkdir(parents=True, exist_ok=True)
    found = False
    for blob in client.list_blobs(bucket_name, prefix=prefix):
        name = blob.name.rsplit("/", 1)[-1]
        if not name:
            continue
        blob.download_to_filename(local / name)
        found = True
    if not found:
        raise FileNotFoundError(f"nothing under {uri}")


def _git_sha(root: Path) -> str | None:
    """Best-effort commit hash, so an artifact points back at the code that made it."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def save(
    out_dir: Path | str,  # local path, or gs://bucket/prefix
    models: dict[str, xgb.XGBRegressor],
    features: list[str],
    params: dict,
    target: str,
    train_rows: int,
    coverage: dict | None = None,
    metrics: list[dict] | None = None,
    calibration: dict | None = None,
    dropped: list[str] | None = None,
) -> Path:
    """Write the fitted models plus the card that describes them.

    `models` is keyed by role -- {"mean": ..., "ceiling": ...} -- because the
    two models answer different questions (expected points vs. ceiling) and
    downstream code selects by role, not by filename.
    """
    remote = str(out_dir) if _is_gcs(out_dir) else None
    local = Path(tempfile.mkdtemp(prefix="artifact_")) if remote else Path(out_dir)
    local.mkdir(parents=True, exist_ok=True)

    artifacts = {}
    for role, model in models.items():
        fname = f"model_{role}.ubj"
        model.save_model(local / fname)
        artifacts[role] = fname

    card = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(Path(__file__).resolve().parent.parent),
        "target": target,
        # ORDER MATTERS. This list is the contract with every future slate.
        "features": list(features),
        "n_features": len(features),
        "params": {k: v for k, v in params.items() if not callable(v)},
        "train_rows": int(train_rows),
        "coverage": coverage or {},
        "metrics": metrics or [],
        # The map that turns the raw model output into a number that means what
        # it says. Travels WITH the model because it is part of the model's
        # contract: an artifact served without its calibration overprojects a
        # lineup by ~7.4 points and nothing downstream can tell.
        "calibration": calibration or {},
        # Features deliberately excluded. `features` already implies this by
        # omission, but a human reading the registry needs it stated: two
        # artifacts with different feature counts are otherwise indistinguishable
        # at a glance, and the one you want depends on what upstream published.
        "dropped": list(dropped) if dropped else [],
        "artifacts": artifacts,
        "versions": {
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "pandas": pd.__version__,
        },
    }
    (local / CARD_NAME).write_text(json.dumps(card, indent=2))

    if remote:
        _upload_dir(local, remote)
        return remote
    return local


def load(model_dir: Path | str) -> tuple[dict[str, xgb.Booster], dict]:
    """Read the artifacts back. Returns ({role: model}, card)."""
    if _is_gcs(model_dir):
        staged = Path(tempfile.mkdtemp(prefix="artifact_"))
        _download_dir(str(model_dir), staged)
        model_dir = staged
    model_dir = Path(model_dir)
    card_path = model_dir / CARD_NAME
    if not card_path.exists():
        raise FileNotFoundError(
            f"no {CARD_NAME} in {model_dir}. Train and export first:\n"
            f"    python src/train.py --export"
        )
    card = json.loads(card_path.read_text())

    models = {}
    for role, fname in card["artifacts"].items():
        # The NATIVE Booster, not XGBRegressor. XGBRegressor is xgboost's
        # scikit-learn wrapper and constructing one imports sklearn -- which the
        # serving image does not carry, and should not: inference computes no
        # metrics. This failed a deployment on 2026-09-17 with
        # "ImportError: sklearn needs to be installed in order to use this
        # module", raised from the constructor before a single byte was read.
        #
        # Nothing is lost. .ubj stores trees -- splits, thresholds, leaf values
        # -- not a Python object graph, so the wrapper was never doing anything
        # at load time. Predictions are identical because training uses a fixed
        # n_estimators with no early stopping, so there is no best_iteration for
        # the wrapper to apply on our behalf.
        m = xgb.Booster()
        m.load_model(str(model_dir / fname))
        models[role] = m

    if xgb.__version__ != card["versions"]["xgboost"]:
        warnings.warn(
            f"artifact trained on xgboost {card['versions']['xgboost']}, "
            f"loading under {xgb.__version__}. .ubj is forward-compatible, but "
            f"note the mismatch if predictions look wrong."
        )
    return models, card


def align(df: pd.DataFrame, card: dict, *, label: str = "slate",
          info: dict | None = None) -> pd.DataFrame:
    """Force a live table into the exact feature order the model was trained on.

    This is the single function standing between you and training/serving skew.
    Passing raw `df[some_columns]` to `.predict()` works right up until the
    column set drifts by one -- and XGBoost will not complain, it will just
    read feature 12 out of the slot where feature 11 used to be and return
    confident nonsense.

    WHY `info` EXISTS: A PRINT IS NOT A RETURN VALUE

    This function has always known when a request arrived half-empty. It said so
    on stdout and then handed back a frame that looks exactly like a good one,
    so the caller had no way to act on it. Online that is survivable -- a human
    is reading the response. In a batch job nobody is reading anything: a
    malformed input file produces three hundred confident near-mean numbers, a
    JOB_STATE_SUCCEEDED, and no red anywhere. The count has to travel with the
    predictions, so `info` is filled in for a caller that passes a dict.

    AND WHY IT IS COUNTED PER ROW, NOT JUST PER COLUMN

    `missing` below is a COLUMN-level check, and it is blind to the failure that
    actually happens in batch. `pd.DataFrame(instances)` unions the keys of
    every instance, so one fully-populated row is enough to create a column --
    and the rows that carry nothing for it get NaN in a column that is, as far
    as this check can tell, present. `missing` reports zero and 90% of the slate
    is empty. `present_per_row` is the number that survives that: it counts, for
    each row, how many of the pinned features actually hold a value. It also
    subsumes the column case, because a column absent from the whole request is
    null on every row.
    """
    feats = card["features"]
    missing = [c for c in feats if c not in df.columns]
    extra = [c for c in df.columns if c not in feats]

    if missing:
        # NaN is a real value to XGBoost -- it learned a default direction for
        # every split -- so this degrades rather than crashes. Still worth saying.
        print(f"  align({label}): {len(missing)} feature(s) absent, filled NaN: "
              f"{missing[:8]}{' ...' if len(missing) > 8 else ''}")
    if extra:
        print(f"  align({label}): {len(extra)} column(s) not in the model, dropped")

    X = df.reindex(columns=feats)

    # FORCE THE DTYPES, DO NOT TRUST THEM. There are two ways a feature ends up
    # with no value, and pandas gives them different dtypes:
    #
    #   absent from the request   -> reindex creates it as float64 NaN.  Works.
    #   present as JSON null      -> pd.DataFrame builds an OBJECT column of
    #                                None.  xgb.DMatrix rejects object dtype
    #                                outright: "DataFrame.dtypes for data must
    #                                be int, float, bool or category."
    #
    # Which is backwards from every intuition -- sending nothing is safe and
    # sending an honest null is fatal. It went unnoticed through 4.2 and 4.3
    # because the hand-written test payload had no nulls in it. Every line of a
    # real batch file does, so the first real batch job would have failed on it
    # at $node-minutes with the reason buried in Cloud Logging.
    #
    # astype rather than pd.to_numeric(errors="coerce") on purpose: None becomes
    # NaN either way, but a genuine string in a numeric feature SHOULD raise
    # here. Coercing it would turn a caller's bad data into a confident number.
    X = X.astype("float64")

    if info is not None:
        info.update({
            "n_features": len(feats),
            "absent": len(missing),
            "absent_names": missing,
            "dropped": len(extra),
            # int64 Series indexed like df. The caller decides what a bad number
            # is -- there is no threshold here on purpose. "How many features is
            # too few" depends on which features, and inventing a cutoff would
            # replace a real number with a guess dressed up as a decision.
            "present_per_row": X.notna().sum(axis=1),
        })

    return X


def predict(df: pd.DataFrame, models: dict, card: dict, *, label: str = "slate",
            info: dict | None = None) -> pd.DataFrame:
    """Aligned prediction for every role in the artifact. Returns a frame of columns.

    Pass `info` as an empty dict to get the feature-coverage diagnostics back
    alongside the predictions -- see align().
    """
    X = align(df, card, label=label, info=info)
    # The Booster API predicts on a DMatrix, not a DataFrame. Built from the
    # aligned frame so the DMatrix carries the column names, which buys a second
    # skew guard for free: the booster stores the feature names it was fit on
    # and RAISES on a mismatch. align() prevents the shift; this catches it if
    # align() is ever bypassed. Missing values stay NaN, which is what the model
    # was trained to read.
    dmat = xgb.DMatrix(X)
    return pd.DataFrame(
        {f"pred_{role}": m.predict(dmat) for role, m in models.items()},
        index=df.index,
    )
