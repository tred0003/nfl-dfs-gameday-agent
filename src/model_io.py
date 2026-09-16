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
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import xgboost as xgb

CARD_NAME = "model_card.json"


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
    out_dir: Path | str,
    models: dict[str, xgb.XGBRegressor],
    features: list[str],
    params: dict,
    target: str,
    train_rows: int,
    coverage: dict | None = None,
    metrics: list[dict] | None = None,
) -> Path:
    """Write the fitted models plus the card that describes them.

    `models` is keyed by role -- {"mean": ..., "ceiling": ...} -- because the
    two models answer different questions (expected points vs. ceiling) and
    downstream code selects by role, not by filename.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {}
    for role, model in models.items():
        fname = f"model_{role}.ubj"
        model.save_model(out_dir / fname)
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
        "artifacts": artifacts,
        "versions": {
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "pandas": pd.__version__,
        },
    }
    (out_dir / CARD_NAME).write_text(json.dumps(card, indent=2))
    return out_dir


def load(model_dir: Path | str) -> tuple[dict[str, xgb.XGBRegressor], dict]:
    """Read the artifacts back. Returns ({role: model}, card)."""
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
        m = xgb.XGBRegressor()
        m.load_model(model_dir / fname)
        models[role] = m

    if xgb.__version__ != card["versions"]["xgboost"]:
        warnings.warn(
            f"artifact trained on xgboost {card['versions']['xgboost']}, "
            f"loading under {xgb.__version__}. .ubj is forward-compatible, but "
            f"note the mismatch if predictions look wrong."
        )
    return models, card


def align(df: pd.DataFrame, card: dict, *, label: str = "slate") -> pd.DataFrame:
    """Force a live table into the exact feature order the model was trained on.

    This is the single function standing between you and training/serving skew.
    Passing raw `df[some_columns]` to `.predict()` works right up until the
    column set drifts by one -- and XGBoost will not complain, it will just
    read feature 12 out of the slot where feature 11 used to be and return
    confident nonsense.
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

    return df.reindex(columns=feats)


def predict(df: pd.DataFrame, models: dict, card: dict, *, label: str = "slate") -> pd.DataFrame:
    """Aligned prediction for every role in the artifact. Returns a frame of columns."""
    X = align(df, card, label=label)
    return pd.DataFrame(
        {f"pred_{role}": m.predict(X) for role, m in models.items()},
        index=df.index,
    )
