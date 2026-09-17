"""Serving container for the projection model.

THE CONTRACT -- and it is NOT the training contract

A custom TRAINING container runs to completion and exits; the platform hands it
AIP_MODEL_DIR and reads whatever it wrote. A custom SERVING container is a
long-running HTTP server, and the platform talks to it over four environment
variables:

    AIP_HTTP_PORT      port to listen on            (default 8080)
    AIP_HEALTH_ROUTE   GET, must return 200 WHEN READY TO SERVE
    AIP_PREDICT_ROUTE  POST, {"instances": [...]} -> {"predictions": [...]}
    AIP_STORAGE_URI    Cloud Storage dir holding the model artifacts

Confusing the two is a standard exam trap. Nothing here applies to a training
job, and none of AIP_MODEL_DIR applies here.

The health route is subtler than it looks: it must return 200 only once the
model is actually loaded. Returning 200 immediately and loading lazily means the
platform routes live traffic to a container that is not ready, and the first
requests fail for no visible reason.

WHY NOT A PREBUILT CONTAINER

Google ships a prebuilt XGBoost serving image, and for a single model it is the
shorter path -- it looks for `model.bst` in the artifact directory and serves it.
This artifact cannot use it, for three reasons:

  1. TWO MODELS. `mean` answers "what will he score" (cash games); `ceiling` is
     the 85th percentile, "how high can he go" (tournaments). Callers need both.
  2. A CALIBRATION MAP. The raw model output overprojects the players an
     optimizer selects by ~7 points per lineup. The correction lives in the
     model card and has to be applied at inference time, or every downstream
     number is 5% hot. A prebuilt container knows nothing about it.
  3. THE FEATURE ORDER CONTRACT. The card pins the exact column order. XGBoost
     will not complain about a shifted column -- it reads position 12 where 11
     should be and returns confident nonsense. `model_io.align()` forces every
     request into the trained order, filling absent columns with NaN (which the
     model handles natively) and dropping unknown ones.

Point 3 is the entire reason this file exists rather than a prebuilt image. It
is where training/serving skew is actually prevented, in production, on every
request.

EVERY PREDICTION CARRIES ITS OWN FEATURE COUNT

align() degrades instead of crashing -- absent features become NaN, which the
model reads as a real value -- so an under-specified request comes back looking
exactly like a good one. Ask this server for a projection with three of the
eighty-three features and it answers with a confident number near the pool mean
and a 200. That is correct behaviour and a terrible thing to be unable to see.

So `features_present` ships on every prediction, beside the number it qualifies.
It is per-prediction rather than per-response on purpose: batch prediction pairs
predictions[i] with instances[i] and makes no promise about any other key in the
body, so a response-level summary is the one place the count could be dropped on
the way out. Riding on the prediction, it survives into the output JSONL and is
still there when the job is read back a week later.

Local test:
    AIP_STORAGE_URI=models/current python src/serve.py
    curl -X POST localhost:8080/predict -H 'Content-Type: application/json' \
         -d '{"instances": [{"dkp_ewm": 12.5, "salary": 6000, "position": "WR"}]}'
"""

from __future__ import annotations

import os
import traceback

import pandas as pd
from flask import Flask, jsonify, request

import calibrate
import model_io

app = Flask(__name__)

PORT = int(os.environ.get("AIP_HTTP_PORT", 8080))
HEALTH_ROUTE = os.environ.get("AIP_HEALTH_ROUTE", "/health")
PREDICT_ROUTE = os.environ.get("AIP_PREDICT_ROUTE", "/predict")
STORAGE_URI = os.environ.get("AIP_STORAGE_URI", "models/current")

MODELS: dict = {}
CARD: dict = {}
READY = False


def load() -> None:
    """Load once at startup. Health stays down until this finishes."""
    global MODELS, CARD, READY
    MODELS, CARD = model_io.load(STORAGE_URI)
    READY = True
    cal = CARD.get("calibration") or {}
    print(f"loaded {len(MODELS)} model(s) from {STORAGE_URI}: {list(MODELS)}", flush=True)
    print(f"  {CARD['n_features']} features pinned, git_sha {CARD.get('git_sha')}", flush=True)
    print(f"  calibration: {'yes, ' + str(cal.get('n', 0)) + ' rows' if cal else 'NONE'}",
          flush=True)


@app.route(HEALTH_ROUTE, methods=["GET"])
def health():
    # 200 only when the model is actually in memory. Returning 200 early makes
    # the platform send real traffic to a container that cannot answer.
    if not READY:
        return jsonify({"status": "loading"}), 503
    return jsonify({"status": "ok", "models": list(MODELS),
                    "features": CARD["n_features"]}), 200


@app.route(PREDICT_ROUTE, methods=["POST"])
def predict():
    """{"instances": [{feature: value, ...}, ...]} -> {"predictions": [...]}

    Each prediction carries both models' output, the raw uncalibrated mean so a
    caller can see what the correction did rather than trust it, and
    `features_present` / `features_expected` so a caller can see how much of the
    model's input actually arrived.
    """
    if not READY:
        return jsonify({"error": "model still loading"}), 503
    try:
        body = request.get_json(force=True)
        instances = body.get("instances")
        if not isinstance(instances, list) or not instances:
            return jsonify({"error": "body must be {\"instances\": [ {...}, ... ]}"}), 400

        df = pd.DataFrame(instances)

        # Rebuild the position one-hots exactly as training does. Training runs
        # pd.get_dummies(position) and the model learned pos_QB/pos_RB/pos_WR/
        # pos_TE; a caller naturally sends position="QB" instead. Without this
        # all four arrive NaN and the model loses the position gate entirely --
        # train/serve skew in the one place it is least excusable, and align()
        # cannot catch it because the columns are legitimately absent.
        #
        # Built from the CARD's feature list rather than from whatever positions
        # happen to be in this request: a slate with no quarterback must still
        # produce a pos_QB column of zeros, or the column set shifts.
        if "position" in df.columns:
            for col in (f for f in CARD["features"] if f.startswith("pos_")):
                df[col] = (df["position"].astype(str) == col[len("pos_"):]).astype(float)

        # THE SKEW GUARD. Reindex to the trained column order before the model
        # sees anything. Missing -> NaN (a real value to XGBoost); unknown ->
        # dropped. Without this a caller sending columns in a different order
        # gets confident garbage and no error.
        info: dict = {}
        preds = model_io.predict(df, MODELS, CARD, label="request", info=info)

        out = pd.DataFrame({"mean_raw": preds["pred_mean"]})
        if "pred_ceiling" in preds:
            out["ceiling"] = preds["pred_ceiling"]

        # Counted AFTER the dummy rebuild above, so a caller who sent
        # position="QB" is credited with the four pos_ columns it produced
        # rather than penalised for not having sent them by hand.
        #
        # Both numbers, not a ratio: 6 of 83 is a sentence a human reads at a
        # glance, and it stays readable if the feature count ever changes.
        out["features_present"] = info["present_per_row"].values
        out["features_expected"] = info["n_features"]

        cal = CARD.get("calibration") or {}
        if cal and "salary" in df.columns and "position" in df.columns:
            src = df.assign(_p=preds["pred_mean"].values)
            out["mean"] = calibrate.apply(src, cal, "_p").values
            out["calibrated"] = True
        else:
            out["mean"] = preds["pred_mean"]
            out["calibrated"] = False

        cols = ["mean", "ceiling", "mean_raw", "calibrated",
                "features_present", "features_expected"]
        return jsonify({"predictions": out[[c for c in cols if c in out]]
                        .to_dict(orient="records")}), 200

    except Exception as e:  # a 500 with a traceback beats a silent hang
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


load()

if __name__ == "__main__":
    # Flask's dev server is single-threaded and not for production; gunicorn is
    # the entrypoint in the image. This path is for local testing only.
    app.run(host="0.0.0.0", port=PORT)
