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

    Each prediction carries both models' output plus the raw uncalibrated mean,
    so a caller can see what the correction did rather than having to trust it.
    """
    if not READY:
        return jsonify({"error": "model still loading"}), 503
    try:
        body = request.get_json(force=True)
        instances = body.get("instances")
        if not isinstance(instances, list) or not instances:
            return jsonify({"error": "body must be {\"instances\": [ {...}, ... ]}"}), 400

        df = pd.DataFrame(instances)

        # THE SKEW GUARD. Reindex to the trained column order before the model
        # sees anything. Missing -> NaN (a real value to XGBoost); unknown ->
        # dropped. Without this a caller sending columns in a different order
        # gets confident garbage and no error.
        preds = model_io.predict(df, MODELS, CARD, label="request")

        out = pd.DataFrame({"mean_raw": preds["pred_mean"]})
        if "pred_ceiling" in preds:
            out["ceiling"] = preds["pred_ceiling"]

        cal = CARD.get("calibration") or {}
        if cal and "salary" in df.columns and "position" in df.columns:
            src = df.assign(_p=preds["pred_mean"].values)
            out["mean"] = calibrate.apply(src, cal, "_p").values
            out["calibrated"] = True
        else:
            out["mean"] = preds["pred_mean"]
            out["calibrated"] = False

        cols = ["mean", "ceiling", "mean_raw", "calibrated"]
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
