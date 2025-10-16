import os
from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from mlops_diabetes.data import FEATURE_ORDER
from mlops_diabetes.version import APP_VERSION
from .schemas import PredictRequest, PredictResponse


MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model_v0_1.joblib"))

app = FastAPI(title="Diabetes Progression API", version=APP_VERSION)

_model = None
_model_meta = {}


def load_model():
    global _model, _model_meta
    if not MODEL_PATH.exists():
        # Defer loading if artifact missing (e.g., in CI before training)
        return False
    payload = joblib.load(MODEL_PATH)
    _model = payload["model"]
    _model_meta = {
        "feature_order": payload.get("feature_order"),
        "model_version": payload.get("model_version", "unknown"),
    }
    return True


@app.on_event("startup")
async def _startup():
    # Attempt to load if available; tolerate missing artifact
    load_model()


@app.get("/health")
async def health():
    return {"status": "ok", "model_version": _model_meta.get("model_version")}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    if _model is None:
        # Try lazy load once
        if not load_model():
            raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        features: List[float] = [getattr(req, f) for f in FEATURE_ORDER]
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        pred = float(_model.predict([features])[0])
        return PredictResponse(prediction=pred)
    except Exception as e:  # observability for bad input
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")
