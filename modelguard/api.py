from typing import Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from .core import FEATURES, active, append_feedback, monitor

app = FastAPI(title="ModelGuard API", version="1.0.0", description="Prediction, feedback, and model monitoring")


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/docs")


class PredictionRequest(BaseModel):
    features: Dict[str, float]


class FeedbackRequest(BaseModel):
    features: Dict[str, float]
    label: int = Field(ge=0, le=1)


def checked(values: dict[str, float]) -> list[float]:
    missing = sorted(set(FEATURES) - set(values))
    extra = sorted(set(values) - set(FEATURES))
    if missing or extra:
        raise HTTPException(422, detail={"missing_features": missing, "unknown_features": extra})
    row = [values[f] for f in FEATURES]
    if not np.isfinite(row).all():
        raise HTTPException(422, detail="Feature values must be finite numbers")
    return row


@app.get("/health")
def health():
    _, info = active()
    return {"status": "ok", "active_model": info}


@app.get("/schema")
def schema():
    return {"features": FEATURES, "labels": {0: "malignant", 1: "benign"}}


@app.post("/predict")
def predict(request: PredictionRequest):
    row = checked(request.features)
    model, info = active()
    probability = model.predict_proba([row])[0]
    prediction = int(np.argmax(probability))
    return {"prediction": prediction, "label": "benign" if prediction else "malignant",
            "confidence": float(probability[prediction]), "model_version": info["version"]}


@app.post("/feedback", status_code=202)
def feedback(request: FeedbackRequest):
    checked(request.features)
    append_feedback(request.features, request.label)
    return {"accepted": True, "message": "Labeled observation recorded"}


@app.post("/monitor")
def run_monitor(min_rows: int = 20):
    if min_rows < 4:
        raise HTTPException(422, detail="min_rows must be at least 4")
    return monitor(min_rows=min_rows)
