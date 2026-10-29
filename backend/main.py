"""FastAPI service for the saved cloud fault Random Forest model."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field


MODEL_PATH = Path(__file__).resolve().parent / "random_forest_model.pkl"
FEATURE_NAMES = ("total_logs", "unique_components")
model: Any | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    # Pickle/joblib files can execute code when loaded. Only load a trusted artifact.
    model = joblib.load(MODEL_PATH)
    expected = tuple(getattr(model, "feature_names_in_", ()))
    if expected and expected != FEATURE_NAMES:
        raise RuntimeError(
            f"Model features are {expected}; API expects {FEATURE_NAMES}. "
            "Update FEATURE_NAMES and the prediction schema to match the trained model."
        )
    yield
    model = None


app = FastAPI(
    title="Cloud Fault Prediction API",
    description="Random Forest inference for log-block features.",
    version="1.0.0",
    lifespan=lifespan,
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_logs: int = Field(ge=0, description="Number of log lines in the block")
    unique_components: int = Field(ge=0, description="Distinct components in the block")
    model: str = "random_forest"


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "Cloud Fault Prediction API", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, str]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ok", "model": "random_forest"}


@app.get("/metrics")
def metrics() -> dict[str, list[Any]]:
    # No held-out evaluation metrics are bundled with this model artifact.
    return {"models": []}


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    if payload.model not in {"random_forest", "random forest"}:
        raise HTTPException(
            status_code=422,
            detail="Only the random_forest model is available.",
        )

    started = time.perf_counter()
    features = np.asarray(
        [[payload.total_logs, payload.unique_components]], dtype=float
    )
    prediction = int(model.predict(features)[0])
    probabilities = getattr(model, "predict_proba", None)
    confidence = None
    if probabilities:
        class_probabilities = model.predict_proba(features)[0]
        classes = list(getattr(model, "classes_", range(len(class_probabilities))))
        if prediction in classes:
            confidence = float(class_probabilities[classes.index(prediction)])

    return {
        "prediction": prediction,
        "label": "Anomaly" if prediction == 1 else "Normal",
        "confidence": confidence,
        "model": "random_forest",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }
