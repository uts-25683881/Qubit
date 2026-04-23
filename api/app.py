from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


# Make sure we can import training modules from project root.
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from train.stgcn_model import STGCNClassifier  # noqa: E402
from src.skeleton_utils import normalise_skeleton_sequence  # noqa: E402


MODEL_PATH = BASE_DIR / "models" / "stgcn_best.pth"
LABEL_INFO_PATH = BASE_DIR / "models" / "stgcn_label_info.pkl"
POSTURE_MODEL_PATH = BASE_DIR / "models" / "posture_classifier.pkl"
POSTURE_SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
POSTURE_LABEL_ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"
WEB_INDEX_PATH = BASE_DIR / "web" / "index.html"
WEB_TESTER_PATH = BASE_DIR / "web" / "tester.html"
UNKNOWN_THRESHOLD = 0.65

app = FastAPI(title="Qubit ST-GCN API", version="1.0.0")


class PredictRequest(BaseModel):
    # Expected shape: [T, V, C] = [24, 33, 3]
    window: List[List[List[float]]] = Field(..., description="Skeleton sequence [T, V, C]")


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probs: List[float]
    classes: List[str]


class PosturePredictRequest(BaseModel):
    # Single frame landmarks with shape [33, 4] = [x, y, z, visibility]
    frame: List[List[float]] = Field(..., description="Single-frame landmarks [33, 4]")


class PosturePredictResponse(BaseModel):
    label: str
    confidence: float
    probs: List[float]
    classes: List[str]


MODEL = None
CLASSES: List[str] = []
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POSTURE_MODEL = None
POSTURE_SCALER = None
POSTURE_LABEL_ENCODER = None
POSTURE_FEATURE_NAMES = [name for i in range(33) for name in (f"x{i}", f"y{i}", f"z{i}", f"v{i}")]


def load_artifacts() -> tuple[STGCNClassifier, List[str]]:
    """
    Load model weights and class labels.
    """
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not LABEL_INFO_PATH.is_file():
        raise FileNotFoundError(f"Label info not found: {LABEL_INFO_PATH}")

    label_info = joblib.load(LABEL_INFO_PATH)
    classes = list(label_info["classes"])
    base_channels = int(label_info.get("base_channels", 32))

    model = STGCNClassifier(num_classes=len(classes), base_channels=base_channels)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model, classes


def load_posture_artifacts():
    if not POSTURE_MODEL_PATH.is_file() or not POSTURE_SCALER_PATH.is_file():
        return None, None, None

    posture_model = joblib.load(POSTURE_MODEL_PATH)
    posture_scaler = joblib.load(POSTURE_SCALER_PATH)
    posture_label_encoder = (
        joblib.load(POSTURE_LABEL_ENCODER_PATH) if POSTURE_LABEL_ENCODER_PATH.is_file() else None
    )
    return posture_model, posture_scaler, posture_label_encoder


@app.on_event("startup")
def on_startup():
    global MODEL, CLASSES, POSTURE_MODEL, POSTURE_SCALER, POSTURE_LABEL_ENCODER
    MODEL, CLASSES = load_artifacts()
    POSTURE_MODEL, POSTURE_SCALER, POSTURE_LABEL_ENCODER = load_posture_artifacts()


@app.get("/", include_in_schema=False)
def serve_web_ui():
    if not WEB_INDEX_PATH.is_file():
        raise HTTPException(status_code=404, detail=f"Web UI not found: {WEB_INDEX_PATH}")
    return FileResponse(WEB_INDEX_PATH)


@app.get("/tester", include_in_schema=False)
def serve_web_tester():
    if not WEB_TESTER_PATH.is_file():
        raise HTTPException(status_code=404, detail=f"Web tester not found: {WEB_TESTER_PATH}")
    return FileResponse(WEB_TESTER_PATH)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "num_classes": len(CLASSES),
        "unknown_threshold": UNKNOWN_THRESHOLD,
        "posture_model_ready": POSTURE_MODEL is not None and POSTURE_SCALER is not None,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    arr = np.array(req.window, dtype=np.float32)
    if arr.ndim != 3:
        raise HTTPException(status_code=400, detail="window must be 3D [T, V, C]")
    if arr.shape[1] != 33 or arr.shape[2] != 3:
        raise HTTPException(
            status_code=400,
            detail=f"Expected shape [T, 33, 3], got [{arr.shape[0]}, {arr.shape[1]}, {arr.shape[2]}]",
        )

    arr = normalise_skeleton_sequence(arr)

    # Convert [T, V, C] -> [1, C, T, V, 1]
    x = np.transpose(arr, (2, 0, 1))  # [C, T, V]
    x = x[:, :, :, None][None, ...]   # [1, C, T, V, 1]
    x_tensor = torch.from_numpy(x).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(x_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    top_conf = float(probs[pred_idx])
    top_label = CLASSES[pred_idx]
    out_label = f"unknown({top_label})" if top_conf < UNKNOWN_THRESHOLD else top_label
    return PredictResponse(
        label=out_label,
        confidence=top_conf,
        probs=[float(p) for p in probs],
        classes=CLASSES,
    )


@app.post("/predict_posture", response_model=PosturePredictResponse)
def predict_posture(req: PosturePredictRequest):
    if POSTURE_MODEL is None or POSTURE_SCALER is None:
        raise HTTPException(status_code=500, detail="Posture model not loaded")

    arr = np.array(req.frame, dtype=np.float32)
    if arr.shape != (33, 4):
        raise HTTPException(
            status_code=400,
            detail=f"Expected shape [33, 4], got [{arr.shape[0]}, {arr.shape[1] if arr.ndim > 1 else 'NA'}]",
        )

    flat = arr.reshape(-1)  # [132]
    features_df = pd.DataFrame([flat], columns=POSTURE_FEATURE_NAMES)
    scaled = POSTURE_SCALER.transform(features_df)

    pred_idx = int(POSTURE_MODEL.predict(scaled)[0])
    if POSTURE_LABEL_ENCODER is not None:
        label = str(POSTURE_LABEL_ENCODER.inverse_transform([pred_idx])[0])
        classes = [str(c) for c in POSTURE_LABEL_ENCODER.classes_]
    else:
        label = str(pred_idx)
        classes = []

    if hasattr(POSTURE_MODEL, "predict_proba"):
        probs = POSTURE_MODEL.predict_proba(scaled)[0].astype(float).tolist()
        confidence = float(max(probs))
        if not classes:
            classes = [str(i) for i in range(len(probs))]
    else:
        probs = []
        confidence = 1.0

    return PosturePredictResponse(
        label=label,
        confidence=confidence,
        probs=probs,
        classes=classes,
    )
