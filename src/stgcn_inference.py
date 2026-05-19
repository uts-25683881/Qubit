"""
Shared ST-GCN inference: model paths, confidence gating, and tensor helpers.
Used by api/app.py and src/detect_stgcn.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch

from train.stgcn_model import STGCNClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "stgcn_best.pth"
LABEL_INFO_PATH = BASE_DIR / "models" / "stgcn_label_info.pkl"

WINDOW_SIZE = 24
UNKNOWN_THRESHOLD = 0.70
IDLE_LABEL = "idle"
# Per-class overrides: live webcam softmax is often flatter for some actions.
MIN_CONFIDENCE_BY_CLASS: Dict[str, float] = {
    "jumping_jack": 0.52,
}


def min_confidence_for_class(
    class_name: str,
    default_threshold: float = UNKNOWN_THRESHOLD,
    overrides: Optional[Dict[str, float]] = None,
) -> float:
    merged = {**MIN_CONFIDENCE_BY_CLASS, **(overrides or {})}
    return float(merged[class_name]) if class_name in merged else float(default_threshold)


def load_stgcn_model(
    device: Optional[torch.device] = None,
) -> Tuple[STGCNClassifier, List[str], torch.device]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not LABEL_INFO_PATH.is_file():
        raise FileNotFoundError(f"Label info not found: {LABEL_INFO_PATH}")

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_info = joblib.load(LABEL_INFO_PATH)
    classes = list(label_info["classes"])
    base_channels = int(label_info.get("base_channels", 32))

    model = STGCNClassifier(num_classes=len(classes), base_channels=base_channels)
    state = torch.load(MODEL_PATH, map_location=dev)
    model.load_state_dict(state)
    model.to(dev)
    model.eval()
    return model, classes, dev


def sequence_to_tensor(sequence: np.ndarray) -> torch.Tensor:
    """Normalised [T, V, 3] -> [1, C, T, V, 1] float32 tensor."""
    seq = np.asarray(sequence, dtype=np.float32)
    x = np.transpose(seq, (2, 0, 1))
    x = x[:, :, :, None][None, ...]
    return torch.from_numpy(x)


def predict_probs(model: STGCNClassifier, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits = model(x)
        return torch.softmax(logits, dim=1).cpu().numpy()[0]


def label_from_prediction(
    probs: np.ndarray,
    classes: List[str],
    default_threshold: float = UNKNOWN_THRESHOLD,
    overrides: Optional[Dict[str, float]] = None,
) -> Tuple[str, float, int]:
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    top_label = classes[pred_idx]
    need = min_confidence_for_class(top_label, default_threshold, overrides)
    out_label = IDLE_LABEL if confidence < need else top_label
    return out_label, confidence, pred_idx


def frame_to_skeleton(results) -> Optional[np.ndarray]:
    """
    MediaPipe pose -> [33, 3]. Prefer world landmarks, then image landmarks.
    """
    if getattr(results, "pose_world_landmarks", None) is not None:
        points = [[lm.x, lm.y, lm.z] for lm in results.pose_world_landmarks.landmark]
        return np.array(points, dtype=np.float32)

    if getattr(results, "pose_landmarks", None) is not None:
        points = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
        return np.array(points, dtype=np.float32)

    return None
