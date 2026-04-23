from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import torch


# Make sure we can import training modules from project root.
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from train.stgcn_model import STGCNClassifier  # noqa: E402
from src.skeleton_utils import normalise_skeleton_frame  # noqa: E402


MODEL_PATH = BASE_DIR / "models" / "stgcn_best.pth"
LABEL_INFO_PATH = BASE_DIR / "models" / "stgcn_label_info.pkl"
WINDOW_SIZE = 24
EMA_ALPHA = 0.6
UNKNOWN_THRESHOLD = 0.65
RESET_MISSING_FRAMES = 8


def load_model_and_labels():
    """
    Load trained ST-GCN model and class labels.
    """
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not LABEL_INFO_PATH.is_file():
        raise FileNotFoundError(f"Label info not found: {LABEL_INFO_PATH}")

    label_info = joblib.load(LABEL_INFO_PATH)
    classes = list(label_info["classes"])
    base_channels = int(label_info.get("base_channels", 32))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STGCNClassifier(num_classes=len(classes), base_channels=base_channels)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, classes, device


def frame_to_skeleton(results) -> np.ndarray | None:
    """
    Convert MediaPipe landmarks to [33, 3] float32 array.
    Priority:
    1) pose_world_landmarks (closer to world-coordinate training data)
    2) pose_landmarks (normalized image coordinates) as fallback
    """
    if getattr(results, "pose_world_landmarks", None) is not None:
        points = []
        for lm in results.pose_world_landmarks.landmark:
            points.append([lm.x, lm.y, lm.z])
        return np.array(points, dtype=np.float32)

    if getattr(results, "pose_landmarks", None) is not None:
        points = []
        for lm in results.pose_landmarks.landmark:
            points.append([lm.x, lm.y, lm.z])
        return np.array(points, dtype=np.float32)

    return None


def to_model_input(window: deque[np.ndarray]) -> torch.Tensor:
    """
    Convert a skeleton window into ST-GCN input [1, C, T, V, M].
    """
    seq = np.stack(list(window), axis=0)            # [T, V, 3]
    x = np.transpose(seq, (2, 0, 1))                # [3, T, V]
    x = x[:, :, :, None]                            # [3, T, V, 1]
    x = x[None, ...]                                # [1, 3, T, V, 1]
    return torch.from_numpy(x.astype(np.float32))


def run_demo(unknown_threshold: float):
    model, classes, device = load_model_and_labels()

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("ST-GCN demo started. Press Q to quit.")
    print(f"Device: {device}")

    window: deque[np.ndarray] = deque(maxlen=WINDOW_SIZE)
    smooth_probs = None
    missing_count = 0
    prev_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        current_time = time.perf_counter()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

        label_text = "Collecting sequence..."
        conf_text = ""
        color = (0, 255, 255)

        skeleton = frame_to_skeleton(results)
        if skeleton is not None:
            missing_count = 0
            skeleton = normalise_skeleton_frame(skeleton)
            window.append(skeleton)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if len(window) == WINDOW_SIZE:
                x = to_model_input(window).to(device)
                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                if smooth_probs is None:
                    smooth_probs = probs
                else:
                    smooth_probs = EMA_ALPHA * probs + (1.0 - EMA_ALPHA) * smooth_probs

                pred_idx = int(np.argmax(smooth_probs))
                confidence = float(smooth_probs[pred_idx])
                top_indices = np.argsort(smooth_probs)[::-1][:3]
                top3_text = " | ".join([f"{classes[i]}:{smooth_probs[i]:.2f}" for i in top_indices])

                if confidence < unknown_threshold:
                    label_text = f"unknown({classes[pred_idx]})"
                    conf_text = f"conf: {confidence:.2f}"
                    color = (0, 255, 255)
                else:
                    label_text = classes[pred_idx]
                    conf_text = f"conf: {confidence:.2f}"
                    color = (0, 255, 0)
            else:
                top3_text = ""
        else:
            missing_count += 1
            if missing_count >= RESET_MISSING_FRAMES:
                window.clear()
                smooth_probs = None
            top3_text = ""

        cv2.putText(
            frame,
            f"{label_text} {conf_text} | FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Window: {len(window)}/{WINDOW_SIZE}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if top3_text:
            cv2.putText(
                frame,
                f"Top3: {top3_text}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow("ST-GCN Real-Time Demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ST-GCN webcam demo.")
    parser.add_argument(
        "--unknown-threshold",
        type=float,
        default=UNKNOWN_THRESHOLD,
        help="Confidence threshold below which prediction is shown as unknown.",
    )
    args = parser.parse_args()
    run_demo(unknown_threshold=args.unknown_threshold)
