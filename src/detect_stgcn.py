from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.skeleton_utils import normalise_skeleton_frame  # noqa: E402
from src.stgcn_inference import (  # noqa: E402
    IDLE_LABEL,
    UNKNOWN_THRESHOLD,
    WINDOW_SIZE,
    frame_to_skeleton,
    load_stgcn_model,
    min_confidence_for_class,
    predict_probs,
    sequence_to_tensor,
)

EMA_ALPHA = 0.6
RESET_MISSING_FRAMES = 8


def run_demo(unknown_threshold: float, class_conf_overrides: dict[str, float] | None = None):
    model, classes, device = load_stgcn_model()

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
        top3_text = ""

        skeleton = frame_to_skeleton(results)
        if skeleton is not None:
            missing_count = 0
            skeleton = normalise_skeleton_frame(skeleton)
            window.append(skeleton)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if len(window) == WINDOW_SIZE:
                seq = np.stack(list(window), axis=0)
                x = sequence_to_tensor(seq).to(device)
                probs = predict_probs(model, x)

                if smooth_probs is None:
                    smooth_probs = probs
                else:
                    smooth_probs = EMA_ALPHA * probs + (1.0 - EMA_ALPHA) * smooth_probs

                pred_idx = int(np.argmax(smooth_probs))
                confidence = float(smooth_probs[pred_idx])
                top_indices = np.argsort(smooth_probs)[::-1][:3]
                top3_text = " | ".join([f"{classes[i]}:{smooth_probs[i]:.2f}" for i in top_indices])

                need = min_confidence_for_class(
                    classes[pred_idx], unknown_threshold, class_conf_overrides
                )
                if confidence < need:
                    label_text = IDLE_LABEL
                    conf_text = f"conf: {confidence:.2f} (need {need:.2f})"
                    color = (0, 255, 255)
                else:
                    label_text = classes[pred_idx]
                    conf_text = f"conf: {confidence:.2f}"
                    color = (0, 255, 0)
        else:
            missing_count += 1
            if missing_count >= RESET_MISSING_FRAMES:
                window.clear()
                smooth_probs = None

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
        help="Default min softmax for all classes except those in --jumping-jack-min-conf.",
    )
    parser.add_argument(
        "--jumping-jack-min-conf",
        type=float,
        default=None,
        help=(
            "Min softmax to accept jumping_jack (default 0.52). "
            "Lower if UI still shows idle while doing jacks."
        ),
    )
    args = parser.parse_args()
    overrides: dict[str, float] = {}
    if args.jumping_jack_min_conf is not None:
        overrides["jumping_jack"] = float(args.jumping_jack_min_conf)
    run_demo(unknown_threshold=args.unknown_threshold, class_conf_overrides=overrides or None)
