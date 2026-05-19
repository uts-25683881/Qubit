import os
import time
import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd


MODEL_PATH = "models/posture_classifier.pkl"
SCALER_PATH = "models/scaler.pkl"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

LANDMARK_START = 0
NUM_LANDMARKS = 33

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

MODEL = joblib.load(MODEL_PATH)
SCALER = joblib.load(SCALER_PATH)
LABEL_ENCODER = joblib.load(LABEL_ENCODER_PATH) if os.path.exists(LABEL_ENCODER_PATH) else None

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

POSE = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def get_feature_names():
    names = []
    for i in range(LANDMARK_START, NUM_LANDMARKS):
        names.extend([f"x{i}", f"y{i}", f"z{i}", f"v{i}"])
    return names

FEATURE_NAMES = get_feature_names()

def pose_bbox_xyxy(frame_shape, pose_landmarks, pad_ratio=0.05):
    """Axis-aligned box from all pose landmarks (normalized x,y → pixels)."""
    h, w = frame_shape[:2]
    xs = [lm.x * w for lm in pose_landmarks.landmark]
    ys = [lm.y * h for lm in pose_landmarks.landmark]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    pw = (x2 - x1) * pad_ratio
    ph = (y2 - y1) * pad_ratio
    x1 = max(0, int(x1 - pw))
    y1 = max(0, int(y1 - ph))
    x2 = min(w - 1, int(x2 + pw))
    y2 = min(h - 1, int(y2 + ph))
    return x1, y1, x2, y2
    
def extract_landmarks(results):
    if results.pose_landmarks is None:
        return None

    values = []
    for i, lm in enumerate(results.pose_landmarks.landmark):
        if i < LANDMARK_START:
            continue
        values.extend([lm.x, lm.y, lm.z, lm.visibility])

    return np.array(values, dtype=np.float32)


def decode_label(pred_idx):
    if LABEL_ENCODER is not None:
        return str(LABEL_ENCODER.inverse_transform([pred_idx])[0])
    return {0: "correct", 1: "incorrect"}.get(pred_idx, str(pred_idx))


def get_confidence(scaled_features):
    if hasattr(MODEL, "predict_proba"):
        probs = MODEL.predict_proba(scaled_features)[0]
        return float(np.max(probs))
    return 1.0


def get_prediction(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = POSE.process(rgb)

    landmarks = extract_landmarks(results)
    if landmarks is None:
        return None

    features_df = pd.DataFrame([landmarks], columns=FEATURE_NAMES)
    scaled_features = SCALER.transform(features_df)

    pred_idx = int(MODEL.predict(scaled_features)[0])
    label = decode_label(pred_idx)
    confidence = get_confidence(scaled_features)

    return label, confidence, results


def run_detection():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Webcam started. Press Q to quit.")

    prev_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        prediction = get_prediction(frame)

        current_time = time.perf_counter()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

        if prediction is None:
            cv2.putText(
                frame,
                f"No person detected | FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            label, confidence, results = prediction

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )
                bx1, by1, bx2, by2 = pose_bbox_xyxy(frame.shape, results.pose_landmarks)
                bbox_color = (0, 255, 0) if label == "correct" else (0, 0, 255)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), bbox_color, 2, cv2.LINE_AA)

            color = (0, 255, 0) if label == "correct" else (0, 0, 255)
            cv2.putText(
                frame,
                f"{label} | conf: {confidence:.2f} | FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

            print(f"\rLabel: {label:<9} | Confidence: {confidence:.2f} | FPS: {fps:.1f}", end="")

        cv2.imshow("Real-Time Posture Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q")):
            break

    print("\nExiting...")
    cap.release()
    cv2.destroyAllWindows()


def main():
    run_detection()


if __name__ == "__main__":
    main()
