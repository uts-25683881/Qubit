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

FEATURE_NAMES = [
    "neck_angle",
    "spine_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_knee_angle",
    "right_knee_angle",
    "shoulder_hip_ratio",
    "shoulder_y",
    "hip_y",
    "knee_y",
    "shoulder_hip_dy",
    "hip_knee_dy",
    "nose_shoulder_dy",
    "trunk_lean",
]

# Load once at startup
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

MODEL = joblib.load(MODEL_PATH)
SCALER = joblib.load(SCALER_PATH)
LABEL_ENCODER = joblib.load(LABEL_ENCODER_PATH) if os.path.exists(LABEL_ENCODER_PATH) else None

# MediaPipe Pose
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


def _get_xyz(landmarks: np.ndarray, idx: int) -> np.ndarray:
    """
    Extract (x, y, z) for one landmark from flattened (132,) array.
    """
    base = idx * 4
    return landmarks[base:base + 3]


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Angle at point B in 3D.
    """
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean distance in 3D.
    """
    return float(np.linalg.norm(a - b))


def extract_landmarks(results) -> np.ndarray | None:
    """
    Convert MediaPipe results to flattened (132,) array:
    [x0, y0, z0, v0, x1, y1, z1, v1, ..., x32, y32, z32, v32]
    """
    if results.pose_landmarks is None:
        return None

    row = []
    for lm in results.pose_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z, lm.visibility])

    return np.array(row, dtype=np.float32)


def extract_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Must match the training feature engineering exactly.
    """
    nose = _get_xyz(landmarks, 0)
    l_shoulder = _get_xyz(landmarks, 11)
    r_shoulder = _get_xyz(landmarks, 12)
    l_hip = _get_xyz(landmarks, 23)
    r_hip = _get_xyz(landmarks, 24)
    l_knee = _get_xyz(landmarks, 25)
    r_knee = _get_xyz(landmarks, 26)
    l_ankle = _get_xyz(landmarks, 27)
    r_ankle = _get_xyz(landmarks, 28)

    mid_shoulder = (l_shoulder + r_shoulder) / 2.0
    mid_hip = (l_hip + r_hip) / 2.0
    mid_knee = (l_knee + r_knee) / 2.0

    neck_angle = _angle(l_shoulder, nose, r_shoulder)
    spine_angle = _angle(nose, mid_shoulder, mid_hip)

    left_hip_angle = _angle(l_shoulder, l_hip, l_knee)
    right_hip_angle = _angle(r_shoulder, r_hip, r_knee)

    left_knee_angle = _angle(l_hip, l_knee, l_ankle)
    right_knee_angle = _angle(r_hip, r_knee, r_ankle)

    shoulder_width = _distance(l_shoulder, r_shoulder)
    hip_width = _distance(l_hip, r_hip)
    shoulder_hip_ratio = hip_width / (shoulder_width + 1e-6)

    body_height = abs(nose[1] - mid_knee[1]) + 1e-6

    shoulder_y = (mid_shoulder[1] - nose[1]) / body_height
    hip_y = (mid_hip[1] - nose[1]) / body_height
    knee_y = (mid_knee[1] - nose[1]) / body_height

    shoulder_hip_dy = (mid_hip[1] - mid_shoulder[1]) / body_height
    hip_knee_dy = (mid_knee[1] - mid_hip[1]) / body_height
    nose_shoulder_dy = (mid_shoulder[1] - nose[1]) / body_height

    trunk_lean = float(mid_hip[0] - mid_shoulder[0])

    return np.array([
        neck_angle,
        spine_angle,
        left_hip_angle,
        right_hip_angle,
        left_knee_angle,
        right_knee_angle,
        shoulder_hip_ratio,
        shoulder_y,
        hip_y,
        knee_y,
        shoulder_hip_dy,
        hip_knee_dy,
        nose_shoulder_dy,
        trunk_lean,
    ], dtype=np.float32)


def decode_label(pred_idx: int) -> str:
    if LABEL_ENCODER is not None:
        return str(LABEL_ENCODER.inverse_transform([pred_idx])[0])

    # Fallback only if label_encoder.pkl is missing
    return {0: "correct", 1: "incorrect"}.get(pred_idx, str(pred_idx))


def get_confidence(scaled_features: np.ndarray, pred_idx: int) -> float:
    if hasattr(MODEL, "predict_proba"):
        probs = MODEL.predict_proba(scaled_features)[0]
        return float(np.max(probs))

    if hasattr(MODEL, "decision_function"):
        score = MODEL.decision_function(scaled_features)
        if np.ndim(score) == 1:
            return float(1.0 / (1.0 + np.exp(-abs(score[0]))))

    return 1.0


def get_prediction(frame) -> tuple[str, float, object] | None:
    """
    Returns:
        (label, confidence, results) or None if no pose is detected
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = POSE.process(rgb)

    landmarks = extract_landmarks(results)
    if landmarks is None:
        return None

    features = extract_features(landmarks)
    pd.DataFrame([features], columns=FEATURE_NAMES).to_csv(
    "live_features.csv",
    mode="a",
    header=not os.path.exists("live_features.csv"),
    index=False
    )
    features_df = pd.DataFrame([features], columns=FEATURE_NAMES)
    scaled_features = SCALER.transform(features_df)

    pred_idx = int(MODEL.predict(scaled_features)[0])
    label = decode_label(pred_idx)
    confidence = get_confidence(scaled_features, pred_idx)

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
            print("\rNo person detected", end="")
        else:
            label, confidence, results = prediction

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

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

            print(
                f"\rLabel: {label:<9} | Confidence: {confidence:.2f} | FPS: {fps:.1f}",
                end=""
            )

        cv2.imshow("Real-Time Posture Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("\nExiting...")
    cap.release()
    cv2.destroyAllWindows()


def main():
    run_detection()


if __name__ == "__main__":
    main()
