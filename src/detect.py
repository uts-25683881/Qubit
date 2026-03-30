import os
import time
import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd


MODEL_PATH = "Qubit-main/models/posture_classifier.pkl"
SCALER_PATH = "Qubit-main/models/scaler.pkl"
LABEL_ENCODER_PATH = "Qubit-main/models/label_encoder.pkl"

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


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Missing scaler file: {SCALER_PATH}")

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
    min_tracking_confidence=0.5
)


def get_point(landmarks_array, index):
    start = index * 4
    return landmarks_array[start:start + 4]


def midpoint(p1, p2):
    return np.array([
        (p1[0] + p2[0]) / 2.0,
        (p1[1] + p2[1]) / 2.0,
        (p1[2] + p2[2]) / 2.0
    ], dtype=np.float32)


def calculate_angle(a, b, c):
    a = np.array(a[:2], dtype=np.float32)
    b = np.array(b[:2], dtype=np.float32)
    c = np.array(c[:2], dtype=np.float32)

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    return float(angle)


def extract_landmarks(results):
    if results.pose_landmarks is None:
        return None

    landmarks = []
    for lm in results.pose_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])

    return np.array(landmarks, dtype=np.float32)


def extract_features(landmarks_array):
    nose = get_point(landmarks_array, 0)

    left_shoulder = get_point(landmarks_array, 11)
    right_shoulder = get_point(landmarks_array, 12)

    left_hip = get_point(landmarks_array, 23)
    right_hip = get_point(landmarks_array, 24)

    left_knee = get_point(landmarks_array, 25)
    right_knee = get_point(landmarks_array, 26)

    left_ankle = get_point(landmarks_array, 27)
    right_ankle = get_point(landmarks_array, 28)

    shoulder_mid = midpoint(left_shoulder, right_shoulder)
    hip_mid = midpoint(left_hip, right_hip)
    knee_mid = midpoint(left_knee, right_knee)

    neck_angle = calculate_angle(nose, shoulder_mid, hip_mid)
    spine_angle = calculate_angle(shoulder_mid, hip_mid, knee_mid)
    left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

    shoulder_width = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
    hip_width = np.linalg.norm(left_hip[:2] - right_hip[:2])
    shoulder_hip_ratio = float(shoulder_width / hip_width) if hip_width > 0 else 0.0

    shoulder_y = float((left_shoulder[1] + right_shoulder[1]) / 2.0)
    hip_y = float((left_hip[1] + right_hip[1]) / 2.0)
    knee_y = float((left_knee[1] + right_knee[1]) / 2.0)

    shoulder_hip_dy = float(hip_y - shoulder_y)
    hip_knee_dy = float(knee_y - hip_y)
    nose_shoulder_dy = float(shoulder_y - nose[1])

    trunk_lean = float(abs(shoulder_mid[0] - hip_mid[0]))

    features = [
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
        trunk_lean
    ]

    return np.array(features, dtype=np.float32)


def decode_label(pred_idx):
    if LABEL_ENCODER is not None:
        return str(LABEL_ENCODER.inverse_transform([pred_idx])[0])
    return {0: "correct", 1: "incorrect"}.get(pred_idx, str(pred_idx))


def get_prediction(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = POSE.process(rgb)

    landmarks_array = extract_landmarks(results)
    if landmarks_array is None:
        return None

    features = extract_features(landmarks_array)
    features_df = pd.DataFrame([features], columns=FEATURE_NAMES)
    scaled_features = SCALER.transform(features_df)

    pred_idx = int(MODEL.predict(scaled_features)[0])
    label = decode_label(pred_idx)

    if hasattr(MODEL, "predict_proba"):
        confidence = float(np.max(MODEL.predict_proba(scaled_features)[0]))
    else:
        confidence = 1.0

    return label, confidence, results


def run_detection():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Webcam started. Press Q to quit.")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        prediction = get_prediction(frame)

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if current_time != prev_time else 0
        prev_time = current_time

        if prediction is None:
            cv2.putText(
                frame,
                f"No person detected | FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
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
                2
            )

            print(f"\rLabel: {label:<9} | Confidence: {confidence:.2f} | FPS: {fps:.1f}", end="")

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
