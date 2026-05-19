import time
import cv2
import mediapipe as mp

from src.detect import get_prediction, get_bounding_box, draw_bounding_box


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def open_camera():
    """
    Open webcam with a Windows-friendly fallback.
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    return cap


def draw_status_text(frame, label, confidence, fps):
    """
    Draw class label, confidence, and FPS on the frame.
    """
    is_correct = label.lower() == "correct"
    color = (0, 255, 0) if is_correct else (0, 0, 255)

    display_label = "Correct Posture" if is_correct else "Incorrect Posture"
    text = f"{display_label} | Confidence: {confidence:.2f} | FPS: {fps:.1f}"

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_no_detection(frame, fps):
    """
    Draw message when no person is detected.
    """
    cv2.putText(
        frame,
        f"No person detected | FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main():
    cap = open_camera()
    print("System started. Press Q to quit.")

    prev_time = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        prediction = get_prediction(frame)

        current_time = time.perf_counter()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

        if prediction is None:
            draw_no_detection(frame, fps)
        else:
            label, confidence, results = prediction

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            bbox = get_bounding_box(results, frame.shape)
            draw_bounding_box(frame, bbox, label)
            draw_status_text(frame, label, confidence, fps)

        cv2.imshow("Real-Time Posture Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("Exiting...")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
