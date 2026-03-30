import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
 
# Paths
BASE_DIR   = Path.cwd().parent
DATA_DIR   = BASE_DIR / "data" / "raw"
OUT_DIR    = BASE_DIR / "data" / "landmarks"
MODEL_PATH = BASE_DIR / "models" / "pose_landmarker_heavy.task"  # downloaded separately
 
# Config 
SPLITS  = ["train", "test", "valid"]
CLASSES = ["correct", "incorrect"]
 
# Skeleton drawing style 
POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),
    (9,10),(11,12),(11,13),(13,15),(15,17),(15,19),(17,19),
    (12,14),(14,16),(16,18),(16,20),(18,20),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(27,29),(27,31),(29,31),
    (24,26),(26,28),(28,30),(28,32),(30,32),
]
 
 
def draw_skeleton_on_black(image_shape, detection_result) -> np.ndarray:
    h, w = image_shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    for pose_landmarks in detection_result.pose_landmarks:
        # Convert normalised coords to pixel coords
        points = [
            (int(lm.x * w), int(lm.y * h))
            for lm in pose_landmarks
        ]

        # Draw connections
        for start_idx, end_idx in POSE_CONNECTIONS:
            cv2.line(canvas, points[start_idx], points[end_idx],
                     color=(0, 0, 255), thickness=2)

        # Draw landmark dots on top
        for x, y in points:
            cv2.circle(canvas, (x, y), radius=4,
                       color=(255, 255, 0), thickness=-1)

    return canvas
 
 
def process_image(img_path: Path, detector, out_path: Path) -> bool:
    """
    Reads one image, runs pose detection, draws skeleton on black canvas,
    saves to out_path.
    Returns True if successful, False otherwise.
    """
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"  [WARN] Could not read: {img_path.name}")
        return False
 
    # New API uses mp.Image (RGB)
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
 
    result = detector.detect(mp_image)
 
    if not result.pose_landmarks:
        print(f"  [WARN] No pose detected: {img_path.name}")
        return False
 
    canvas = draw_skeleton_on_black(frame.shape, result)
    cv2.imwrite(str(out_path), canvas)
    return True
 
 
def run_extraction():
    if not MODEL_PATH.exists():
        print(f"  Model file not found: {MODEL_PATH}")
        print("   Download it first by running:")
        print("   wget -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task")
        return
 
    # New Tasks API setup
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options      = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,   # static images, not video
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
 
    print("── MediaPipe Pose Landmark Extraction (Tasks API) ──────")
 
    total_extracted = 0
    total_failed    = 0
 
    with vision.PoseLandmarker.create_from_options(options) as detector:
        for split in SPLITS:
            print(f"\n  Processing {split}...")
            split_extracted = 0
            split_failed    = 0
 
            for cls_name in CLASSES:
                src_dir = DATA_DIR / split / cls_name
                dst_dir = OUT_DIR  / split / cls_name
                dst_dir.mkdir(parents=True, exist_ok=True)
 
                if not src_dir.exists():
                    print(f"  [SKIP] {src_dir} not found")
                    continue
 
                for img_path in sorted(src_dir.iterdir()):
                    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                        continue
 
                    out_path = dst_dir / img_path.name
                    success  = process_image(img_path, detector, out_path)
 
                    if success:
                        split_extracted += 1
                    else:
                        split_failed += 1
 
            print(f"  {split_extracted} extracted  |    {split_failed} failed")
            total_extracted += split_extracted
            total_failed    += split_failed
 
    print(f"\n── Summary ─────────────────────────────────────────────")
    print(f"  Total extracted : {total_extracted}")
    print(f"  Total failed    : {total_failed}")
    print(f"  Output folder   : {OUT_DIR.resolve()}")
 
 
if __name__ == "__main__":
    run_extraction()