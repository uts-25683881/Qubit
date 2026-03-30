import csv
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
 
# Paths 
BASE_DIR   = Path.cwd().parent
DATA_DIR   = BASE_DIR / "data" / "raw"
OUT_DIR    = BASE_DIR / "data" / "landmarks"
MODEL_PATH = BASE_DIR / "models" / "pose_landmarker_heavy.task"
 
# Config 
SPLITS        = ["train", "test", "valid"]
CLASSES       = ["correct", "incorrect"]
NUM_LANDMARKS = 33
 
# CSV header: filename, class, then x0,y0,z0,v0 ... x32,y32,z32,v32 ───────
HEADER = ["filename", "class"]
for i in range(NUM_LANDMARKS):
    HEADER += [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]
 
 
def process_split(split: str, detector) -> tuple:
    """
    Runs pose detection on all images in one split.
    Writes results to data/landmarks/{split}.csv.
    Returns (extracted, failed) counts.
    """
    out_path = OUT_DIR / f"{split}.csv"
    extracted, failed = 0, 0
 
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
 
        for cls_name in CLASSES:
            src_dir = DATA_DIR / split / cls_name
            if not src_dir.exists():
                print(f"  [SKIP] {src_dir} not found")
                continue
 
            for img_path in sorted(src_dir.iterdir()):
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
 
                frame = cv2.imread(str(img_path))
                if frame is None:
                    print(f"  [WARN] Could not read: {img_path.name}")
                    failed += 1
                    continue
 
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result   = detector.detect(mp_image)
 
                if not result.pose_landmarks:
                    print(f"  [WARN] No pose detected: {img_path.name}")
                    failed += 1
                    continue
 
                # Flatten all 33 landmarks → x, y, z, visibility
                row = [img_path.name, cls_name]
                for lm in result.pose_landmarks[0]:   # [0] = first person
                    row += [lm.x, lm.y, lm.z, lm.visibility]
 
                writer.writerow(row)
                extracted += 1
 
    return extracted, failed
 
 
def run_extraction():
    if not MODEL_PATH.exists():
        print(f" Model file not found at: {MODEL_PATH}")
        print("   Download it by running:")
        print("   wget -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task -P models/")
        return
 
    OUT_DIR.mkdir(parents=True, exist_ok=True)
 
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
 
    print("── MediaPipe Landmark Extraction ───────────────────────")
 
    total_extracted, total_failed = 0, 0
 
    with vision.PoseLandmarker.create_from_options(options) as detector:
        for split in SPLITS:
            print(f"\n  Processing {split}...")
            extracted, failed = process_split(split, detector)
            print(f"  {extracted} extracted  |   {failed} failed")
            total_extracted += extracted
            total_failed    += failed

    all_csvs = [OUT_DIR / f"{split}.csv" for split in SPLITS]
    existing = [p for p in all_csvs if p.exists()]
    
    if existing:
        combined = pd.concat([pd.read_csv(p) for p in existing], ignore_index=True)
        combined_path = OUT_DIR / "all.csv"
        combined.to_csv(combined_path, index=False)
        print(f"    all.csv    →  {len(combined)} rows (combined)")
 
    print(f"\n── Summary ─────────────────────────────────────────────")
    print(f"  Total extracted : {total_extracted}")
    print(f"  Total failed    : {total_failed}")
    print(f"  Output folder   : {OUT_DIR.resolve()}")
    print(f"\n  Files written:")
    for split in SPLITS:
        p = OUT_DIR / f"{split}.csv"
        if p.exists():
            rows = sum(1 for _ in open(p)) - 1
            print(f"    {split}.csv  →  {rows} rows")
 
 
if __name__ == "__main__":
    run_extraction()
 