import csv
import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import argparse
 
# Paths
BASE_DIR   = Path(__file__).resolve().parents[1]
DATA_DIR   = BASE_DIR / "data" / "raw"
OUT_DIR    = BASE_DIR / "data" / "landmarks"
MODEL_PATH = BASE_DIR / "models" / "pose_landmarker_heavy.task"

ROBOFLOW_ROOT = BASE_DIR / "dataset" / "Sitting Posture Classification.v1i.multiclass"
ROBOFLOW_LABEL_COLS = [
    "leaning_backward",
    "leaning_forward",
    "leaning_left",
    "leaning_right",
    "upright",
]

# Config 
SPLITS        = ["train", "test", "valid"]
CLASSES       = ["correct", "incorrect"]
NUM_LANDMARKS = 33
USE_ROBOFLOW = True 

# CSV header: filename, class, then x0,y0,z0,v0 ... x32,y32,z32,v32 ───────
HEADER = ["filename", "class"]
for i in range(NUM_LANDMARKS):
    HEADER += [f"x{i}", f"y{i}", f"z{i}", f"v{i}"]
 
def load_roboflow_split_df(split: str) -> pd.DataFrame:
    """Read Roboflow _classes.csv; column 'class' is the fine label (one of ROBOFLOW_LABEL_COLS)."""
    csv_path = ROBOFLOW_ROOT / split / "_classes.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    df["class"] = df[ROBOFLOW_LABEL_COLS].idxmax(axis=1)
    return df

def landmarks_vector_from_result(result) -> list | None:
    """33 landmarks × (x, y, z, visibility) → length-132 list, or None if no pose."""
    if not result.pose_landmarks:
        return None
    row: list = []
    for lm in result.pose_landmarks[0]:
        row += [lm.x, lm.y, lm.z, lm.visibility]
    return row

def process_roboflow_split(split: str, detector) -> tuple:
    """
    Roboflow multiclass folder: _classes.csv + flat images in split/.
    Writes OUT_DIR / {split}.csv, same HEADER as process_split.
    """
    split_dir = ROBOFLOW_ROOT / split
    if not split_dir.is_dir():
        print(f"  [SKIP] {split_dir} not found")
        return 0, 0

    df = load_roboflow_split_df(split)
    out_path = OUT_DIR / f"{split}.csv"
    extracted, failed = 0, 0

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

        for filename, cls_name in zip(df["filename"], df["class"]):
            img_path = split_dir / filename
            if not img_path.is_file():
                print(f"  [WARN] Missing file: {img_path.name}")
                failed += 1
                continue

            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"  [WARN] Could not read: {img_path.name}")
                failed += 1
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            vec = landmarks_vector_from_result(result)
            if vec is None:
                print(f"  [WARN] No pose detected: {img_path.name}")
                failed += 1
                continue

            writer.writerow([filename, cls_name] + vec)
            extracted += 1

    return extracted, failed

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
 
def run_extraction(*, use_roboflow: bool | None = None):
    rf = USE_ROBOFLOW if use_roboflow is None else use_roboflow

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
            if rf:
                extracted, failed = process_roboflow_split(split, detector)
            else:
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
    parser = argparse.ArgumentParser(description="Extract pose landmarks to data/landmarks/.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--roboflow", action="store_true", help="Roboflow multiclass under dataset/")
    group.add_argument("--raw", action="store_true", help="data/raw/{split}/correct|incorrect/")
    args = parser.parse_args()
    mode: bool | None = None
    if args.roboflow:
        mode = True
    elif args.raw:
        mode = False
    run_extraction(use_roboflow=mode)
 