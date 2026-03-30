import numpy as np
import pandas as pd
from pathlib import Path
 
# Paths 
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR.parent / "data"
 
# MediaPipe Pose landmark indices (for reference) 
# 0:  nose
# 11: left shoulder   12: right shoulder
# 23: left hip        24: right hip
# 25: left knee       26: right knee
# 27: left ankle      28: right ankle
 
 
def _get_xyz(landmarks: np.ndarray, idx: int) -> np.ndarray:
    """
    Extracts (x, y, z) for a landmark by index.
    landmarks is a flat (132,) array: [x0,y0,z0,v0, x1,y1,z1,v1, ...]
    """
    base = idx * 4
    return landmarks[base:base + 3]   # x, y, z only (skip visibility)
 
 
def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Computes the angle (degrees) at point B formed by vectors BA and BC.
    a, b, c are each (x, y, z) arrays.
    """
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))
 
 
def _distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two (x, y, z) points."""
    return float(np.linalg.norm(a - b))
 
 
def extract_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Transforms a raw (132,) landmark array into a compact (14,) feature vector.
 
    Features:
        [0]  neck_angle          — angle at nose (11, 0, 12)
        [1]  spine_angle         — angle at left hip (11, 23, 25) using left side
        [2]  left_hip_angle      — angle at left hip (11, 23, 25)
        [3]  right_hip_angle     — angle at right hip (12, 24, 26)
        [4]  left_knee_angle     — angle at left knee (23, 25, 27)
        [5]  right_knee_angle    — angle at right knee (24, 26, 28)
        [6]  shoulder_hip_ratio  — hip width / shoulder width (scale-invariant)
        [7]  shoulder_y          — normalised y position of shoulder midpoint
        [8]  hip_y               — normalised y position of hip midpoint
        [9]  knee_y              — normalised y position of knee midpoint
        [10] shoulder_hip_dy     — vertical drop shoulder → hip
        [11] hip_knee_dy         — vertical drop hip → knee
        [12] nose_shoulder_dy    — vertical offset nose → shoulder midpoint
        [13] trunk_lean          — horizontal offset shoulder midpoint → hip midpoint
 
    Args:
        landmarks: np.ndarray of shape (132,) —
                   flattened [x0,y0,z0,v0, x1,y1,z1,v1, ..., x32,y32,z32,v32]
 
    Returns:
        np.ndarray of shape (14,)
    """
    # Extract key joint coordinates 
    nose        = _get_xyz(landmarks, 0)
    l_shoulder  = _get_xyz(landmarks, 11)
    r_shoulder  = _get_xyz(landmarks, 12)
    l_hip       = _get_xyz(landmarks, 23)
    r_hip       = _get_xyz(landmarks, 24)
    l_knee      = _get_xyz(landmarks, 25)
    r_knee      = _get_xyz(landmarks, 26)
    l_ankle     = _get_xyz(landmarks, 27)
    r_ankle     = _get_xyz(landmarks, 28)
 
    # Midpoints
    mid_shoulder = (l_shoulder + r_shoulder) / 2
    mid_hip      = (l_hip + r_hip) / 2
    mid_knee     = (l_knee + r_knee) / 2
 
    # Joint angles 
    # Neck: angle at nose between left and right shoulder
    neck_angle       = _angle(l_shoulder, nose, r_shoulder)
 
    # Spine: angle at shoulder midpoint between nose and hip midpoint
    spine_angle      = _angle(nose, mid_shoulder, mid_hip)
 
    # Hip angles (left and right independently)
    left_hip_angle   = _angle(l_shoulder, l_hip, l_knee)
    right_hip_angle  = _angle(r_shoulder, r_hip, r_knee)
 
    # Knee angles (left and right independently)
    left_knee_angle  = _angle(l_hip, l_knee, l_ankle)
    right_knee_angle = _angle(r_hip, r_knee, r_ankle)
 
    # ── Scale-invariant distances ──────────────────────────────────────────────
    shoulder_width  = _distance(l_shoulder, r_shoulder)
    hip_width       = _distance(l_hip, r_hip)
 
    # Normalise hip width by shoulder width — removes camera distance effect
    shoulder_hip_ratio = hip_width / (shoulder_width + 1e-6)
 
    # ── Vertical alignment ratios (y increases downward in image space) ────────
    shoulder_y     = float(mid_shoulder[1])
    hip_y          = float(mid_hip[1])
    knee_y         = float(mid_knee[1])
 
    shoulder_hip_dy  = hip_y - shoulder_y         # positive = hip below shoulder
    hip_knee_dy      = knee_y - hip_y             # positive = knee below hip
    nose_shoulder_dy = mid_shoulder[1] - nose[1]  # positive = shoulder below nose
 
    # ── Trunk lean (horizontal offset) ────────────────────────────────────────
    # Positive = hips shifted right of shoulders (forward lean in profile view)
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
 
 
def build_feature_dataset(
    input_csv:  Path = None,
    output_csv: Path = None,
) -> pd.DataFrame:
    """
    Reads a landmarks CSV (produced by extract.py),
    applies extract_features() to each row, and saves the result.
 
    Args:
        input_csv:  Path to input landmarks CSV.
                    Defaults to data/landmarks/{split}.csv 
        output_csv: Path to save feature CSV.
                    Defaults to data/landmarks_features/{split}.csv
 
    Returns:
        DataFrame with columns: filename, class, + 14 feature columns.
    """
    if input_csv is None:
        raise ValueError("Provide input_csv path.")
    if output_csv is None:
        output_csv = input_csv.parent.parent / "landmarks_features" / input_csv.name
 
    output_csv.parent.mkdir(parents=True, exist_ok=True)
 
    df = pd.read_csv(input_csv)
 
    # Landmark columns are everything after 'filename' and 'class'
    landmark_cols = [c for c in df.columns if c not in ("filename", "class")]
    landmark_data = df[landmark_cols].values.astype(np.float32)  # (N, 132)
 
    features = np.array([extract_features(row) for row in landmark_data])  # (N, 14)
 
    out_df = pd.DataFrame(features, columns=FEATURE_NAMES)
    out_df.insert(0, "filename", df["filename"].values)
    out_df.insert(1, "class",    df["class"].values)
 
    out_df.to_csv(output_csv, index=False)
    print(f" {len(out_df)} rows → {output_csv}")
 
    return out_df
 
 
def run_feature_engineering():
    landmarks_dir  = DATA_DIR / "landmarks"
    features_dir   = DATA_DIR / "landmarks_features"
    features_dir.mkdir(parents=True, exist_ok=True)
 
    splits = ["train", "test", "valid"]
 
    print("── Feature Engineering ─────────────────────────────────")
    print(f"  Input  : {landmarks_dir.resolve()}")
    print(f"  Output : {features_dir.resolve()}")
    print(f"  Features ({len(FEATURE_NAMES)}): {', '.join(FEATURE_NAMES)}\n")
 
    all_dfs = []
    for split in splits:
        src = landmarks_dir / f"{split}.csv"
        dst = features_dir  / f"{split}.csv"
        if not src.exists():
            print(f"  [SKIP] {src.name} not found")
            continue
        print(f"  Processing {split}.csv...")
        df = build_feature_dataset(input_csv=src, output_csv=dst)
        all_dfs.append(df)
 
    # Also write a single combined file for convenience
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = features_dir / "all.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n Combined → {combined_path} ({len(combined)} total rows)")
 
    print(f"\n── Done ────────────────────────────────────────────────")
 
 
if __name__ == "__main__":
    run_feature_engineering()