import numpy as np
import pandas as pd
from pathlib import Path
import sys


# Sliding window configuration.
WINDOW = 24 # number of frames in a window
STRIDE = 4 # number of frames to stride

# Input and output paths.
DATA_DIR = Path("dataset/Physical Exercise Recognition Time Series Dataset")
LANDMARKS_CSV = DATA_DIR / "landmarks.csv"
LABELS_CSV = DATA_DIR / "labels.csv"
OUTPUT_PATH = Path("data/stgcn/stgcn_windows.npz")

# Get project root directory
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


from src.skeleton_utils import normalise_skeleton_sequence

def get_joint_names(df: pd.DataFrame) -> list[str]:
    """
    Infer joint names from x_* columns and keep only joints with x/y/z.
    """
    x_cols = [c for c in df.columns if c.startswith("x_")] # x_noses, x_eyes, x_ears, etc.
    joint_names = [c[2:] for c in x_cols] # nose, eyes, ears, etc.
    valid = [j for j in joint_names if f"y_{j}" in df.columns and f"z_{j}" in df.columns] # only keep joints with x, y and z coordinates
    return valid # list of joint names

def frame_to_tensor(row: pd.Series, joints: list[str]) -> np.ndarray:
    """
    Convert one row into frame tensor with shape [V, 3].
    """
    points = [] # list of [x, y, z] coordinates
    for joint in joints:
        points.append([row[f"x_{joint}"], row[f"y_{joint}"], row[f"z_{joint}"]]) # x, y, z coordinates
    return np.array(points, dtype=np.float32) # [V, 3], 33 joints (nodes) * 3 coordinates (x, y, z)

def build_sliding_windows() -> None:
    """
    Build sliding-window samples for ST-GCN and save as NPZ.
    Output X shape: [N, C, T, V, M].
    """
    if not LANDMARKS_CSV.is_file():
        raise FileNotFoundError(f"Missing landmarks file: {LANDMARKS_CSV}")
    if not LABELS_CSV.is_file():
        raise FileNotFoundError(f"Missing labels file: {LABELS_CSV}")

    landmarks_df = pd.read_csv(LANDMARKS_CSV) # read landmarks dataframe
    labels_df = pd.read_csv(LABELS_CSV) # read labels dataframe

    # Map each video id to its class label.
    label_map = dict(zip(labels_df["vid_id"], labels_df["class"]))
    joints = get_joint_names(landmarks_df) # list of joint names
    v_count = len(joints) # number of joints (nodes)
    print(f"Detected joints: {v_count}") # print number of joints (nodes)

    sequence_list = []  # list of sequences [T, V, 3]
    label_list = []  # string labels per window
    vid_list = [] # list of video ids
    skipped_windows = 0 # number of skipped windows

    # Process one video at a time to preserve temporal order.
    for vid_id, group_df in landmarks_df.groupby("vid_id", sort=False):
        if vid_id not in label_map:
            continue

        # Sort frames to guarantee correct sequence order.
        frame_order = group_df.sort_values("frame_order")
        frames = np.stack([frame_to_tensor(row, joints) for _, row in frame_order.iterrows()], axis=0)
        total_frames = frames.shape[0]

        if total_frames < WINDOW:
            continue

        # Build overlapping windows: [T, V, 3].
        for start in range(0, total_frames - WINDOW + 1, STRIDE):
            seq = frames[start:start + WINDOW]  # [T, V, 3]
            seq = normalise_skeleton_sequence(seq)
            # Skip windows containing NaN/Inf after normalisation.
            if not np.isfinite(seq).all():
                skipped_windows += 1
                continue
            sequence_list.append(seq)
            label_list.append(label_map[vid_id])
            vid_list.append(vid_id)

    if not sequence_list:
        raise RuntimeError("No windows generated. Check WINDOW/STRIDE and dataset content.")

    x = np.stack(sequence_list).astype(np.float32)  # [N, T, V, 3]
    y_text = np.array(label_list)

    # Encode string labels into integer class ids.
    classes = sorted(np.unique(y_text))
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    y = np.array([class_to_idx[label] for label in y_text], dtype=np.int64)

    # Convert to ST-GCN common format: [N, C, T, V, M].
    x_stgcn = np.transpose(x, (0, 3, 1, 2))  # [N, 3, T, V]
    x_stgcn = x_stgcn[:, :, :, :, None]      # [N, 3, T, V, 1]

    # Save all artefacts required for training.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        X=x_stgcn,
        y=y,
        classes=np.array(classes),
        vid_ids=np.array(vid_list),
    )

    print(f"Saved -> {OUTPUT_PATH}")
    print(f"X shape: {x_stgcn.shape} (N, C, T, V, M)")
    print(f"y shape: {y.shape}")
    print(f"classes: {classes}")
    print(f"skipped_windows: {skipped_windows}")
    print(f"feature_stats: min={x.min():.3f}, max={x.max():.3f}, q01={np.quantile(x, 0.01):.3f}, q99={np.quantile(x, 0.99):.3f}")


if __name__ == "__main__":
    build_sliding_windows()
