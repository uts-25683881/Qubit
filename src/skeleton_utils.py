from __future__ import annotations

import numpy as np


# MediaPipe landmark indices used for normalisation anchors.
LEFT_SHOULDER = 11   # Left shoulder joint index
RIGHT_SHOULDER = 12  # Right shoulder joint index
LEFT_HIP = 23        # Left hip joint index
RIGHT_HIP = 24       # Right hip joint index

# Safety hyperparameters for robust normalisation.
MIN_BODY_SCALE = 0.1  # Minimum denominator to avoid scale explosion
COORD_CLIP = 8.0      # Clip normalised coordinates into [-8, 8]

def normalise_skeleton_frame(
    frame: np.ndarray,
    eps: float = 1e-6,
    min_scale: float = MIN_BODY_SCALE,
    clip_value: float = COORD_CLIP,
) -> np.ndarray:
    """
    Normalise one skeleton frame [V, 3] by:
    1) hip-centre translation
    2) body-scale normalisation
    3) coordinate clipping for robustness
    """
    # Ensure a numeric array with stable precision.
    arr = np.asarray(frame, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected frame shape [V, 3], got {arr.shape}")
    if arr.shape[0] <= RIGHT_HIP:
        raise ValueError(f"Expected at least 25 joints, got {arr.shape[0]}")

    # Centre the body around the midpoint between both hips.
    hip_centre = 0.5 * (arr[LEFT_HIP] + arr[RIGHT_HIP])
    shifted = arr - hip_centre

    # Estimate body size from shoulder width / torso length.
    shoulder_width = np.linalg.norm(arr[LEFT_SHOULDER] - arr[RIGHT_SHOULDER])
    torso = np.linalg.norm(hip_centre - 0.5 * (arr[LEFT_SHOULDER] + arr[RIGHT_SHOULDER]))
    body_scale = max(shoulder_width, torso, min_scale, eps)

    # Scale then clamp outliers to a bounded range.
    normalised = shifted / body_scale
    return np.clip(normalised, -clip_value, clip_value)

def normalise_skeleton_sequence(
    sequence: np.ndarray,
    eps: float = 1e-6,
    min_scale: float = MIN_BODY_SCALE,
    clip_value: float = COORD_CLIP,
) -> np.ndarray:
    """
    Normalise a sequence [T, V, 3] one frame at a time.
    """
    # Validate sequence layout before per-frame processing.
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim != 3 or seq.shape[2] != 3:
        raise ValueError(f"Expected sequence shape [T, V, 3], got {seq.shape}")

    out = np.empty_like(seq)
    for i in range(seq.shape[0]):
        # Reuse single-frame normalisation for consistency.
        out[i] = normalise_skeleton_frame(
             seq[i],
            eps=eps,
            min_scale=min_scale,
            clip_value=clip_value,
        )
    return out
