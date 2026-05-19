from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from train.stgcn_model import STGCNClassifier  # noqa: E402
from train.train_stgcn import (  # noqa: E402
    TEST_VIDEO_FRACTION,
    VAL_OF_TRAINVAL_VIDEO_FRACTION,
    _split_train_val_test_videos,
    _video_ids_and_labels,
)

DATA_PATH = BASE_DIR / "data" / "stgcn" / "stgcn_windows.npz"
MODEL_PATH = BASE_DIR / "models" / "stgcn_best.pth"
LABEL_INFO_PATH = BASE_DIR / "models" / "stgcn_label_info.pkl"
OUT_REPORT = BASE_DIR / "docs" / "model_evaluation_report.md"

RANDOM_STATE = 42
PASS_THRESHOLD = 0.80


@dataclass
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_npz(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    x = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    vid_ids = data["vid_ids"] if "vid_ids" in data.files else np.arange(len(y))
    return x, y, vid_ids


def split_three_way(x: np.ndarray, y: np.ndarray, vid_ids: np.ndarray) -> SplitData:
    """Same protocol as `train/train_stgcn.py` (stratified by video label when possible)."""
    unique, y_vid = _video_ids_and_labels(vid_ids, y)
    train_v, val_v, test_v = _split_train_val_test_videos(unique, y_vid, RANDOM_STATE)
    train_m = np.isin(vid_ids, train_v)
    val_m = np.isin(vid_ids, val_v)
    test_m = np.isin(vid_ids, test_v)
    return SplitData(
        x_train=x[train_m],
        y_train=y[train_m],
        x_val=x[val_m],
        y_val=y[val_m],
        x_test=x[test_m],
        y_test=y[test_m],
    )


@torch.no_grad()
def predict(
    model: STGCNClassifier,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    preds: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start : start + batch_size]).to(device)
        logits = model(xb)
        preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=np.int64)


def to_markdown_table_from_report(classes: list[str], report: dict) -> str:
    lines = [
        "| Class | Precision | Recall | F1-Score | Support |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for cls in classes:
        row = report.get(cls, {})
        lines.append(
            f"| {cls} | {row.get('precision', 0.0):.4f} | {row.get('recall', 0.0):.4f} | "
            f"{row.get('f1-score', 0.0):.4f} | {int(row.get('support', 0))} |"
        )
    return "\n".join(lines)


def to_markdown_confusion(classes: list[str], cm: np.ndarray) -> str:
    header = "| Actual \\ Predicted | " + " | ".join(classes) + " |"
    sep = "| --- | " + " | ".join(["---:"] * len(classes)) + " |"
    rows = [header, sep]
    for i, cls in enumerate(classes):
        values = " | ".join(str(int(v)) for v in cm[i])
        rows.append(f"| {cls} | {values} |")
    return "\n".join(rows)


def build_report_text(
    classes: list[str],
    train_acc: float,
    val_acc: float,
    test_acc: float,
    report_train: dict,
    report_val: dict,
    report_test: dict,
    cm_train: np.ndarray,
    cm_val: np.ndarray,
    cm_test: np.ndarray,
) -> str:
    verdict = "PASS" if test_acc >= PASS_THRESHOLD else "REVIEW"
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tbl_tr = to_markdown_table_from_report(classes, report_train)
    tbl_va = to_markdown_table_from_report(classes, report_val)
    tbl_te = to_markdown_table_from_report(classes, report_test)
    macro_te = report_test.get("macro avg", {})
    macro_f1 = float(macro_te.get("f1-score", 0.0))

    return f"""# Model Evaluation Report

## Overview

This report is auto-generated from the latest ST-GCN artefacts:

- Dataset: `data/stgcn/stgcn_windows.npz`
- Model weights: `models/stgcn_best.pth`
- Labels: `models/stgcn_label_info.pkl`
- Generated at: `{generated_at}`

The evaluation uses the **same video-level train / val / test protocol** as training:

- Hold out **{TEST_VIDEO_FRACTION:.0%}** of videos for test (stratified when possible).
- Of the remaining videos, hold out **{VAL_OF_TRAINVAL_VIDEO_FRACTION:.0%}** for validation.
- Approximate overall split: **60% train / 20% val / 20% test** by video count (`random_state={RANDOM_STATE}`).

---

## Final Model

- Model: **ST-GCN (single-person skeleton action classifier)**
- Classes: **{", ".join(classes)}**

---

## Summary Accuracies (best checkpoint)

| Split | Accuracy |
| --- | ---: |
| Train | **{train_acc:.4f}** |
| Validation | **{val_acc:.4f}** |
| Test | **{test_acc:.4f}** |

---

## Test Set (held-out videos)

Primary metrics for generalisation. Macro-averaged F1: **{macro_f1:.4f}**.

### Per-Class Metrics (Test)

{tbl_te}

### Confusion Matrix (Test)

{to_markdown_confusion(classes, cm_test)}

---

## Validation Set

### Per-Class Metrics (Validation)

{tbl_va}

### Confusion Matrix (Validation)

{to_markdown_confusion(classes, cm_val)}

---

## Training Set (same videos used during fit)

### Per-Class Metrics (Train)

{tbl_tr}

### Confusion Matrix (Train)

{to_markdown_confusion(classes, cm_train)}

---

## Supporting Artefacts

- Training curves (train / val / test per epoch): `docs/stgcn_training_curves.png`
- Confusion matrices: `docs/confusion_matrix_stgcn_train.png`, `docs/confusion_matrix_stgcn_val.png`, `docs/confusion_matrix_stgcn_test.png`
- Legacy alias: `docs/confusion_matrix_stgcn.png` (validation)

---

## Threshold Check

- Required minimum **test** accuracy: **>= {PASS_THRESHOLD:.2f}**
- Achieved test accuracy: **{test_acc:.4f}**

**Verdict: {verdict}**

---

## Additional Notes

- Early stopping and checkpointing use **validation** only; the test set is not used for model selection.
- If train accuracy is much higher than validation or test, consider regularisation or more data.

---
"""


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not LABEL_INFO_PATH.is_file():
        raise FileNotFoundError(f"Missing label info: {LABEL_INFO_PATH}")

    label_info = joblib.load(LABEL_INFO_PATH)
    classes = [str(c) for c in label_info["classes"]]
    base_channels = int(label_info.get("base_channels", 32))

    x, y, vid_ids = load_npz(DATA_PATH)
    split = split_three_way(x, y, vid_ids)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STGCNClassifier(num_classes=len(classes), base_channels=base_channels).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    y_train_pred = predict(model, split.x_train, device)
    y_val_pred = predict(model, split.x_val, device)
    y_test_pred = predict(model, split.x_test, device)

    labels = np.arange(len(classes))
    train_acc = float(accuracy_score(split.y_train, y_train_pred))
    val_acc = float(accuracy_score(split.y_val, y_val_pred))
    test_acc = float(accuracy_score(split.y_test, y_test_pred))

    report_train = classification_report(
        split.y_train,
        y_train_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    report_val = classification_report(
        split.y_val,
        y_val_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    report_test = classification_report(
        split.y_test,
        y_test_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    cm_train = confusion_matrix(split.y_train, y_train_pred, labels=labels)
    cm_val = confusion_matrix(split.y_val, y_val_pred, labels=labels)
    cm_test = confusion_matrix(split.y_test, y_test_pred, labels=labels)

    text = build_report_text(
        classes=classes,
        train_acc=train_acc,
        val_acc=val_acc,
        test_acc=test_acc,
        report_train=report_train,
        report_val=report_val,
        report_test=report_test,
        cm_train=cm_train,
        cm_val=cm_val,
        cm_test=cm_test,
    )
    OUT_REPORT.write_text(text, encoding="utf-8")
    print(f"Updated report: {OUT_REPORT}")


if __name__ == "__main__":
    main()
