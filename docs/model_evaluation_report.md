# Model Evaluation Report

## Overview

This report is auto-generated from the latest ST-GCN artefacts:

- Dataset: `data/stgcn/stgcn_windows.npz`
- Model weights: `models/stgcn_best.pth`
- Labels: `models/stgcn_label_info.pkl`
- Generated at: `2026-04-26 17:28:30`

The evaluation uses the **same video-level train / val / test protocol** as training:

- Hold out **20%** of videos for test (stratified when possible).
- Of the remaining videos, hold out **25%** for validation.
- Approximate overall split: **60% train / 20% val / 20% test** by video count (`random_state=42`).

---

## Final Model

- Model: **ST-GCN (single-person skeleton action classifier)**
- Classes: **jumping_jack, pull_up, push_up, situp, squat**

---

## Summary Accuracies (best checkpoint)

| Split | Accuracy |
| --- | ---: |
| Train | **0.9855** |
| Validation | **0.8091** |
| Test | **0.8130** |

---

## Test Set (held-out videos)

Primary metrics for generalisation. Macro-averaged F1: **0.7954**.

### Per-Class Metrics (Test)

| Class | Precision | Recall | F1-Score | Support |
| --- | ---: | ---: | ---: | ---: |
| jumping_jack | 0.8200 | 0.9057 | 0.8607 | 870 |
| pull_up | 0.8209 | 0.7530 | 0.7855 | 919 |
| push_up | 0.8684 | 0.9157 | 0.8914 | 807 |
| situp | 0.8797 | 0.8909 | 0.8852 | 788 |
| squat | 0.5914 | 0.5214 | 0.5542 | 583 |

### Confusion Matrix (Test)

| Actual \ Predicted | jumping_jack | pull_up | push_up | situp | squat |
| --- | ---: | ---: | ---: | ---: | ---: |
| jumping_jack | 788 | 13 | 0 | 0 | 69 |
| pull_up | 23 | 692 | 43 | 20 | 141 |
| push_up | 31 | 0 | 739 | 37 | 0 |
| situp | 9 | 30 | 47 | 702 | 0 |
| squat | 110 | 108 | 22 | 39 | 304 |

---

## Validation Set

### Per-Class Metrics (Validation)

| Class | Precision | Recall | F1-Score | Support |
| --- | ---: | ---: | ---: | ---: |
| jumping_jack | 0.9049 | 0.8222 | 0.8616 | 1018 |
| pull_up | 0.7576 | 0.8460 | 0.7994 | 909 |
| push_up | 0.8763 | 0.9357 | 0.9050 | 840 |
| situp | 0.9025 | 0.8557 | 0.8785 | 714 |
| squat | 0.4951 | 0.4649 | 0.4795 | 542 |

### Confusion Matrix (Validation)

| Actual \ Predicted | jumping_jack | pull_up | push_up | situp | squat |
| --- | ---: | ---: | ---: | ---: | ---: |
| jumping_jack | 837 | 80 | 0 | 0 | 101 |
| pull_up | 30 | 769 | 23 | 6 | 81 |
| push_up | 1 | 0 | 786 | 0 | 53 |
| situp | 27 | 38 | 16 | 611 | 22 |
| squat | 30 | 128 | 72 | 60 | 252 |

---

## Training Set (same videos used during fit)

### Per-Class Metrics (Train)

| Class | Precision | Recall | F1-Score | Support |
| --- | ---: | ---: | ---: | ---: |
| jumping_jack | 0.9742 | 0.9872 | 0.9807 | 2261 |
| pull_up | 0.9863 | 0.9810 | 0.9837 | 2425 |
| push_up | 0.9977 | 0.9986 | 0.9981 | 2158 |
| situp | 0.9991 | 0.9960 | 0.9976 | 2251 |
| squat | 0.9629 | 0.9551 | 0.9590 | 1493 |

### Confusion Matrix (Train)

| Actual \ Predicted | jumping_jack | pull_up | push_up | situp | squat |
| --- | ---: | ---: | ---: | ---: | ---: |
| jumping_jack | 2232 | 3 | 0 | 0 | 26 |
| pull_up | 19 | 2379 | 1 | 1 | 25 |
| push_up | 0 | 0 | 2155 | 0 | 3 |
| situp | 0 | 8 | 0 | 2242 | 1 |
| squat | 40 | 22 | 4 | 1 | 1426 |

---

## Supporting Artefacts

- Training curves (train / val / test per epoch): `docs/stgcn_training_curves.png`
- Confusion matrices: `docs/confusion_matrix_stgcn_train.png`, `docs/confusion_matrix_stgcn_val.png`, `docs/confusion_matrix_stgcn_test.png`
- Legacy alias: `docs/confusion_matrix_stgcn.png` (validation)

---

## Threshold Check

- Required minimum **test** accuracy: **>= 0.80**
- Achieved test accuracy: **0.8130**

**Verdict: PASS**

---

## Additional Notes

- Early stopping and checkpointing use **validation** only; the test set is not used for model selection.
- If train accuracy is much higher than validation or test, consider regularisation or more data.

---
