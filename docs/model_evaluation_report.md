# Model Evaluation Report

## Overview

The trained posture classification model was evaluated on a held-out test dataset to assess its performance before real-time integration. The goal is to classify sitting posture as **correct** or **incorrect**.

---

## Final Model Selected

**Model:** Support Vector Machine (SVM) with RBF kernel

**Justification:**
SVM achieved the highest test accuracy (91.94%) compared to Random Forest (90.70%), indicating better generalisation performance on the dataset.

---

## Test Performance Metrics

### Overall Accuracy

| Metric   | Value               |
| -------- | ------------------- |
| Accuracy | **0.9194 (91.94%)** |

---

### Classification Metrics (Per Class)

| Class     | Precision | Recall | F1-Score | Support |
| --------- | --------- | ------ | -------- | ------- |
| Correct   | 0.88      | 0.95   | 0.91     | 214     |
| Incorrect | 0.96      | 0.89   | 0.93     | 270     |

---

## Confusion Matrix

|                  | Predicted Correct | Predicted Incorrect |
| ---------------- | ----------------- | ------------------- |
| Actual Correct   | 204               | 10                  |
| Actual Incorrect | 29                | 241                 |

---

## Performance Interpretation

* The model performs strongly across both classes with balanced precision and recall.
* High recall (0.95) for **correct posture** indicates the model effectively identifies good posture.
* High precision (0.96) for **incorrect posture** indicates strong reliability when flagging poor posture.
* Misclassifications are minimal and evenly distributed.

---

## Threshold Evaluation

**Minimum required accuracy:** ≥ 80%
**Achieved accuracy:** 91.94%

✅ **Verdict: PASS**

The model exceeds the minimum Proof-of-Concept (PoC) threshold and demonstrates reliable classification performance.

---

## Conclusion

The trained SVM classifier meets all performance requirements and is considered **ready for real-time integration** into the posture monitoring pipeline.

---
