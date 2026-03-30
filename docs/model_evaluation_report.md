# Model Evaluation Report

## Overview

The posture classification model was trained using **raw body landmark coordinates** extracted from MediaPipe Pose. A total of **32 body landmarks (excluding nose)** were used, each represented by *(x, y, z, visibility)*, resulting in **128 input features per sample**.

The model was evaluated using a **70/30 stratified train-test split** to ensure balanced class distribution.

---

## Final Model Selected

**Model:** Random Forest Classifier

**Justification:**
The Random Forest model achieved the highest accuracy (**94.63%**) compared to the SVM baseline (**92.36%**), demonstrating better performance in capturing patterns in raw landmark data.

---

## Test Performance Metrics

### Overall Accuracy

| Metric   | Value               |
| -------- | ------------------- |
| Accuracy | **0.9463 (94.63%)** |

---

### Classification Metrics (Per Class)

| Class     | Precision | Recall | F1-Score | Support |
| --------- | --------- | ------ | -------- | ------- |
| Correct   | 0.93      | 0.95   | 0.94     | 214     |
| Incorrect | 0.96      | 0.94   | 0.95     | 270     |

---

## Confusion Matrix

|                  | Predicted Correct | Predicted Incorrect |
| ---------------- | ----------------- | ------------------- |
| Actual Correct   | 204               | 10                  |
| Actual Incorrect | 16                | 254                 |

---

## Performance Interpretation

* The model demonstrates **strong and balanced performance** across both classes.
* High recall for **correct posture (0.95)** indicates reliable detection of good posture.
* High precision for **incorrect posture (0.96)** shows strong confidence when identifying poor posture.
* Very low misclassification:

  * Only **10 correct postures misclassified**
  * Only **16 incorrect postures misclassified**
* Raw landmark input allows the model to directly learn spatial body patterns without dependency on handcrafted features.

---

## Threshold Evaluation

**Minimum required accuracy:** ≥ 80%
**Achieved accuracy:** 94.63%

✅ **Verdict: PASS**

The model significantly exceeds the required Proof-of-Concept (PoC) threshold.

---

## Real-Time Integration Readiness

The trained model has been successfully prepared for real-time deployment:

* Landmark extraction via MediaPipe Pose
* Direct use of raw landmark coordinates (no feature engineering mismatch)
* Scaled input using StandardScaler
* Prediction using Random Forest classifier

The system is capable of real-time inference and meets the performance target of ≥ 20 FPS on standard hardware.

---

## Conclusion

The raw landmark-based approach provides a **simpler and more robust pipeline**, eliminating inconsistencies between training and inference stages. The model achieves high accuracy and balanced performance, making it suitable for real-time posture detection.

The system is **ready for integration into live webcam-based posture monitoring**.

---

## Future Improvements

* Add temporal smoothing to reduce prediction flickering
* Expand dataset with varied camera angles and lighting conditions
* Explore deep learning approaches (e.g., LSTM for posture sequences)
* Add user feedback overlay (e.g., “Straighten your back”) for better UX

---
