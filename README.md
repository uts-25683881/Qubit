# Qubit

Qubit is a posture and exercise recognition PoC built around MediaPipe skeleton landmarks.

The project currently contains two tracks:

- Legacy single-frame posture pipeline (RandomForest/SVM).
- New sequence-based ST-GCN exercise recognition pipeline (recommended).

---

## Current Recommended Pipeline (ST-GCN)

### Objective

Recognize 5 exercise classes from skeleton sequences:

- `jumping_jack`
- `pull_up`
- `push_up`
- `situp`
- `squat`

### Data Source

- `dataset/Physical Exercise Recognition Time Series Dataset/landmarks.csv`
- `dataset/Physical Exercise Recognition Time Series Dataset/labels.csv`

### Core Steps

1. Build sliding-window sequence dataset (`[N, C, T, V, M]`).
2. Train ST-GCN model.
3. Run real-time webcam demo.
4. (Optional) Serve predictions via FastAPI for UI integration.

---

## Environment Setup

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:

- `mediapipe`
- `opencv-python`
- `numpy`
- `pandas`
- `torch`
- `fastapi`
- `uvicorn`

### 2) GPU setup (recommended for training speed)

If you use NVIDIA GPU, install CUDA-enabled PyTorch in your conda env:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Verify:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

---

## Build ST-GCN Dataset

Generate training windows from time-series landmarks:

```bash
python train/data_preprocessing_stgcn.py
```

Expected output:

- `data/stgcn/stgcn_windows.npz`
- Console stats including:
  - `X shape`
  - `y shape`
  - `classes`
  - `feature_stats`
  - `skipped_windows`

---

## Train ST-GCN

### Standard training

```bash
python train/train_stgcn.py --epochs 30 --batch-size 128 --num-workers 4
```

Key behaviours:

- Video-level split (prevents leakage across windows from same video).
- Early stopping (`--patience`) and checkpointing use `--best-metric` (default `val_loss`; use `val_acc` for legacy behaviour).
- Default learning rate `--lr 5e-4` (override with `--lr 1e-3` if you want the previous default).
- `ReduceLROnPlateau` on validation loss (disable with `--no-lr-scheduler`).
- AdamW `--weight-decay` (default `1e-4`; increase if overfitting).
- Mixed precision on CUDA.

Artifacts:

- `models/stgcn_best.pth`
- `models/stgcn_label_info.pkl`
- `docs/stgcn_training_curves.png` (train/val loss and val accuracy per epoch)
- `docs/confusion_matrix_stgcn.png` (validation, best checkpoint; same as `confusion_matrix_stgcn_val.png`)
- `docs/confusion_matrix_stgcn_val.png`
- `docs/confusion_matrix_stgcn_train.png` (training set, best checkpoint)

---

## Real-Time Demo

Run webcam inference:

```bash
python src/detect_stgcn.py --unknown-threshold 0.55
```

Notes:

- Uses `pose_world_landmarks` first, then falls back to `pose_landmarks`.
- Uses unknown gating to reduce forced misclassification.
- Displays top-3 probabilities on screen.

---

## API for UI Integration

Start server:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:

- `GET /health`
- `POST /predict` with `window` shape `[T, 33, 3]`

Response includes:

- `label` (may be `unknown(...)`)
- `confidence`
- `probs`
- `classes`

---

## Legacy Pipeline (Single-Frame)

These files are kept for baseline/reference:

- `src/extract.py`
- `train/train.py`
- `src/detect.py`

This pipeline is not the primary path for sequence-based exercise recognition.

---

## Troubleshooting

### 1) Model predicts one class too often

- Rebuild NPZ and retrain.
- Confirm `feature_stats` are reasonable after dataset build.
- Use real webcam input (avoid testing by filming a screen).
- Tune `--unknown-threshold`.

### 2) Training is too slow

- Ensure CUDA-enabled PyTorch is installed.
- Increase `--batch-size` if GPU memory allows.
- For a quick smoke test, lower `--epochs` and `--patience` (for example `--epochs 10 --patience 3`).

### 3) `ModuleNotFoundError: No module named 'torch'`

- Torch is not installed in the active environment.
- Install torch in the same environment used to run scripts.

---

## Project Structure (Important Files)

- `train/data_preprocessing_stgcn.py` - ST-GCN sequence preprocessing and dataset builder
- `train/stgcn_model.py` - ST-GCN model
- `train/train_stgcn.py` - ST-GCN training entry
- `src/skeleton_utils.py` - shared normalization utilities
- `src/detect_stgcn.py` - real-time webcam demo
- `api/app.py` - FastAPI service for UI/backend integration
