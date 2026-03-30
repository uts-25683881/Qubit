# Qubit

The purpose of this PoC is to validate the feasibility of the core technological workflow. This prototype maintains minimal functionality and does not support processing multiple complex application scenarios, while also not requiring a complete user interface or posture scoring mechanism. Its primary objective is to demonstrate that the fundamental components of the system can be effectively integrated, while providing posture analysis and feedback evaluation within a real-time operational environment.



## Setup

### 1. Install dependencies:
pip install -r requirements.txt

### 2. Download MediaPipe pose model (optional)

For **`src/extract.py`** only (images → `data/landmarks/*.csv`). Skip if you already have a landmarks CSV.

Save as **`models/pose_landmarker_heavy.task`** (create `models/` if missing).

Linux / macOS:

```bash
wget -q -O models/pose_landmarker_heavy.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task
```

Windows (PowerShell):

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task" -OutFile "models\pose_landmarker_heavy.task"
```
