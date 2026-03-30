# Qubit

The purpose of this PoC is to validate the feasibility of the core technological workflow. This prototype maintains minimal functionality and does not support processing multiple complex application scenarios, while also not requiring a complete user interface or posture scoring mechanism. Its primary objective is to demonstrate that the fundamental components of the system can be effectively integrated, while providing posture analysis and feedback evaluation within a real-time operational environment.



## Setup

### 1. Install dependencies:
pip install -r requirements.txt

### 2. Download MediaPipe model
Run this once from the project root:

wget -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

Or on Windows (PowerShell):
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task" -OutFile "pose_landmarker_heavy.task"

**Place the downloaded file 'pose_landmarker_heavy.task' in models/ folder.**
