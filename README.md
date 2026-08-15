# Sentinel

Sentinel is a streaming ML inference system for monitoring multivariate server telemetry with multiple anomaly detectors.

The project focuses on the systems side of machine learning: serving trained detector artifacts, scoring telemetry continuously, switching detectors while a session is running, applying calibrated thresholds, and monitoring inference behavior.

## Features

- Real-time telemetry scoring
- Runtime detector switching
- Persistent inference sessions
- Detector-specific calibrated thresholds
- Live alert monitoring
- P95 inference latency tracking
- Offline detector diagnostics
- Artifact persistence
- MLflow experiment tracking

## Detectors

Sentinel currently includes:

- Random baseline
- Z-Score
- PCA
- Isolation Forest
- LSTM Autoencoder

All detectors follow a common interface so the serving system can operate different detection approaches without changing the inference pipeline.

## Dataset

Sentinel is demonstrated on the **Server Machine Dataset (SMD)**:

- 28 server machines
- 38 telemetry signals per machine
- labeled anomaly regions for evaluation

SMD is used as the workload for demonstrating the inference system rather than as the main contribution of the project.

## System Design

Sentinel separates offline experimentation from online inference.

Offline:

`Telemetry → Train Detector → Evaluate → Calibrate Threshold → Persist Artifact`

Online:

`Telemetry → Active Detector → Score → Threshold Decision → Alert → Monitoring`

The stream session is maintained independently from the selected detector, allowing detectors to be switched without restarting the telemetry timeline.

## Tech Stack

- **Python / NumPy** — data processing and inference
- **scikit-learn** — PCA and Isolation Forest
- **PyTorch** — LSTM Autoencoder
- **FastAPI** — inference service
- **Server-Sent Events** — live inference streaming
- **MLflow** — experiment tracking
- **Hydra** — experiment configuration
- **JavaScript** — monitoring dashboard