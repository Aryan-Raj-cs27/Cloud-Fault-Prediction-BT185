# Cloud-Fault-Prediction-BT185
**Project Title:** Fault-Aware Machine Learning and Deep Learning-Based Algorithm for Cloud Architecture

## Team Members
* Aryan Raj (Core ML/DL & Backend)
* Dheeraj Kumar (Data Engineering & Co-Training)
* Priyansh Kumar (Frontend & Visualization)
* Navneet Kumar (Quality Assurance & Testing)

## Project Overview
This project proposes a predictive hybrid architecture to continuously monitor cloud infrastructure telemetry and system logs. It utilizes a Random Forest module for immediate anomaly detection and a CNN-LSTM architecture to identify complex, time-series fault patterns.

## Frontend dashboard

The React/Vite monitoring dashboard is in `frontend/`. It includes live polling, manual single-sample inference, CSV batch evaluation, model analytics, an interactive alert center, configurable chart history, and exportable prediction history. Seeded telemetry and alerts are labeled as preview data until the first live prediction replaces them.

### Run locally

1. Install Node.js 18 or newer.
2. From `frontend/`, run `npm install`.
3. Copy `.env.example` to `.env` and set `VITE_API_BASE_URL` to the backend URL (for example `http://localhost:8000`).
4. Run `npm run dev` in `frontend/` and open the URL Vite prints.

### API contract

The dashboard uses configurable endpoints `VITE_HEALTH_ENDPOINT` (`GET /health`), `VITE_PREDICTION_ENDPOINT` (`POST /predict`), and `VITE_METRICS_ENDPOINT` (`GET /metrics`). Override their paths in `.env` to match the backend. Prediction requests send the entered telemetry fields plus a `model` value (`random_forest` or `cnn_lstm`). Responses should include a prediction/result/label/status and may include confidence/probability/score, model, timestamp, and latency_ms. The UI accepts common aliases for these fields. CSV rows are sent one at a time as prediction feature objects.

The analytics page accepts model comparison objects in a `models`, `comparison`, or `metrics` array, with `name`/`model`, accuracy, precision, recall, f1/f1_score, and optional confusion_matrix values. Preview metrics are shown if no metrics endpoint or compatible payload is available.
