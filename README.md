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

The dashboard uses configurable endpoints `VITE_HEALTH_ENDPOINT` (`GET /health`), `VITE_PREDICTION_ENDPOINT` (`POST /predict`), and `VITE_METRICS_ENDPOINT` (`GET /metrics`). The available Random Forest accepts `total_logs`, `unique_components`, and `model: "random_forest"`; its artifact was trained on those two log-block features. The dashboard's stream currently generates simulated log-block samples. This prototype does not yet ingest live CPU, memory, or network telemetry, and the CNN-LSTM model is unavailable until its artifact is added. CSV rows are sent one at a time and must include the two model feature columns.

The analytics page accepts model comparison objects in a `models`, `comparison`, or `metrics` array, with `name`/`model`, accuracy, precision, recall, f1/f1_score, and optional confusion_matrix values. Preview metrics are shown if no metrics endpoint or compatible payload is available.
