# Cloud Fault Prediction API

This service loads `random_forest_model.pkl` when FastAPI starts and serves
inference at `POST /predict`.

## Run locally

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API docs are at <http://localhost:8000/docs>. The frontend can use
`VITE_API_BASE_URL=http://localhost:8000`.

## Prediction request

The saved model was trained using two log-block features, so requests must
include those exact values:

```json
{
  "total_logs": 24,
  "unique_components": 3,
  "model": "random_forest"
}
```

The response includes numeric `prediction` (`0` normal, `1` anomaly), a display
`label`, class confidence, timestamp, and inference latency. `/health` reports
model readiness. `/metrics` returns an empty model list because no evaluation
metrics are stored with the artifact.

The React dashboard sends the two model features above. Its live stream uses
simulated log-block samples for prototype demonstration; it is not connected to
real cloud telemetry. A telemetry-trained model and its feature contract are
needed before CPU, memory, network, or disk telemetry can produce meaningful
predictions. The CNN-LSTM option is unavailable because no such artifact is
included.

`joblib` uses Python pickle under the hood. Load only a model file from a
trusted source. The scikit-learn version used to load the file should match the
version used to create it (1.9.1 for the artifact in this repository).
