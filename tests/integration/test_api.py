"""API schema tests on a tiny synthetic model bundle (no real data or artefacts needed)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from faultaware.features.store import DataBundle, save_feature_spec
from faultaware.fusion.risk import RiskFusion
from faultaware.models.base import create_model

PASSWORD = "test-password-123"


@pytest.fixture(scope="module")
def client(tiny_data: dict, tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    root = tmp_path_factory.mktemp("apiroot")
    models = root / "models"
    f = tiny_data["features"]
    templates = tiny_data["templates"]
    bundle = DataBundle(
        tiny_data["split"],
        tiny_data["vocab"],
        tiny_data["warn"],
        dict(zip(templates.template_id, templates.template, strict=True)),
        16,
    )
    save_feature_spec(
        bundle, tiny_data["events"].attrs["levels"], models / "serving" / "feature_spec.json"
    )
    sup = create_model("logreg", {"C": 1.0, "max_iter": 500}, 0)
    sup.fit(f["train"], f["train"].y, f["val"], f["val"].y)
    anom = create_model("iforest", {"n_estimators": 20, "max_samples": 64, "n_jobs": 1}, 0)
    anom.fit(f["train"], f["train"].y, f["val"], f["val"].y)
    sup.save(models / "artefacts" / "full" / "logreg")
    anom.save(models / "artefacts" / "full" / "iforest")
    fusion = RiskFusion("logreg", "iforest").fit(
        sup.predict_proba(f["val"]), anom.predict_proba(f["val"]), f["val"].y, [0.5, 1.0]
    )
    fusion.threshold = 0.5
    fusion.save(models / "fusion" / "full" / "fusion.joblib")
    (models / "fusion" / "full" / "fusion.json").write_text(
        json.dumps(
            {
                "supervised_model": "logreg",
                "anomaly_model": "iforest",
                "alpha": fusion.alpha,
                "threshold": 0.5,
                "observe_seconds": None,
            }
        ),
        encoding="utf-8",
    )
    (root / "configs" / "scheduler").mkdir(parents=True)
    (root / "configs" / "scheduler" / "fars.yaml").write_text("tau_low: 0.3\ntau_high: 0.6\n")
    (root / "app").mkdir()
    lines = tiny_data["lines"]
    sample_blk = lines[0].split("blk_")[1].split()[0]
    sample = [ln for ln in lines if f"blk_{sample_blk}" in ln]
    (root / "app" / "demo_samples.json").write_text(
        json.dumps([{"block_id": f"blk_{sample_blk}", "label": "Normal", "log_lines": sample}])
    )
    (root / "reports").mkdir()
    (root / "reports" / "results_summary.json").write_text(
        json.dumps(
            {
                "primary_window": "full",
                "windows": {
                    "full": {
                        "table": [
                            {
                                "model": "logreg",
                                "family": "classical ML",
                                "test_pr_auc": "0.9 ± 0.0",
                                "test_roc_auc": "0.9 ± 0",
                                "test_precision": "1 ± 0",
                                "test_recall": "1 ± 0",
                                "test_f1": "1 ± 0",
                                "test_fpr": "0 ± 0",
                                "latency_ms_per_sample": 0.01,
                            }
                        ],
                        "confusion": {"logreg": [[1, 0], [0, 1]]},
                        "best_supervised": "logreg",
                        "best_anomaly": "iforest",
                        "best_classical": "logreg",
                        "best_deep": "lstm",
                        "fusion": {},
                        "significance": [],
                        "early_warning": [
                            {
                                "model": "logreg",
                                "val_pr_auc": "0.2 ± 0.0",
                                "test_pr_auc": "0.7 ± 0.1",
                                "test_f1": "0.6 ± 0.0",
                                "test_alive": "10 (2 faults)",
                            }
                        ],
                    }
                },
            }
        )
    )
    (root / "reports" / "simulation_summary.json").write_text(json.dumps({"comparison": []}))
    (root / "reports" / "simulation_replay.json").write_text(
        json.dumps({"fars": {"metrics": {}, "trace": [{"t": i} for i in range(100)]}})
    )

    import app.api as api
    from app.security import save_user

    save_user(root / "secrets" / "auth.json", "admin", PASSWORD)
    api.ROOT, api.MODELS_DIR = root, models
    api.REPORTS_DIR, api.SAMPLES_FILE = root / "reports", root / "app" / "demo_samples.json"
    api.AUTH_FILE = root / "secrets" / "auth.json"
    with TestClient(api.app) as c:
        assert (
            c.post("/auth/login", json={"username": "admin", "password": PASSWORD}).status_code
            == 200
        )
        yield c, sample


def test_health(client) -> None:  # type: ignore[no-untyped-def]
    c, _ = client
    body = c.get("/health").json()
    assert body["status"] == "ok" and body["models_loaded"] == ["full"]


def test_predict_schema(client) -> None:  # type: ignore[no-untyped-def]
    c, sample = client
    r = c.post("/predict", json={"log_lines": sample, "window": "full"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"Normal", "Anomaly"} and 0 <= body["risk"] <= 1
    assert body["risk_level"] in {"low", "elevated", "high"}
    assert body["prediction"] in {0, 1} and body["events_observed"] == len(sample)


def test_predict_validation_errors(client) -> None:  # type: ignore[no-untyped-def]
    c, _ = client
    assert c.post("/predict", json={"log_lines": []}).status_code == 422
    assert c.post("/predict", json={"log_lines": ["garbage"]}).status_code == 422
    assert c.post("/predict", json={"log_lines": ["x"], "window": "w99"}).status_code == 422
    assert c.post("/predict", json={"log_lines": ["x"], "extra": 1}).status_code == 422


def test_model_info_metrics_samples_simulation(client) -> None:  # type: ignore[no-untyped-def]
    c, _ = client
    assert "full" in c.get("/model-info").json()["windows"]
    m = c.get("/metrics").json()
    assert m["models"][0]["name"] == "logreg" and m["models"][0]["pr_auc"] == 0.9
    assert m["early_warning"][0]["test_pr_auc"] == 0.7
    assert m["early_warning"][0]["test_recall"] is None  # older summaries lack the field
    assert c.get("/metrics", params={"window": "nope"}).status_code == 422
    assert c.get("/samples/next").json()["label"] == "Normal"
    assert c.get("/samples/next", params={"label": "Anomaly"}).status_code == 404
    sim = c.get("/simulation", params={"max_points": 10}).json()
    assert len(sim["replay"]["fars"]["trace"]) == 10
    assert "name" in c.get("/").json()


def test_numpy_unused_guard() -> None:
    assert np.isfinite(1.0)


def test_security_headers_and_docs_disabled(client) -> None:  # type: ignore[no-untyped-def]
    c, _ = client
    r = c.get("/health")
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'none'" in r.headers["content-security-policy"]
    assert c.get("/docs").status_code == 404 and c.get("/openapi.json").status_code == 404


def test_cross_origin_post_rejected(client) -> None:  # type: ignore[no-untyped-def]
    c, sample = client
    r = c.post("/predict", json={"log_lines": sample}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_auth_flow_and_protection(tiny_data: dict, client) -> None:  # type: ignore[no-untyped-def]
    import app.api as api

    c, sample = client
    assert c.get("/auth/me").json()["username"] == "admin"
    with TestClient(api.app) as anon:
        for path in ("/metrics", "/simulation", "/samples/next", "/model-info", "/auth/me"):
            assert anon.get(path).status_code == 401, path
        assert anon.post("/predict", json={"log_lines": sample}).status_code == 401
        bad = anon.post("/auth/login", json={"username": "admin", "password": "wrong-pass-123"})
        assert bad.status_code == 401 and bad.json()["detail"] == "Invalid username or password."
        ok = anon.post("/auth/login", json={"username": "admin", "password": PASSWORD})
        assert ok.status_code == 200
        cookie = ok.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite=strict" in cookie
        assert anon.get("/metrics").status_code == 200
        assert anon.post("/auth/logout").status_code == 200
        assert anon.get("/metrics").status_code == 401


def test_login_lockout(client) -> None:  # type: ignore[no-untyped-def]
    import app.api as api
    from app.security import LoginLimiter

    saved = api.LIMITER
    api.LIMITER = LoginLimiter(max_failures=2, window=60, lockout=60)
    try:
        with TestClient(api.app) as anon:
            for _ in range(2):
                anon.post("/auth/login", json={"username": "admin", "password": "nope-nope-nope"})
            r = anon.post("/auth/login", json={"username": "admin", "password": PASSWORD})
            assert r.status_code == 429 and "retry-after" in r.headers
    finally:
        api.LIMITER = saved


def test_production_server_serves_dashboard_and_api(client, tmp_path) -> None:  # type: ignore[no-untyped-def]
    import importlib
    import os

    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>FaultAware</title>", encoding="utf-8")
    (dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    os.environ["FAULTAWARE_FRONTEND_DIST"] = str(dist)
    try:
        import app.api as api

        root, models, auth = api.ROOT, api.MODELS_DIR, api.AUTH_FILE
        reports, samples = api.REPORTS_DIR, api.SAMPLES_FILE
        mod = importlib.reload(api)
        mod.ROOT, mod.MODELS_DIR, mod.AUTH_FILE = root, models, auth
        mod.REPORTS_DIR, mod.SAMPLES_FILE = reports, samples
        with TestClient(mod.server) as c:
            page = c.get("/")
            assert page.status_code == 200 and "FaultAware" in page.text
            assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
            assert "immutable" in c.get("/assets/app.js").headers["cache-control"]
            assert c.get("/api/health").json()["status"] in {"ok", "degraded"}
            assert c.get("/api/metrics").status_code == 401
            # Browsers send the page's own origin on same-origin POSTs; it must be accepted
            # even though it is not in CORS_ORIGINS (regression: production login was 403).
            same = {"Origin": "http://testserver"}
            ok = c.post(
                "/api/auth/login",
                json={"username": "admin", "password": PASSWORD},
                headers=same,
            )
            assert ok.status_code == 200 and c.get("/api/auth/me").status_code == 200
            evil = {"Origin": "https://evil.example"}
            assert c.post("/api/auth/logout", headers=evil).status_code == 403
    finally:
        os.environ.pop("FAULTAWARE_FRONTEND_DIST", None)
        importlib.reload(api)
