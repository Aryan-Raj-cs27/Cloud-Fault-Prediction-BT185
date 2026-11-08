"""FastAPI service: fault-risk prediction, model analytics and FARS simulation results.

Run:  uvicorn app.api:app --port 8000        (create the account first: scripts/create_user.py)

Public endpoints:
    GET  /health            readiness (no internal details)
    POST /auth/login        {username, password} -> HttpOnly session cookie
Authenticated endpoints:
    POST /auth/logout       revoke the session
    GET  /auth/me           current user
    GET  /model-info        fused models, alpha, thresholds per observation window
    POST /predict           raw HDFS log lines of one block -> risk score + band
    GET  /samples/next      a real held-out TEST block (lines + true label) for live replay
    GET  /metrics           test-set model comparison (from reports/results_summary.json)
    GET  /simulation        FARS vs baselines results + downsampled cluster replay

Environment:
    FAULTAWARE_AUTH_FILE     account file (default secrets/auth.json)
    FAULTAWARE_COOKIE_SECURE 1 behind HTTPS (Secure + __Host- cookie)
    FAULTAWARE_DOCS          1 to expose /docs (off by default)
    CORS_ORIGINS             comma-separated browser origins allowed to call the API
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field

from app.security import (
    SECURITY_HEADERS,
    LoginLimiter,
    Session,
    SessionStore,
    authenticate,
    load_user,
)
from faultaware.serving import RiskService

ROOT = Path(os.getenv("FAULTAWARE_ROOT", Path(__file__).resolve().parents[1]))
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
SAMPLES_FILE = ROOT / "app" / "demo_samples.json"
AUTH_FILE = Path(os.getenv("FAULTAWARE_AUTH_FILE", ROOT / "secrets" / "auth.json"))
COOKIE_SECURE = os.getenv("FAULTAWARE_COOKIE_SECURE", "0") == "1"
COOKIE_NAME = "__Host-fa_session" if COOKIE_SECURE else "fa_session"
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if o.strip()
]
STATE: dict[str, Any] = {}
SESSIONS = SessionStore()
LIMITER = LoginLimiter()


def _tau() -> tuple[float, float]:
    tuned = ROOT / "configs" / "scheduler" / "fars_tuned.yaml"
    base = ROOT / "configs" / "scheduler" / "fars.yaml"
    cfg = OmegaConf.load(tuned if tuned.is_file() else base)
    return float(cfg.tau_low), float(cfg.tau_high)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


@asynccontextmanager
async def lifespan(_: FastAPI):  # type: ignore[no-untyped-def]
    """Load artefacts once at startup (never per request)."""
    try:
        STATE["service"] = RiskService(MODELS_DIR, _tau())
        STATE["error"] = None
    except (FileNotFoundError, KeyError):  # API stays up and reports why (without paths)
        STATE["service"] = None
        STATE["error"] = "Trained models are not available yet. Run `make train eval`."
    STATE["user"] = load_user(AUTH_FILE)
    STATE["samples"] = _read_json(SAMPLES_FILE) or []
    STATE["results"] = _read_json(REPORTS_DIR / "results_summary.json")
    STATE["simulation"] = _read_json(REPORTS_DIR / "simulation_summary.json")
    STATE["replay"] = _read_json(REPORTS_DIR / "simulation_replay.json")
    STATE["rng"] = random.Random(0)
    yield
    STATE.clear()


docs = os.getenv("FAULTAWARE_DOCS", "0") == "1"
app = FastAPI(
    title="BT-185 Fault-Aware Cloud API",
    version="1.1.0",
    lifespan=lifespan,
    description="Early fault prediction on HDFS logs + FARS scheduling results.",
    docs_url="/docs" if docs else None,
    redoc_url=None,
    openapi_url="/openapi.json" if docs else None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Accept"],
)


def _origin_allowed(origin: str, request: Request) -> bool:
    """Same-origin requests, plus explicitly configured development origins.

    A POST is same-origin when the ``Origin`` host equals the ``Host`` header the browser sent.
    A cross-site page cannot forge either header, so this blocks CSRF without listing the
    deployment domain anywhere (the scheme is not compared: TLS ends at the reverse proxy).
    """
    if origin in ALLOWED_ORIGINS:
        return True
    host = request.headers.get("host")
    return bool(host) and urlsplit(origin).netloc.lower() == str(host).lower()


@app.middleware("http")
async def harden(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Reject cross-site state changes and add security headers to every response."""
    origin = request.headers.get("origin")
    if request.method == "POST" and origin and not _origin_allowed(origin, request):
        return JSONResponse({"detail": "origin not allowed"}, status_code=403)
    response = await call_next(request)
    for key, value in SECURITY_HEADERS.items():
        response.headers.setdefault(key, value)
    if COOKIE_SECURE:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


def require_user(request: Request) -> Session:
    """Dependency: the caller must hold a valid session cookie."""
    session = SESSIONS.get(request.cookies.get(COOKIE_NAME))
    if session is None:
        raise HTTPException(status_code=401, detail="authentication required")
    return session


class LoginRequest(BaseModel):
    """Credentials for the single provisioned account."""

    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class PredictRequest(BaseModel):
    """Raw log lines of ONE HDFS block, in any order."""

    model_config = ConfigDict(extra="forbid")
    log_lines: list[str] = Field(
        min_length=1, max_length=5000, description="Raw HDFS log lines mentioning one block"
    )
    window: str | None = Field(
        default=None,
        pattern="^(w60|w10|full)$",
        description="Observation window; default w60 (early warning)",
    )


def _service() -> RiskService:
    service = STATE.get("service")
    if service is None:
        raise HTTPException(status_code=503, detail=STATE.get("error") or "not ready")
    return service


@app.get("/")
def root() -> dict[str, str]:
    """Service banner."""
    return {"name": "BT-185 Fault-Aware Cloud API"}


@app.get("/health")
def health() -> dict[str, Any]:
    """Readiness probe (public: reveals no data or paths)."""
    service = STATE.get("service")
    return {
        "status": "ok" if service else "degraded",
        "models_loaded": sorted(service.windows) if service else [],
        "detail": STATE.get("error"),
        "auth_configured": STATE.get("user") is not None,
    }


@app.post("/auth/login")
def login(req: LoginRequest, request: Request, response: Response) -> dict[str, str]:
    """Verify credentials and set the session cookie."""
    client = request.client.host if request.client else "unknown"
    wait = LIMITER.retry_after(client)
    if wait:
        raise HTTPException(
            status_code=429,
            detail=f"Too many failed attempts. Try again in {(wait + 59) // 60} min.",
            headers={"Retry-After": str(wait)},
        )
    user = STATE.get("user")
    if user is None:
        raise HTTPException(
            status_code=503,
            detail="No account configured. Run scripts/create_user.py on the server.",
        )
    if not authenticate(user, req.username, req.password):
        LIMITER.failure(client)
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    LIMITER.success(client)
    token = SESSIONS.create(user.username)
    response.set_cookie(
        COOKIE_NAME,
        token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="strict",
        path="/",
        max_age=int(SESSIONS.max_age_seconds),
    )
    return {"username": user.username}


@app.post("/auth/logout")
def logout(request: Request, response: Response) -> dict[str, str]:
    """Revoke the current session and clear the cookie."""
    SESSIONS.revoke(request.cookies.get(COOKIE_NAME))
    response.delete_cookie(
        COOKIE_NAME, path="/", secure=COOKIE_SECURE, httponly=True, samesite="strict"
    )
    return {"status": "signed out"}


protected = APIRouter(dependencies=[Depends(require_user)])


@protected.get("/auth/me")
def me(session: Annotated[Session, Depends(require_user)]) -> dict[str, Any]:
    """The signed-in user."""
    return {"username": session.username, "session_started": session.created}


@protected.get("/model-info")
def model_info() -> dict[str, Any]:
    """Fused models, fusion weights and thresholds per observation window."""
    return _service().info()


@protected.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    """Predict the fault risk of one block from its raw log lines."""
    if any(len(line) > 4000 for line in req.log_lines):
        raise HTTPException(status_code=422, detail="log lines must be at most 4000 characters")
    try:
        return _service().predict(req.log_lines, req.window)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@protected.get("/samples/next")
def next_sample(
    label: str | None = Query(default=None, pattern="^(Normal|Anomaly)$"),
) -> dict[str, Any]:
    """Return a real held-out test block (raw lines + ground truth) for live replay."""
    samples = STATE.get("samples") or []
    if label:
        samples = [s for s in samples if s["label"] == label]
    if not samples:
        raise HTTPException(status_code=404, detail="no demo samples; run make demo-samples")
    return STATE["rng"].choice(samples)


@protected.get("/metrics")
def metrics(window: str | None = Query(default=None, pattern="^(w60|w10|full)$")) -> dict[str, Any]:
    """Test-set comparison of all models for one window (default: primary)."""
    results = STATE.get("results")
    if not results:
        return {"models": [], "detail": "run scripts/evaluate.py"}
    tag = window or results["primary_window"]
    if tag not in results["windows"]:
        raise HTTPException(status_code=404, detail=f"unknown window {tag}")
    w = results["windows"][tag]
    rows = [
        {
            "name": r["model"],
            "family": r["family"],
            "pr_auc": _mean(r["test_pr_auc"]),
            "roc_auc": _mean(r["test_roc_auc"]),
            "precision": _mean(r["test_precision"]),
            "recall": _mean(r["test_recall"]),
            "f1": _mean(r["test_f1"]),
            "fpr": _mean(r["test_fpr"]),
            "pr_auc_text": r["test_pr_auc"],
            "f1_text": r["test_f1"],
            "latency_ms": r["latency_ms_per_sample"],
            "confusion_matrix": w["confusion"].get(r["model"]),
        }
        for r in w["table"]
    ]
    return {
        "window": tag,
        "windows": list(results["windows"]),
        "models": rows,
        "best": {
            k: w[k] for k in ("best_supervised", "best_anomaly", "best_classical", "best_deep")
        },
        "fusion": w["fusion"],
        "significance": w["significance"],
        "early_warning": [
            {
                "name": r["model"],
                "val_pr_auc": _mean(r["val_pr_auc"]),
                "test_pr_auc": _mean(r["test_pr_auc"]),
                "test_f1": _mean(r["test_f1"]),
                "test_precision": _mean(r.get("test_precision")),
                "test_recall": _mean(r.get("test_recall")),
                "test_alive": r["test_alive"],
            }
            for r in w.get("early_warning") or []
        ],
        "censoring": results.get("censoring"),
        "lead_time_hours": w.get("lead_time_hours"),
        "shap": w.get("shap"),
        "calibration": w.get("calibration"),
        "data": results.get("data", {}).get("split_counts"),
    }


def _mean(text: Any) -> float | None:
    try:
        return float(str(text).split("±")[0])
    except ValueError:
        return None


@protected.get("/simulation")
def simulation(max_points: int = Query(default=240, ge=10, le=5000)) -> dict[str, Any]:
    """FARS vs baselines summary plus a downsampled per-tick cluster replay."""
    summary = STATE.get("simulation")
    if not summary:
        return {"comparison": [], "detail": "run scripts/run_simulation.py"}
    replay = STATE.get("replay") or {}
    trimmed = {}
    for name, rec in replay.items():
        trace = rec["trace"]
        step = max(1, len(trace) // max_points)
        trimmed[name] = {"metrics": rec["metrics"], "trace": trace[::step]}
    return {**summary, "replay": trimmed}


app.include_router(protected)


# ---------------------------------------------------------------------------------------------
# Production entry point: one origin serves the built dashboard at "/" and this API at "/api".
#   uvicorn app.api:server --host 0.0.0.0 --port 8000
# ---------------------------------------------------------------------------------------------
FRONTEND_DIST = Path(os.getenv("FAULTAWARE_FRONTEND_DIST", ROOT / "frontend" / "dist"))
PAGE_CSP = (
    "default-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; base-uri 'self'; form-action 'self'; "
    "frame-ancestors 'none'; object-src 'none'"
)

server = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
server.mount("/api", app)


@server.middleware("http")
async def page_headers(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Security headers for the static dashboard (API responses keep their stricter set)."""
    response = await call_next(request)
    if not request.url.path.startswith("/api"):
        for key, value in SECURITY_HEADERS.items():
            if key not in {"Content-Security-Policy", "Cache-Control"}:
                response.headers.setdefault(key, value)
        response.headers["Content-Security-Policy"] = PAGE_CSP
        cacheable = request.url.path.startswith("/assets/")
        response.headers["Cache-Control"] = (
            "public, max-age=31536000, immutable" if cacheable else "no-cache"
        )
        if COOKIE_SECURE:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


if FRONTEND_DIST.is_dir():
    from fastapi.staticfiles import StaticFiles

    server.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="dashboard")
