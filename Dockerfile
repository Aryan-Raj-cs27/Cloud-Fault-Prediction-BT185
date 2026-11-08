# syntax=docker/dockerfile:1
# FaultAware console: dashboard + API in one image (CPU inference).
# Trained artefacts are NOT baked in; mount ./models, ./reports, ./secrets and
# ./app/demo_samples.json at runtime (see docker-compose.yml and DEPLOYMENT.md).

# ---- 1. build the dashboard ------------------------------------------------------------
FROM node:20-alpine AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- 2. runtime -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    FAULTAWARE_ROOT=/srv FAULTAWARE_FRONTEND_DIST=/srv/frontend/dist
WORKDIR /srv

COPY requirements.txt pyproject.toml README.md ./
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY src ./src
COPY app ./app
COPY configs ./configs
COPY scripts/create_user.py ./scripts/create_user.py
RUN pip install --no-deps -e .
COPY --from=web /web/dist ./frontend/dist

RUN useradd --create-home --uid 10001 appuser && mkdir -p /srv/secrets && chown appuser /srv/secrets
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"
CMD ["uvicorn", "app.api:server", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
