# BT-185 — one-command workflow (CLAUDE.md §10.3). On Windows use .\make.ps1 <target>.
PY ?= .venv/bin/python
# GPU wheels by default; use TORCH_INDEX=https://download.pytorch.org/whl/cpu on CPU-only hosts.
TORCH_INDEX ?= https://download.pytorch.org/whl/cu128
ifeq ($(OS),Windows_NT)
PY := .venv/Scripts/python.exe
endif

.PHONY: setup data tune train eval simulate demo-samples reproduce test test-all check lint demo api frontend build-web serve user package clean

setup:            ## create venv, install deps + pre-commit hooks
	uv venv .venv --python 3.11
	uv pip install --python $(PY) torch --index-url $(TORCH_INDEX)
	uv pip install --python $(PY) -r requirements.txt
	uv pip install --python $(PY) -r requirements-dev.txt
	uv pip install --python $(PY) -e ".[dev]" --no-deps
	$(PY) -m pre_commit install

data:             ## download (if missing) + parse + sessionise + split
	$(PY) scripts/download_data.py
	$(PY) scripts/make_dataset.py

tune:             ## Optuna search on validation (writes configs/model/tuned.yaml)
	$(PY) scripts/tune.py

train:            ## all models x windows x 5 seeds (resumable)
	$(PY) scripts/train.py --skip-existing

eval:             ## single final evaluation on TEST + figures/tables + fusion
	$(PY) scripts/evaluate.py

simulate:         ## FARS vs baselines, sanity checks, ablations
	$(PY) scripts/run_simulation.py

demo-samples:     ## extract real held-out blocks for the dashboard replay
	$(PY) scripts/make_demo_samples.py

reproduce: data tune train eval simulate demo-samples  ## every reported number, end to end

lint:
	$(PY) -m ruff check src scripts app tests
	$(PY) -m ruff format --check src scripts app tests

test:             ## fast tests (CPU, < 3 min)
	$(PY) -m pytest -m "not slow" --cov --cov-fail-under=80

test-all:
	$(PY) -m pytest --cov

check: lint       ## lint + type-check + tests (must pass before any commit)
	$(PY) -m mypy src app
	$(MAKE) test

api:              ## development API on 127.0.0.1:8000 (dashboard proxies /api to it)
	$(PY) -m uvicorn app.api:app --host 127.0.0.1 --port 8000

frontend:         ## development dashboard on http://localhost:5173
	cd frontend && npm install && npm run dev

build-web:        ## production build of the dashboard (frontend/dist)
	cd frontend && npm ci && npm run build

serve: build-web  ## production: dashboard + API on one origin, http://localhost:8000
	$(PY) -m uvicorn app.api:server --host 0.0.0.0 --port 8000

user:             ## create/replace the single console account (prompts for password)
	$(PY) scripts/create_user.py --username admin

package:          ## bundle serving artefacts for a GitHub Release
	$(PY) scripts/package_artifacts.py

demo:             ## full stack behind HTTPS (docker + Caddy)
	docker compose up -d --build

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
