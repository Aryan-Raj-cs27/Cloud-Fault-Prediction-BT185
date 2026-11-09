# Progress — BT-185

Last updated: 2026-11-09

## ▶ Resume here (2026-11-09, work paused for a PC reset)

**Done and pushed on `feat/complete-pipeline`:**
- [x] **Docker tested** on Windows + Docker Desktop: `docker compose up -d --build` from a *fresh clone*,
  installed release bundle, created admin → `https://localhost/api/health` OK (3 models, auth configured),
  dashboard 200, HTTP→HTTPS 308, wrong password 401, login 200 with `__Host-fa_session` cookie,
  `/api/model-info`, `/metrics`, `/simulation`, `/samples/next` all 200.
- [x] **Fix:** `.gitignore` rule `models/` also hid `src/faultaware/models/` — the model package was never
  in git (fresh clones crashed). Rules now anchored (`/models/`, `/runs/`, `/release/`); package committed;
  regression test `tests/unit/test_repo_hygiene.py`.
- [x] **Fix:** CI was red (Lint): `pip install -e ".[dev]" --no-deps` never installed ruff/mypy/pytest.
  Added pinned `requirements-dev.txt`, used by CI, `Makefile` and `make.ps1 setup`.
- [x] **Fix:** release `.sha256` was written with CRLF on Windows (`sha256sum -c` failed on Linux); now LF + test.
- [x] Release bundle rebuilt (old zip had a stale `reports/simulation_summary.json`):
  `release/faultaware-artifacts.zip`, sha256 `ac46a83e990285f22091639e8010fdae4c429cb356eba7bb71f7bc060fff341a`.
  **`release/` is not in git — upload it to the GitHub Release or back it up before the reset.**
- [x] CI `docker` job now builds `linux/amd64` + `linux/arm64` with buildx (on `main` / manual runs).
- [x] Local `make check`: ruff, strict mypy, 126 tests passed, coverage 93 %. CI replay in a Linux container
  from a fresh clone: lint + format + mypy green, coverage 93 %.

**Verified findings to put into `docs/DEPLOYMENT.md` (not written yet):**
- On a **Linux** host, `scripts/create_user.py` writes `secrets/auth.json` as mode 0600 owned by the host
  user; the container (uid 10001) then crash-loops with `PermissionError`. Fix (tested):
  `sudo chown 10001:10001 secrets/auth.json`. Windows/Docker Desktop hides this.
- Cloudflare quick tunnel works with the same-origin check: login with tunnel `Host` + `Origin` → 200,
  foreign Origin → 403 (tested with simulated headers). Use the no-Docker server (`make serve`, port 8000,
  `FAULTAWARE_COOKIE_SECURE=1`) + `cloudflared tunnel --url http://localhost:8000`. Tunnelling to the Caddy
  stack instead would fail login (Host `localhost` ≠ Origin `*.trycloudflare.com`).

**Still pending (from the last request):**
- [ ] CI replay: the final pytest pass/fail line was cut off (stopped before the rerun finished). Check the
  GitHub Actions run for the pushed head commit — it should now be green.
- [ ] arm64 image build: started locally, **stopped before finishing** — not verified. Run
  `docker buildx build --platform linux/arm64 .` or trigger the CI workflow manually (`workflow_dispatch`).
- [ ] Write `docs/DEPLOYMENT.md`: Oracle Cloud Always Free A1 (aarch64) VM — create VM, open 80/443
  (VCN security list + `iptables`), install Docker, clone, download + install release bundle, create admin
  + `chown 10001`, DuckDNS domain, `DOMAIN=<name>.duckdns.org docker compose up -d --build`; plus a short
  Cloudflare quick-tunnel section (demo-day backup). Don't deploy anything.
- [ ] README: "fresh machine setup" section (install, dataset, release bundle, admin user, run with/without
  Docker) + mark Docker as tested.
- [ ] GitHub Release `v1.0.0` with `release/faultaware-artifacts.zip` + `.sha256` (needs `gh` or the web UI;
  the assistant could not use the stored GitHub token).
- [ ] Open the PR `feat/complete-pipeline` → `main` with a description of all changes; confirm CI green.
- [ ] Final summary: done / left for the team / local files not in git.

| Phase | Status | Evidence |
|---|---|---|
| P0 Setup (repo, CI, configs) | ✅ done | `pyproject.toml`, `configs/`, `.github/workflows/ci.yml`, `Makefile`, `make.ps1` |
| P1 Problem formulation | ✅ done | label/horizon definition in `docs/DECISIONS.md` D-003 |
| P2 Data pipeline + EDA | ✅ done | `scripts/make_dataset.py`, `notebooks/01_eda.ipynb`, `tests/unit/test_data.py`, `tests/unit/test_leakage.py` |
| P3 ML baselines | ✅ done | `reports/tables/prediction_*.md`, `early_warning_*.md` (strict, D-013) |
| P4 DL models | ✅ done | LSTM, GRU, CNN-LSTM, Transformer; `reports/figures/training_curves.png` |
| P5 Anomaly detection + fusion + calibration | ✅ done | Isolation Forest, LSTM-AE, `fusion/risk.py`, `reports/figures/reliability.png` |
| P6 Simulator + FARS + baselines | ✅ done | `src/faultaware/simulator`, `scheduler/fars.py`, `tests/integration/test_simulator.py` |
| P7 Experiments, ablations, stats | ✅ done | `reports/tables/scheduler_*.md`, `ablations.md`, `significance_*.md` |
| P8 Demo app | ✅ done | `app/api.py` (login-protected), `frontend/` console v2, `Dockerfile`, `docker-compose.yml` + Caddy, release bundle scripts |
| P9 Report | ✅ draft | `docs/report/report.md`, `docs/report/references.bib` (verify!), `docs/PROJECT_GUIDE.md`; slides + demo video: team |

## Final run (2026-11-08)
- 210/210 training runs, `scripts/evaluate.py`, `scripts/run_simulation.py` (20 test seeds) complete; numbers in `docs/EXPERIMENTS.md`.
- Release bundle built and install round-trip verified (`release/faultaware-artifacts.zip`).
- `make check`: ruff, strict mypy, 123 tests, coverage 93 %. Production server (`app.api:server`) checked in a real browser.

## Open items for the team
- [ ] Verify every reference in `docs/report/references.bib` on Google Scholar / IEEE Xplore (CLAUDE.md §13.1).
- [ ] Record the 2–3 min demo video (backup for the viva).
- [ ] Change the console password (`.\make.ps1 user`): the team lead sets it in their own terminal.
- [x] Docs are versioned in git (only CLAUDE.md is ignored); repository public with the guide's approval (D-015).
- [x] `docker compose up` built and tested (2026-11-09, fresh clone, login over HTTPS) — see "Resume here".
- [ ] Optional extension: DQN/PPO scheduler (CLAUDE.md §2.1 step 5).
- [ ] Optional: second dataset (BGL or Google trace) for cross-dataset validation.
