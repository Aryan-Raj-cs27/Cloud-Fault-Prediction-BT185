# Progress — BT-185

Last updated: 2026-11-08

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
- [ ] Build `docker compose up` once on a machine with Docker (not installed on the dev laptop; Dockerfile is untested locally).
- [ ] Optional extension: DQN/PPO scheduler (CLAUDE.md §2.1 step 5).
- [ ] Optional: second dataset (BGL or Google trace) for cross-dataset validation.
