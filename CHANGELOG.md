# Changelog

## [Unreleased]
### Added
- `docs/DEPLOYMENT.md`: Oracle Cloud Always Free (arm64) deployment with Docker, Caddy and DuckDNS;
  Cloudflare quick tunnel as a demo-day backup; troubleshooting and configuration reference.
- README "Fresh machine setup" section (tools, venv, release bundle, dataset, account, run modes).
- `requirements-dev.txt` with pinned lint/type/test tools; CI `docker` job builds `linux/amd64` + `linux/arm64`.

### Fixed
- `src/faultaware/models/` was excluded by an unanchored `models/` ignore rule (fresh clones crashed).
- CI lint job never installed ruff/mypy/pytest.
- Release `.sha256` sidecar is written with LF so `sha256sum -c` works on Linux.

## [1.1.0] — 2026-11-08
### Added
- Strict early-warning evaluation (`faultaware.eval.metrics.alive_mask`, `evaluate_early_warning`; `reports/tables/early_warning_*.md`) after finding right-censoring at the trace end (D-013). Lead time is now computed over still-running true positives.
- Login-protected console: scrypt password hash, HttpOnly SameSite=Strict session cookies, login rate limiting, origin checks, security headers, single account (`scripts/create_user.py`), no sign-up.
- Production entry point `app.api:server` (dashboard + API on one origin), Docker + Caddy (automatic HTTPS), release bundle with SHA-256 (`scripts/package_artifacts.py`, `scripts/install_artifacts.py`).
- Console v2: Overview, Live inference, Models (incl. strict early-warning table), FARS scheduling (risk heat-map replay), Data & method; single light/dark toggle, collapsible icon sidebar, motion with reduced-motion support.
- `docs/report/report.md`, `docs/report/references.bib`, `docs/PROJECT_GUIDE.md`, `docs/EXPERIMENTS.md`.

- FARS evaluated with two risk-score sources on the same seeds: test period (primary) and validation period (pessimistic) (`scheduler_score_sources.md`, D-014).
- LICENSE (MIT), NOTICE (dataset terms), CITATION.cff, SECURITY.md, CONTRIBUTING.md.

### Changed
- Simulator precursor scores use only sessions still alive after the observation window (genuine predictions).
- Headline results everywhere (README, report, guide, console Overview) are the strict early-warning numbers. The standard 0.963 test PR-AUC is presented as an evaluation bias caused by right-censoring.
- Documentation is versioned in git; only CLAUDE.md stays local. Repository public with the guide's approval (D-015).

### Fixed
- Production login was rejected with 403: the origin check only allowed configured dev origins. Same-origin POSTs (Origin host = Host header) are now accepted; foreign origins are still rejected (regression test added).
- `.gitignore` inline comments had disabled the `secrets/` and `models/` rules.
- Mobile layout: wide tables no longer push panels beyond the viewport (`min-w-0` on panels).


## [1.0.0] — 2026-11-07
### Added
- `src/faultaware` package: streaming HDFS parser with masked templates, block sessions, chronological split with gap, pandera schemas.
- Leakage-safe features (train-only vocabulary, observation windows 10 s / 60 s / full).
- Model ladder behind one `FaultPredictor` interface: always-0, random, WARN rule, logistic regression, random forest (+SMOTE), XGBoost, LSTM (+focal loss), GRU, causal CNN-LSTM, Transformer, Isolation Forest, LSTM-autoencoder.
- Training with early stopping on validation PR-AUC, LR scheduling, gradient clipping; Optuna tuning on validation (`scripts/tune.py`).
- Calibration (isotonic) + risk fusion `r = α·p + (1−α)·a` with α tuned on validation.
- FARS scheduler, six baselines, SimPy simulator with Weibull faults, degradation precursors, checkpointing and migration; oracle/random sanity checks; ablations (anomaly detector, calibration, horizon, thresholds, noise).
- Evaluation: test metrics mean ± std over 5 seeds, paired Wilcoxon/t-tests, bootstrap CIs, SHAP, reliability, lead time; figures and tables in `reports/`.
- FastAPI service (`app/api.py`) and React dashboard rewired to real data (live replay of held-out blocks, analytics, FARS page).
- 104 tests (unit, leakage, data validation, simulator sanity, API); ruff + mypy strict; CI; Dockerfile + compose; Makefile + `make.ps1`.

### Changed
- Prototype RF (2 features, random split, 818 blocks) and Keras CNN-LSTM replaced; prototype notebooks moved to `notebooks/archive/` (random-split results there are not valid, see D-010).
- `backend/` replaced by `app/`.
