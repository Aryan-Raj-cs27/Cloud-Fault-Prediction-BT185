# Contributing

This is the BT-185 capstone project of IILM University (guide: Ms. Niharika Chaudhary). These rules keep the results reproducible and honest.

## Workflow
1. Branch from `main` as `feat/<name>`, `fix/<name>` or `exp/<name>`.
2. Use [Conventional Commits](https://www.conventionalcommits.org/): `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `exp:`, `chore:`.
3. Run `make check` (Windows: `.\make.ps1 check`): ruff, strict mypy and the test suite must pass.
4. Open a pull request. CI must be green, and one teammate reviews it.
5. Update `docs/PROGRESS.md` and `CHANGELOG.md`. Record significant choices in `docs/DECISIONS.md` and experiments in `docs/EXPERIMENTS.md`.

## Code
- Python 3.11; type hints and Google-style docstrings on every function; `logging`, not `print`, in `src/`.
- Every hyperparameter, path and threshold lives in `configs/`. Seed all randomness with `faultaware.utils.seed.set_seed`.
- New functions come with unit tests. A bug fix starts with a failing test.
- Never add a dependency without recording the reason in `docs/DECISIONS.md`.

## Data and evaluation rules
- Splits are chronological. Never shuffle across time.
- Fit preprocessing, oversampling, calibration and thresholds on training or validation data only.
- The test split is read once, by `scripts/evaluate.py`.
- Report mean ± std over at least 5 seeds with paired significance tests. Investigate results that look too good before reporting them.
- Never fabricate results or citations.

## Never commit
Raw or processed data, trained weights, `secrets/`, `.env` files, release bundles, or notebooks with large embedded outputs. `.gitignore` covers these.
