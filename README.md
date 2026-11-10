# FaultAware — Fault-Aware ML/DL Algorithm for Cloud Architecture (BT-185)

**Guide:** Ms. Niharika Chaudhary · **Team:** Aryan Raj (CS-2341005), Dheeraj Kumar (CS-2341062), Navneet Kumar (CS-2341055), Priyansh Kumar (CS-2341080) · IILM University

FaultAware tries to predict storage-block faults in a Hadoop (HDFS) cluster **from only the first 60 seconds of a block's life**. When it catches a fault early, the warning comes a median of 26 minutes before the block's session ends. It fuses a calibrated supervised model with an unsupervised anomaly detector into one risk score per node. **FARS** (Fault-Aware Risk-based Scheduler) then uses those scores to steer work away from nodes that are about to fail: it avoids placing new tasks on them, migrates running tasks off them, and checkpoints more often. FARS is compared with six fault-unaware and reactive schedulers in a discrete-event cloud simulator, with statistical tests over 20 paired seeds.

```
HDFS logs ─► parse & mask ─► block sessions ─► chronological split (+gap) ─► features (first W s only)
   │                                                                              │
   │      ┌──────────── supervised: LR · RF · XGBoost · LSTM · GRU · CNN-LSTM · Transformer
   │      ├──────────── unsupervised: Isolation Forest · LSTM autoencoder
   │      ▼
   └─► calibrated fused risk r = α·p + (1−α)·a ─► FARS (place / migrate / checkpoint) ─► SimPy cluster
                                                   └─► FastAPI + React console (login-protected)
```

## Results
Held-out data, chronological split. Model scores are mean ± std over 5 seeds; scheduler scores are over 20 paired seeds. All values are generated into `reports/tables/`.

### Early-warning prediction (the headline)
A warning only counts as early if the block is **still running when the 60 s window closes**: its fault has not happened yet. This "strict" score is the honest measure of prediction:

| Model | Strict PR-AUC, test period | Strict PR-AUC, validation period |
|---|---|---|
| Rule: any WARN line | 0.294 | 0.063 |
| Logistic regression | 0.750 | 0.155 |
| XGBoost (tuned) | **0.792 ± 0.012** | 0.159 |
| LSTM | 0.777 ± 0.026 | 0.147 |
| Transformer | 0.771 ± 0.021 | 0.151 |
| LSTM autoencoder (no labels) | 0.495 | 0.120 |
| Fused calibrated risk (used by FARS) | 0.742 ± 0.007 | 0.167 |
| *Chance (fault rate among still-running blocks)* | *0.005* | *0.034* |

- **Test period:** 39,977 still-running blocks, 189 faults. The fused model flags 148 of the 189, a median of 26 min (IQR 10–41 min) before the session ends.
- **Validation period:** faults are barely predictable at 60 s, at 4–5× chance. Early-warning skill depends strongly on the period.

**Why some reports show 0.963 for the same task, and why we don't headline it.** The standard protocol scores every test block and gives a test PR-AUC of 0.963 ± 0.003 (XGBoost and LSTM). That number is an **evaluation bias**: the trace ends 2 h into the test period, so 80 % of test faults finish *inside* the 60 s window (median lifetime 4 s). The model then sees the whole fault rather than predicting it. Validation, where this censoring is much weaker, gives about 0.42 under the same protocol. Full tables for both protocols are in `reports/tables/prediction_*.md` and `early_warning_*.md` (decision record D-013).

### FARS vs six baselines (simulation)
FARS is the same in both rows (same τ, workloads and faults); only the source of the node risk scores changes. Both sources use still-running blocks only.

| Risk scores drawn from | Task interruptions | Lost node-minutes (all costs) | SLA violations | p vs every baseline |
|---|---|---|---|---|
| Best baseline (least-loaded / round-robin) | 129.9 | 7,638 | 1.0 % | — |
| FARS, test-period scores | 47.2 (**−64 %**) | 6,512 (**−15 %**) | 0.4 % | ≤ 1e-4 |
| FARS, validation-period scores (pessimistic) | 117.4 (**−10 %**) | 7,515 (**−1.6 %**) | 0.9 % | interruptions ≤ 6e-4; lost time 0.0006–0.022 |

**What else the simulation shows:**
- With weak (validation-period) scores, FARS still never does worse than the baselines, but the gain is small.
- An oracle predictor reaches 24.1 interruptions, and random scores are far worse than every baseline.
- Without calibration, FARS loses more node-time than the baselines.
- With 10 % of the test-period scores replaced by noise, the net benefit disappears.

## Quick start (development)
Requirements: Python 3.11 (via [uv](https://docs.astral.sh/uv/)), Node.js 20+, optional NVIDIA GPU.

```powershell
.\make.ps1 setup          # venv + dependencies (Linux/macOS: make setup)
.\make.ps1 reproduce      # data -> tune -> train -> eval -> simulate (several GPU hours)
.\make.ps1 user           # create the single login account (no sign-up exists)
.\make.ps1 api            # terminal 1: API on 127.0.0.1:8000
.\make.ps1 frontend       # terminal 2: console on http://localhost:5173
```

Progress while training: `python scripts/progress.py`. Every target is listed by running `.\make.ps1` with no arguments.

## Fresh machine setup
Use this on a newly installed PC. It runs the console from the published models without retraining
(retraining needs `make reproduce` and several GPU hours).

**1. Install the tools.** On Windows, use `winget` from a normal PowerShell:
```powershell
winget install --id Git.Git -e
winget install --id astral-sh.uv -e --scope user
winget install --id OpenJS.NodeJS.LTS -e
winget install --id GitHub.cli -e                 # optional: releases, PRs, CI status
winget install --id Docker.DockerDesktop -e       # optional: only for the Docker/HTTPS stack
uv python install 3.11                            # uv manages Python 3.11; no system install needed
```
Open a new terminal so `PATH` updates. Docker Desktop also needs WSL 2: run `wsl --install --no-distribution`
in an **administrator** terminal, restart, and make sure virtualisation is enabled in the BIOS/UEFI.
Linux/macOS: install git, [uv](https://docs.astral.sh/uv/), Node.js 20+ and (optionally) Docker with
your package manager. Use `make` wherever these steps say `.\make.ps1`.

**2. Clone and install dependencies:**
```powershell
git clone https://github.com/Aryan-Raj-cs27/Cloud-Fault-Prediction-BT185.git
cd Cloud-Fault-Prediction-BT185
.\make.ps1 setup          # .venv with Python 3.11, CUDA torch, all deps, pre-commit hook
.\make.ps1 check          # optional: lint + strict mypy + tests (~3 min)
```
On a machine without an NVIDIA GPU, use `make setup TORCH_INDEX=https://download.pytorch.org/whl/cpu`.
If git reports *dubious ownership* (common for an existing folder after a Windows reset), run
`git config --global --add safe.directory <full-path-to-repo>`.

**3. Get the trained models.** Download `faultaware-artifacts.zip` and `faultaware-artifacts.zip.sha256`
from the GitHub Release (`v1.1.0`) into the repository folder, then run:
```powershell
.venv\Scripts\python.exe scripts\install_artifacts.py faultaware-artifacts.zip   # verifies SHA-256
```

**4. Dataset (only to retrain or re-evaluate).** Serving the console doesn't need it.
`.\make.ps1 data` downloads Loghub HDFS_v1 (~1.5 GB of logs) and rebuilds `data/processed/`.

**5. Create the login account:** `.\make.ps1 user` (prompts for a password of at least 12 characters).

**6. Run it, with one of these:**
| Mode | Command | URL |
|---|---|---|
| Development (hot reload) | `.\make.ps1 api` + `.\make.ps1 frontend` in two terminals | http://localhost:5173 |
| Production, no Docker | `.\make.ps1 serve` | http://localhost:8000 |
| Production, Docker + HTTPS (tested 2026-11-09) | `docker compose up -d --build` | https://localhost |

## Deploying from this repository
For a public server, see **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**. It covers an Oracle Cloud Always
Free arm64 VM with DuckDNS and automatic HTTPS, plus a Cloudflare quick tunnel as a demo-day backup.

Trained weights, data and credentials are **never committed** (see `.gitignore`). A deployment combines the repository with a release bundle.

**1. On the training machine** (after `make reproduce`):
```bash
python scripts/package_artifacts.py      # -> release/faultaware-artifacts.zip + .sha256
```
Upload both files to a **GitHub Release** (or copy them to the server with `scp`).

**2. On the server:**
```bash
git clone <repo-url> faultaware && cd faultaware
python3 scripts/install_artifacts.py faultaware-artifacts.zip      # verifies SHA-256, rejects unsafe paths
python3 scripts/create_user.py --username admin                    # prompts for a password (≥ 12 chars)
DOMAIN=console.example.org docker compose up -d --build            # HTTPS via Caddy (Let's Encrypt)
```
Use `DOMAIN=localhost` to test locally with Caddy's own certificate authority. This stack was tested from a fresh clone on 2026-11-09. On a Linux host, also run `sudo chown 10001:10001 secrets/auth.json` after creating the account: the container runs as uid 10001 (see `docs/DEPLOYMENT.md`). Without Docker, run `make serve`: it builds the dashboard and serves the dashboard and API from one process at `http://<host>:8000`. Put it behind an HTTPS reverse proxy and set `FAULTAWARE_COOKIE_SECURE=1`.

### Security model
| Concern | Implementation |
|---|---|
| Transport | TLS terminated by Caddy (automatic certificates), HSTS when `FAULTAWARE_COOKIE_SECURE=1` |
| Passwords | scrypt (N=2¹⁵, r=8, p=1, 16-byte salt); plaintext is never stored or logged |
| Accounts | exactly one, created on the server with `scripts/create_user.py`; no sign-up endpoint exists |
| Sessions | 256-bit random token in an `HttpOnly`, `SameSite=Strict` (`Secure`, `__Host-` over HTTPS) cookie; only its SHA-256 is kept server-side; 1 h idle and 8 h absolute expiry; revoked on logout |
| Brute force | 5 failures in 15 min lock sign-in from that client for 15 min (HTTP 429) |
| Enumeration | identical error message and equal-cost hashing for unknown usernames |
| Cross-site | same-origin API under `/api`; POSTs with a foreign `Origin` are rejected; CORS restricted to configured origins |
| Headers | CSP (API: `default-src 'none'`), `X-Frame-Options: DENY`, `nosniff`, `Referrer-Policy: no-referrer`, COOP/CORP, `Cache-Control: no-store` |
| Input limits | ≤ 5000 log lines × 4000 chars per request; strict schemas (`extra="forbid"`) |
| Surface | API docs disabled unless `FAULTAWARE_DOCS=1`; container runs as non-root on a read-only filesystem |
| Artefacts | release bundle verified by SHA-256; installer rejects path traversal |

To change the password, run `scripts/create_user.py` again and restart the service.

## Quality gates
`make check` runs ruff, strict mypy on `src/` and `app/`, and pytest: 126 tests covering unit, leakage, data validation, simulator sanity, API and security. Line coverage is 93 %; CI enforces at least 80 % (`.github/workflows/ci.yml`).

## Repository layout
`src/faultaware/` library (data, features, models, training, fusion, scheduler, simulator, eval) · `scripts/` command-line entry points · `configs/` every hyperparameter · `app/` API and security · `frontend/` React console · `tests/` · `reports/` generated tables and figures · `deploy/` Caddy config · `notebooks/01_eda.ipynb` exploration.

## Data
Loghub HDFS_v1 (Xu et al., SOSP 2009; redistributed by Loghub). Download with `python scripts/download_data.py`. Research use; cite Loghub and Xu et al.
