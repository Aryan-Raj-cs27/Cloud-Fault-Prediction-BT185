# Experiment log — BT-185

Every number in the report comes from one of the runs below and can be regenerated with the command shown. Run records (git commit, resolved config, seed, dataset version, metrics, artefact paths) are written to `runs/<id>/run.json`.

Hardware: Windows 11 laptop, NVIDIA RTX 4050 (6 GB), Python 3.11, torch 2.11 (CUDA 12.8), scikit-learn 1.9.1, xgboost 3.2.0.

---

## E-01 Data preparation — 2026-11-06
- **Command:** `make data` (`scripts/download_data.py`, `scripts/make_dataset.py`)
- **Result:** 11,175,629 log lines → 54 templates; 575,061 block sessions (16,838 faulty, 2.93 %). Chronological split 70/15/15 with a 30-min gap: train 402,549 (3.26 %), val 68,954 (3.45 %), test 70,185 (1.31 %); 33,373 blocks dropped in the gaps.
- **Checks:** pandera schemas and leakage tests pass (`tests/unit/test_data.py`, `tests/unit/test_leakage.py`).

## E-02 Hyperparameter search — 2026-11-06
- **Command:** `make tune` (`scripts/tune.py`), Optuna TPE, objective = validation PR-AUC at W = 60 s, seed 42.
- **Budget:** XGBoost 40 trials; random forest 12 trials (each RF trial is slow).
- **Result** (`configs/model/tuned.yaml`): XGBoost val PR-AUC 0.420 (max_depth 8, lr 0.197, 300 trees); RF val PR-AUC 0.425 (200 trees, max_depth 24, min_samples_leaf 20). DL models use the defaults in `configs/model/models.yaml`; no search was run for them (time budget). This is stated as a limitation.

## E-03 Model ladder — 2026-11-06 → 2026-11-08
- **Command:** `make train` (`scripts/train.py --skip-existing`)
- **Runs:** 14 models × 3 windows (full, 60 s, 10 s) × 5 seeds (0–4) = 210 runs. Early stopping on validation PR-AUC (patience 4, max 25 epochs), AdamW lr 2e-3, batch 2048, gradient clipping 1.0.
- **Outputs:** `models/preds/<window>/<model>/seed<k>.npz` (val + test scores, validation threshold), `models/artefacts/<window>/<model>/` (seed 0 weights).

## E-04 Final evaluation on the test split — 2026-11-08
- **Command:** `make eval` (`scripts/evaluate.py`). The test split is scored here and nowhere else.
- **Tables:** `reports/tables/prediction_{full,w60,w10}.md`, `early_warning_{w60,w10}.md`, `significance_{full,w60,w10}.md`.
- **Results (W = 60 s, mean ± std over 5 seeds).** The *strict* column is the headline. The standard test PR-AUC is inflated by right-censoring at the trace end (D-013, D-014):

| Model | Test PR-AUC | Test F1 | Strict early-warning PR-AUC* | Validation PR-AUC |
|---|---|---|---|---|
| Rule: any WARN line | 0.108 | 0.026 | 0.294 | 0.067 |
| Logistic regression | 0.953 ± 0.000 | 0.956 | 0.750 | 0.418 |
| Random forest | 0.945 ± 0.008 | 0.956 | 0.746 | 0.428 |
| XGBoost | 0.963 ± 0.003 | 0.957 | **0.792** | 0.421 |
| LSTM | 0.963 ± 0.004 | 0.956 | 0.777 | 0.407 |
| GRU | 0.936 ± 0.018 | 0.950 | 0.758 | 0.408 |
| CNN-LSTM | 0.958 ± 0.005 | 0.957 | 0.755 | 0.407 |
| Transformer | 0.959 ± 0.003 | 0.956 | 0.771 | 0.414 |
| Isolation Forest | 0.301 ± 0.018 | 0.429 | 0.175 | 0.136 |
| LSTM autoencoder | 0.782 ± 0.091 | 0.833 | 0.495 | 0.345 |
| Fused risk (RF + LSTM-AE, α = 0.9) | 0.940 ± 0.008 | 0.955 | 0.742 | 0.428 |

\* Only blocks still running when the 60 s window closes (39,977 test blocks, 189 faults); see D-013.

- **Calibration (W = 60 s, seed 0):** Brier score 0.127 (raw RF) → 0.0019 (isotonic) → 0.0029 (fused).
- **Lead time:** 881 test faults correctly flagged; 148 of them were still running after the window, and for those the warning came a median of 26 min (IQR 10–41 min) before the block session ended.
- **Significance (paired over 5 seeds):** Transformer vs RF, test PR-AUC +0.015 (t-test p = 0.015, Wilcoxon p = 0.0625, the minimum possible with n = 5). Fusion vs RF, −0.005 (p = 0.29): fusion did **not** improve ranking quality at 60 s, but it did improve it at 10 s (+0.004, not significant).
- **Conclusion:** at 60 s, every learned supervised model is within ≈ 0.03 PR-AUC of the others; DL does not clearly beat tuned trees (CLAUDE.md §7.3: reported honestly). Validation and test disagree strongly because of right-censoring at the trace end (D-013).

## E-05 FARS simulation — 2026-11-08
- **Command:** `make simulate` (`scripts/run_simulation.py`), about 10 minutes on 8 CPU cores.
- **Setup:** 40 nodes, 5,000 tasks, Weibull node failures (shape 1.5, scale 1,800 min), 80 % with a 15–60 min hidden precursor. Each tick (5 min) the scheduler sees a risk score drawn from the fused model's real score distribution for still-running blocks at W = 60 s. τ was tuned on 5 validation seeds using validation scores: τ_low = 0.3, τ_high = 0.4, the grid minimum. The validation surface is nearly flat (7,462–7,612 lost node-min over 27 pairs) because validation-period scores carry little signal. The final comparison uses 20 different test seeds with test scores, and every scheduler sees identical workloads and faults per seed.
- **Result (mean ± std, 20 seeds):**

| Scheduler | Task interruptions | Wasted compute (min) | Lost node-min | SLA violation rate | Migrations |
|---|---|---|---|---|---|
| Random | 131.4 ± 16.5 | 1,859 ± 252 | 7,708 ± 262 | 0.010 | 0 |
| Round-robin | 130.5 ± 16.4 | 1,789 ± 247 | 7,638 ± 248 | 0.010 | 0 |
| First-fit | 132.9 ± 14.4 | 1,832 ± 213 | 7,681 ± 207 | 0.010 | 0 |
| Best-fit | 132.8 ± 17.6 | 1,818 ± 273 | 7,667 ± 295 | 0.010 | 0 |
| Least-loaded | 129.9 ± 15.2 | 1,791 ± 242 | 7,640 ± 269 | 0.010 | 0 |
| Reactive | 131.1 ± 18.0 | 1,845 ± 237 | 7,694 ± 236 | 0.010 | 0 |
| **FARS** | **47.2 ± 13.4** | **651 ± 215** | **6,512 ± 209** | **0.004** | 97 |

- **Paired tests vs each baseline:** interruptions −64 %, wasted compute −64 %, SLA violations −60 %, lost node-minutes −15 %; Wilcoxon p ≤ 1e-4 for every metric and baseline (`reports/tables/scheduler_significance.md`). Mean response time −0.5 % (negligible).
- **Sanity checks:** with an oracle signal, FARS reaches 24.1 interruptions (−81 %). With a random signal it is far worse than least-loaded (28,843 lost node-min vs 7,640) because it migrates and checkpoints constantly. Both behave as CLAUDE.md §9 requires.
- **Ablations** (τ re-tuned on validation for each): (a) without the anomaly detector there is no measurable change (6,515 vs 6,512 lost node-min). (b) Without calibration, interruptions are fewer (38.9) but lost node-min is higher (8,228, worse than the baselines) because of more checkpoints and migrations. (c) With a 10 s window, 90.7 interruptions and 7,112 lost node-min. The full session, which is detection and not prediction, is an upper bound: 24.3 interruptions and 6,256 lost node-min.
- **Noise sensitivity:** replacing a share η of the scores with U(0, 1): at η = 0.1 interruptions are unchanged (45.2), but lost node-min (8,128) already exceeds the baselines because false alarms trigger migrations. FARS depends on a low false-alarm rate.

## Reproduce everything
```bash
make reproduce        # data -> tune -> train -> eval -> simulate -> demo-samples
```
Windows: `.\make.ps1 reproduce`. Steps are resumable (`train --skip-existing`).

## E-06 FARS risk-score source (D-014) — 2026-11-08
- **Command:** `make simulate` (step 2b of `scripts/run_simulation.py`); same 20 test seeds, τ = (0.3, 0.4), identical workloads and faults; baselines reused (they ignore risk; the signal has its own random stream).
- **Result** (`reports/tables/scheduler_score_sources.md`, `scheduler_significance_valscores.md`):

| Risk scores drawn from | Interruptions | Lost node-min | SLA violations | Migrations |
|---|---|---|---|---|
| Best baseline | 129.9 | 7,638 | 1.0 % | 0 |
| Test period, still-running blocks | 47.2 ± 13.4 (−64 %) | 6,512 ± 209 (−15 %) | 0.4 % | 97 |
| Validation period, still-running blocks | 117.4 ± 15.9 (−10 %) | 7,515 ± 203 (−1.6 %) | 0.9 % | 16 |

- **Significance (validation-period scores vs each baseline):** interruptions p ≤ 6e-4; lost node-min p = 6e-4 to 0.022; SLA violations vs least-loaded −10 %, p = 0.021.
- **Conclusion:** the gain shrinks sharply with the weaker validation-period scores, but FARS remains significantly better than every baseline. Both rows are reported as the headline range.
