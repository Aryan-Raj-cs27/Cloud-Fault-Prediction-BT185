# Architecture Decision Records — BT-185

Each entry: context → options → decision → reason. Changes to a rule in CLAUDE.md are recorded here first.

---

## D-001 Primary dataset: Loghub HDFS_v1 (not Google Cluster Trace)
- **Context.** CLAUDE.md §3 recommends Google Cluster Trace 2011/2019 as primary but allows Loghub (HDFS/BGL). The team had already downloaded HDFS_v1 (1.5 GB raw log, 575,061 expert-labelled blocks).
- **Options.** (a) Google trace 2019 — terabytes, BigQuery access, weeks of engineering; (b) Alibaba 2018 — ~270 GB; (c) Loghub HDFS_v1 — local, labelled, well-studied.
- **Decision.** HDFS_v1 is the single primary dataset, used end-to-end.
- **Reason.** Expert labels exist (no label engineering), real timestamps allow a chronological split, and a laptop can process it. Limitation (stated in the report): HDFS labels are per *block session*, not per machine, so node-level scheduling is evaluated in simulation driven by the measured predictor quality (D-009).

## D-002 Template extraction by variable masking (instead of Drain)
- **Context.** Models need discrete event types.
- **Options.** Drain parser; Loghub's pre-parsed structured file; regex masking.
- **Decision.** Deterministic masking of block ids, IPs, paths and numbers (`src/faultaware/data/parse.py`), with the level and component prefixed.
- **Reason.** Dependency-free, exact and reproducible; yields 54 templates (52 seen in train). Templates are label-free, and the model vocabulary is fitted on train only (unseen → `<UNK>`).

## D-003 Prediction task: early warning with observation windows
- **Context.** CLAUDE.md §3.1 asks for a label at time *t* with horizon *H*. HDFS sessions last hours (write in ~1 min; deletion/verification much later).
- **Decision.** A sample is a block session; label = Loghub expert label. A prediction is made after observing only the first `W` seconds of the block's life (`W ∈ {10 s, 60 s}`) or the full session (post-hoc detection). Features use only events with offset ≤ `W`. Lead time = `last_ts − (first_ts + W)`.
- **Primary setting:** `W = 60 s` (early warning). Full-session results are reported but are near-saturated (see D-010), so they are not used to claim prediction.

## D-004 Chronological split with a 30-minute gap
- Sessions are ordered by start time: 70 % train / 15 % val / 15 % test of sessions, with blocks starting within 30 min after each boundary dropped (33,373 blocks). Leakage tests assert `max(train) + gap < min(val)`. The test set is used once, by `scripts/evaluate.py`.

## D-005 Imbalance handling
- Class-weighted losses by default (weighted BCE, `scale_pos_weight`, balanced class weights). Compared against SMOTE (`random_forest_smote`, train only) and focal loss (`lstm_focal`). Thresholds maximise validation F1. Accuracy is never a headline metric.

## D-006 PyTorch replaces Keras/TensorFlow
- CLAUDE.md §5 mandates PyTorch. The prototype Keras CNN-LSTM was trained on a random split of 818 blocks; it is superseded by `cnn_lstm` in `src/faultaware/models/deep.py` (causal convolution + LSTM).

## D-007 Experiment tracking: JSON run records instead of MLflow
- **Reason.** MLflow adds a large dependency tree and a server for no extra rigour here. Each run writes `runs/<id>/run.json` with git commit, resolved config, seed, dataset version, metrics and artefact paths (`utils/tracking.py`), which satisfies CLAUDE.md §6.3.

## D-008 Demo: React dashboard (existing) + FastAPI instead of Streamlit
- The team already built a React/Vite dashboard. It now replays real held-out test blocks through the API, shows the model comparison and the FARS simulation (risk heat-map replay). Adding Streamlit would duplicate it.

## D-009 Simulator signal model: empirical score distributions
- **Context.** HDFS has no per-node failure timeline, so FARS cannot be replayed on real node failures.
- **Decision.** SimPy cluster with Weibull node failures; 80 % have a 15–60 min degradation precursor (hidden). Each tick, the scheduler sees a risk score drawn from the *real* fused model's test-set score distribution: anomalous-class scores while a node degrades, normal-class scores otherwise. This is conservative: one positive draw per degradation episode, so a missed precursor stays missed. Oracle and random signals bound it (sanity tests). Thresholds `τ` are tuned on validation seeds with the *validation* score distribution, and final runs use disjoint test seeds and *test* scores.
- **Limitation.** Assumes node degradation is as detectable as a faulty HDFS block at the 60 s mark; this is stated in the report and probed by the noise ablation.

## D-010 Full-session results are "too good": investigated, not leakage
- Val PR-AUC ≈ 0.999 for full sessions triggered the CLAUDE.md "too good" rule. Checks: (1) leakage tests pass (train-only vocabulary, chronological gap, per-block features); (2) a model trained on permuted labels gets val PR-AUC 0.04–0.44 depending on seed. The sign of its scores flips between seeds while `|p − mean|` stays ≈ 0.37, i.e. a *rarity* effect (anomalous blocks have rare count vectors), not label information; (3) the signal is carried by exception/WARN templates that the expert labelling itself was based on (Xu et al., 2009). Conclusion: full-session HDFS detection is near-trivial and widely reported as saturated; the early-warning setting is the meaningful task.

## D-011 Deep-learning batch size 2048 (was 512)
- Training was Python-overhead bound (~6 h projected). Batch size 2048 with LR scaled 1e-3 → 2e-3, plus trimming each batch to its longest sequence (outputs unchanged; models are causal/masked), cut a run to ~1 min. All DL runs reported use the new setting (the 6 runs trained earlier were discarded and retrained).

## D-012 Added dependencies
| Package | Why |
|---|---|
| torch | DL models (CLAUDE.md §5) |
| xgboost, scikit-learn, imbalanced-learn | classical ML + SMOTE |
| simpy | discrete-event simulator |
| pandera | data validation schemas |
| omegaconf | YAML configs with overrides |
| optuna | validation-set hyperparameter search |
| shap | tree explanations |
| fastapi, uvicorn | demo API |
| pyarrow | Parquet storage |
| matplotlib | report figures |

## D-013 Right-censoring at the end of the trace: strict early-warning evaluation
- **Observation.** With W = 60 s, validation PR-AUC ≈ 0.42 but test PR-AUC ≈ 0.96. Investigated under the "too good" rule.
- **Cause.** HDFS_v1 spans ~38.7 h and the test split is its last ~2 h, so test blocks had little time to live: 80 % of test faults end within 60 s (median lifetime 4 s), against 27 % in validation (median 5,410 s) and 36 % in train. A W-second window therefore *sees* most test faults completely; it detects rather than predicts. Blocks that would have failed after the trace ended are labelled normal, which also lowers the test fault rate (1.3 % vs 3.5 %). This is a property of the finite trace, not leakage (leakage tests pass; labels and vocabulary are unchanged).
- **Decision.** (1) Keep the standard tables (comparable with the literature) but add `reports/tables/early_warning_<W>.md`: metrics on sessions still alive when the window closes (`lead_seconds > 0`, `faultaware.eval.metrics.alive_mask`). (2) The simulator's precursor-score distributions use only alive sessions, so FARS consumes genuine predictions. (3) Thresholds and FARS taus are still tuned on validation only.
- **Finding to report.** On alive sessions, validation fault scores are barely separable from normal ones (median fused score 0.036 vs 0.035), while test faults are strongly separable (0.96). The fault mix is non-stationary across periods; any single-period number overstates or understates real early-warning skill. This is the main limitation of the study.

## D-014 Honest headline numbers: strict early warning first, two score sources for FARS
- **Context.** After D-013, the standard 60 s test PR-AUC (0.963) remained the headline in the README, report and guide, although it is inflated by right-censoring at the trace end. FARS was driven only by test-period score distributions, which (even restricted to still-running blocks) are far more separable than validation-period ones.
- **Decision.** (1) The **strict early-warning** numbers are the headline everywhere; 0.963 is reported only as the standard-protocol figure, explained as an evaluation bias. (2) `scripts/run_simulation.py` runs FARS on the same 20 test seeds with two risk-score sources: (a) test period, still-running blocks (primary) and (b) validation period, still-running blocks (pessimistic). τ, workloads and faults are identical; baselines ignore risk and the signal has its own random stream, so the paired baseline runs are reused (`reports/tables/scheduler_score_sources.md`, `scheduler_significance_valscores.md`).
- **Reason.** A reader should see the range of what the predictor is worth to the scheduler, not only its best period. CLAUDE.md §16: flag unfair comparisons; never present a biased number as the result.

## D-015 Repository made public with the guide's approval — 2026-11-08
- **Context.** CLAUDE.md §6.4 keeps the repository private until the guide approves publishing.
- **Decision.** Ms. Niharika Chaudhary approved publishing on 2026-11-08 (reported by the team lead). The existing GitHub repository stays public. The code is released under the MIT licence (`LICENSE`); the HDFS dataset is not redistributed (see `NOTICE`).
- **Still excluded from git:** raw data, trained weights (shipped as a release bundle), credentials, `.env` files and `CLAUDE.md`. Two small prototype model files remain in commits from before 2026-11-08 (they were already public); the team chose not to rewrite history.
