# Fault-Aware Machine Learning and Deep Learning-Based Algorithm for Cloud Architecture

**Project BT-185** · IILM University
**Guide:** Ms. Niharika Chaudhary
**Team:** Aryan Raj (CS-2341005), Dheeraj Kumar (CS-2341062), Navneet Kumar (CS-2341055), Priyansh Kumar (CS-2341080)

> Draft prepared from the repository's generated results (`reports/`, `docs/EXPERIMENTS.md`). Every reference in `references.bib` must be verified by the team before submission (CLAUDE.md §13.1). Citation keys are written as `[key]`.

---

## Abstract

Cloud schedulers usually react to faults only after they happen, and pay for it in lost work, missed deadlines and wasted energy. We study whether machine-learning (ML) and deep-learning (DL) fault predictors can make a scheduler proactive, and what that is worth. On the Loghub HDFS_v1 trace (11.2 M log lines, 575,061 expert-labelled storage-block sessions [xu2009detecting, zhu2023loghub]) we frame an **early-warning** task: predict whether a block will turn out faulty from only the first 60 s (or 10 s) of its log. Splits are strictly chronological with a 30-minute gap. We compare a ladder of 14 models: trivial rules, logistic regression, random forest, XGBoost, LSTM, GRU, CNN-LSTM, Transformer, focal-loss and SMOTE variants, Isolation Forest and an LSTM autoencoder. Each runs with 5 seeds, and a calibrated supervised probability is fused with an anomaly score.

The standard protocol gives a test PR-AUC of 0.963. We show this is an evaluation bias: the end of the finite trace truncates block lifetimes, so most test faults complete inside the observation window. Restricted to blocks still running when the window closes, which is genuine early warning, the best PR-AUC is **0.79 on the test period** (chance 0.005), with warnings a median of 26 minutes ahead. On the validation period it is only **0.16** (chance 0.034). DL models do not outperform tuned XGBoost.

We then propose **FARS**, a Fault-Aware Risk-based Scheduler that avoids, migrates away from and checkpoints risky nodes. It is evaluated in a SimPy cloud simulator driven by the predictor's real score distributions for still-running blocks, against six baselines over 20 paired seeds:
- with test-period scores, FARS cuts task interruptions by 64 % and lost node-time by 15 % (Wilcoxon p < 10⁻⁴);
- with the weaker validation-period scores, the gains shrink to 10 % and 1.6 %, but remain statistically significant.

Oracle and random-predictor sanity checks bound the results, and a noise ablation shows that false alarms, not missed faults, limit the benefit.
---

## 1. Introduction

Large clusters fail constantly at the level of individual components. Disks, nodes, network links and software processes fail every day in any data centre with thousands of machines [schroeder2006failures, reiss2012heterogeneity]. Most production schedulers treat failure as an event to recover from: a task is restarted after its node dies, and the work since its last checkpoint is lost. If failures can be anticipated, a scheduler can instead **(i)** stop placing new work on a node that looks unhealthy, **(ii)** migrate running work away before the failure, and **(iii)** checkpoint more often while risk is elevated.

This project builds and evaluates that loop end to end:

1. a reproducible data pipeline over a public cloud-storage trace;
2. supervised ML and DL predictors, plus unsupervised anomaly detectors, trained and compared under one protocol;
3. a calibrated, fused **risk score**;
4. the **FARS** scheduling algorithm that consumes it;
5. a discrete-event simulator that measures what the predictions are worth at the system level;
6. a login-protected web console and API that demonstrate the system on held-out data.

**Contributions.**
- An *early-warning* formulation of HDFS fault prediction: observation windows of 10 s and 60 s after block creation, chronological splits with a gap, and a train-only vocabulary. Full-session results on this benchmark are near-saturated and say little about prediction (§5.1).
- A finding that matters for anyone evaluating on HDFS_v1: **right-censoring at the end of the trace** inflates test-period early-warning scores. We propose a *strict early-warning* evaluation restricted to sessions still alive at decision time (§6.2).
- FARS, a simple, interpretable risk-threshold scheduler. Its thresholds are tuned only on validation simulations. It is evaluated against six baselines with paired statistics, under two risk-score sources (test and validation period), oracle/random sanity bounds, four ablations and a noise-sensitivity curve (§6.4–6.7).
- An open, tested implementation: 123 automated tests (93 % line coverage) and one-command reproduction.

---

## 2. Literature Review

*(Team: expand each paragraph with 2–4 verified papers. Search keywords are given in brackets.)*

**Log-based anomaly detection.** Xu et al. [xu2009detecting] introduced the HDFS dataset and detected anomalous block sessions with PCA over event-count vectors. DeepLog [du2017deeplog] modelled normal log-key sequences with an LSTM and flagged deviations. LogAnomaly [meng2019loganomaly] and LogRobust [zhang2019logrobust] added semantic template vectors and robustness to evolving logs. Log parsers such as Drain [he2017drain] turn raw lines into templates. Le and Zhang [le2022howfar] showed that many reported results depend on random (non-chronological) splits and that HDFS detection is close to saturated, which motivates our early-warning setting and chronological protocol. *[keywords: log anomaly detection deep learning, HDFS log, DeepLog, log parsing benchmark]*

**Failure prediction in clouds and HPC.** Field studies report heavy-tailed, Weibull-like inter-failure times [schroeder2006failures]. Cluster traces such as Google's [reiss2012heterogeneity] have been used to predict job and task failures from resource usage. *[keywords: job failure prediction Google cluster trace, node failure prediction data center, disk failure prediction SMART]*

**Fault-tolerant and fault-aware scheduling.** Proactive fault tolerance (migration, replication, adaptive checkpointing) has been studied mainly with oracles or synthetic predictors. Fewer works close the loop with a *measured* predictor and statistical comparison. *[keywords: proactive fault tolerance cloud, failure-aware scheduling, adaptive checkpointing failure prediction, VM migration failure prediction]*

**Models and methods used here.** LSTM [hochreiter1997lstm], GRU [cho2014gru], Transformer [vaswani2017attention], temporal convolutions [bai2018tcn], random forests [breiman2001rf], gradient boosting [chen2016xgboost], Isolation Forest [liu2008iforest], LSTM encoder-decoder anomaly detection [malhotra2016lstmed], SMOTE [chawla2002smote], focal loss [lin2017focal], isotonic and Platt calibration [zadrozny2002calibration, platt1999probabilistic], SHAP [lundberg2017shap], Optuna [akiba2019optuna], PR curves under imbalance [saito2015prc], and paired non-parametric tests [wilcoxon1945, demsar2006statistical].

---

## 3. Problem Formulation

**Sample and label.** Each HDFS block *b* produces a session: the time-ordered log events that mention it, from allocation (`first_ts`) to its last event (`last_ts`). Loghub provides an expert label *y_b* ∈ {0, 1} (1 = anomalous block). [xu2009detecting]

**Early-warning prediction.** A prediction is made at decision time *t_b = first_ts + W*, using only events with timestamp ≤ *t_b*. We use *W* ∈ {10 s, 60 s} and, for comparison with the literature, *W* = full session (post-hoc detection). The **lead time** of a correct warning is *last_ts − t_b*.

**Why this matches CLAUDE.md §3.1.** The general definition labels time *t* as 1 if a failure occurs in (*t*, *t + H*]. HDFS has no per-node failure timeline, and its labels are per block, so the natural unit is the block session. The "horizon" becomes the remaining lifetime after the window, and the two required horizons become the two observation windows, 10 s and 60 s.

**Objective of the scheduler.** Given per-node risk *r_i(t)* ∈ [0, 1] every Δ*t* = 5 min, choose placements, migrations and checkpoint frequencies. The goal is to minimise **lost node-minutes** = wasted compute + checkpoint overhead + migration time, while keeping task interruptions, SLA violations and response time low.

---

## 4. Dataset and Preprocessing

| Property | Value |
|---|---|
| Source | Loghub HDFS_v1 [zhu2023loghub]; originally Xu et al. [xu2009detecting] |
| Raw size | 1.58 GB, 11,175,629 lines, ~38.7 h of a 203-node Hadoop cluster (Nov 2008) |
| Labels | 575,061 blocks, 16,838 anomalous (2.93 %) |
| Templates | 54 after masking (52 seen in training) |

**Parsing (`src/faultaware/data/parse.py`).** Each line is split into date, time, PID, level, component and content. Block ids, IP addresses, paths and numbers are masked to obtain a template. The level and component are prefixed, so `INFO` and `WARN` variants stay distinct. Masking is deterministic and dependency-free. Our counts match Loghub's published statistics exactly.

**Sessions (`data/sessions.py`).** Events are grouped per block id, with time offsets relative to `first_ts`, the template sequence, source IPs and levels.

**Chronological split (`data/split.py`).** Blocks are ordered by `first_ts`: 70 % train, 15 % validation, 15 % test. Blocks starting within **30 min** after each boundary are dropped (33,373), so no label can straddle two splits.

| Split | Blocks | Fault rate | Period |
|---|---|---|---|
| Train | 402,549 | 3.26 % | first ~33.4 h |
| Validation | 68,954 | 3.45 % | next ~2 h |
| Test | 70,185 | 1.31 % | last ~2 h |

**Features (`features/build.py`)**, computed only from events with offset ≤ *W*:
- *Tabular*: template count vector over the train-fitted vocabulary (unseen templates → `<UNK>`), plus session statistics: events, unique templates, unique IPs, WARN-level lines, and duration observed.
- *Sequence*: the first 64 template ids, padded, for the RNN, CNN-LSTM and Transformer models.

**Leakage safeguards (tested in `tests/unit/test_leakage.py`).**
- Chronological order and gap: `max(train) + gap < min(val)`.
- The vocabulary and scalers are fitted on train only.
- Features are invariant to events after *W*; this is tested by mutating later events.
- SMOTE is applied only to training data.
- Thresholds, calibration, α and τ are chosen on validation only.
- The test split is read by exactly one script.

**Exploration.** `notebooks/01_eda.ipynb` covers the class balance, the fault rate over time (it varies by hour, from 0.9 % to 6.4 %), the per-IP fault rate (1.6–21 %) and session lengths (median 19 events).

---

## 5. Methodology

### 5.1 Model ladder (CLAUDE.md §7.3)

All models implement one `FaultPredictor` interface (`fit`, `predict_proba`, `save`, `load`).

| Family | Models | Key settings |
|---|---|---|
| Trivial | always-0; random at base rate; **rule**: any WARN line in the window | — |
| Classical ML | logistic regression; random forest; XGBoost; RF + SMOTE | class-balanced weights; RF/XGB tuned with Optuna on validation PR-AUC |
| Deep learning | LSTM, GRU (2 layers, 64 hidden); causal CNN-LSTM; Transformer encoder (2 layers, 4 heads); LSTM + focal loss | weighted BCE, AdamW lr 2e-3, batch 2048, grad-clip 1.0, early stopping on val PR-AUC (patience 4) |
| Unsupervised | Isolation Forest (tabular); LSTM autoencoder (sequence) | trained on **normal training blocks only**; anomaly score = isolation depth / reconstruction error |

Every model is trained with 5 seeds (0–4) for each of the 3 windows, giving 210 runs. Decision thresholds maximise validation F1.

### 5.2 Calibration and risk fusion (`fusion/risk.py`)

The best supervised model (by validation PR-AUC) is calibrated with isotonic regression fitted on validation [zadrozny2002calibration]. The best anomaly detector's score is min-max normalised on validation. The fused risk is

  *r = α · p_cal + (1 − α) · a*,

with α chosen from {0, 0.1, …, 1} to maximise validation PR-AUC. At *W* = 60 s, this selects random forest + LSTM-AE with α = 0.9.

### 5.3 FARS — Fault-Aware Risk-based Scheduler (`scheduler/fars.py`)

Every tick Δ*t* = 5 min:
1. Smooth each node's risk with an EWMA: *r̄_i ← 0.5 · r_i + 0.5 · r̄_i*.
2. **If *r̄_i ≥ τ_high*:** migrate running tasks off node *i* (30 s each, progress kept) to the best healthy node, and checkpoint anything that cannot move.
3. **If *τ_low ≤ r̄_i < τ_high*:** place no new tasks on *i*, and checkpoint its tasks every 10 min instead of every 30 min.
4. **Placement** of a new task on feasible node *j* maximises *score_j = 1.0 · free_capacity_j − 2.0 · r̄_j − 0.5 · migration_cost_j*.

τ_low and τ_high are chosen by grid search on **5 validation seeds** using the *validation* score distribution, which gives τ_low = 0.3 and τ_high = 0.4. All weights are in `configs/scheduler/fars.yaml`.

**Baselines:** Random, Round-Robin, First-Fit, Best-Fit, Least-Loaded, and **Reactive** (restart after failure and avoid the failed node for 60 min).

### 5.4 Cloud simulator (`simulator/`)

The simulator is built on SimPy, with time measured in minutes:
- **Cluster:** 40 nodes, each with unit CPU and memory.
- **Workload:** 5,000 tasks with Poisson arrivals (2.6/min, about 60 % utilisation), lognormal durations (5–600 min), CPU and memory demands of 0.05–0.35, and deadline = arrival + 1.3 × duration.
- **Faults:** node failures follow a Weibull renewal process (shape 1.5, scale 1,800 min) [schroeder2006failures]; repair takes 30 min. **80 %** of failures are preceded by a hidden 15–60 min degradation window, and 20 % arrive with no warning.
- **Signal model:** each tick, the scheduler receives one risk score per node, drawn from the **real fused model's score distribution**: the faulty-block scores while the node degrades, normal-block scores otherwise. Only blocks still alive after the window are used (D-013). The draw is persistent per degradation episode, which is conservative: a missed precursor stays missed.
- **Costs:** every scheduler checkpoints every 30 min at 1 min each, a migration takes 30 s, and a restart adds a 2 min penalty.
- **Fairness and determinism:** every scheduler sees the identical workload and fault sequence for a given seed, and runs are deterministic given the seed.

**Metrics:** task interruptions, wasted compute, lost node-minutes, SLA violation rate, mean response time, makespan, utilisation, migrations, checkpoint overhead, and energy (linear power model, 100–250 W).

### 5.5 Evaluation protocol
- **Prediction:** PR-AUC (primary), ROC-AUC, precision, recall, F1, FPR, Brier score and latency. All are reported as mean ± std over 5 seeds at validation-chosen thresholds.
- **Paired significance:** Wilcoxon signed-rank and paired t-tests over seeds, plus a bootstrap 95 % CI for seed 0. With 5 seeds the smallest attainable two-sided Wilcoxon p is 0.0625, so the t-test is also given.
- **Scheduling:** 20 test seeds, separate from the 5 validation seeds, with paired Wilcoxon tests against each baseline.
- **Explainability:** SHAP on the tree model [lundberg2017shap].

---

## 6. Results

### 6.1 Prediction under the standard protocol (test split, 5 seeds)

> **Read with §6.2.** These scores include blocks whose fault finished inside the observation window. On the test period that is 80 % of faults, so the 60 s and 10 s columns partly measure *detection*. They are reported for comparability with the literature. The headline early-warning results are the strict ones in §6.2.

| Model | Full session (detection) | W = 60 s | W = 10 s |
|---|---|---|---|
| Rule (any WARN) | 0.176 | 0.108 | 0.015 |
| Logistic regression | 0.998 | 0.953 | 0.826 |
| Random forest | 0.996 | 0.945 ± 0.008 | 0.789 ± 0.020 |
| XGBoost | **1.000** | **0.963 ± 0.003** | 0.827 |
| LSTM | 1.000 | **0.963 ± 0.004** | 0.819 ± 0.017 |
| GRU | 1.000 | 0.936 ± 0.018 | 0.821 ± 0.006 |
| CNN-LSTM | 1.000 | 0.958 ± 0.005 | 0.812 ± 0.025 |
| Transformer | 1.000 | 0.959 ± 0.003 | 0.831 ± 0.009 |
| LSTM + focal loss | 1.000 | 0.956 ± 0.009 | 0.828 ± 0.002 |
| RF + SMOTE | 1.000 | 0.913 ± 0.020 | 0.782 ± 0.013 |
| Isolation Forest | 0.400 | 0.301 | 0.640 |
| LSTM autoencoder | 0.849 | 0.782 ± 0.091 | 0.459 |
| **Fused risk** | 1.000 | 0.940 ± 0.008 | **0.834 ± 0.005** |

*Test PR-AUC. Base rate (random ranking) = 0.013. Full tables: `reports/tables/prediction_*.md`.*

Observations:
1. **Full-session detection is saturated.** Even logistic regression reaches 0.998. We investigated this under the "too good" rule (D-010). Leakage tests pass, and the signal comes from exception and WARN templates that the expert labelling was itself based on. This agrees with prior observations [le2022howfar]. We therefore do not use it to claim prediction.
2. **At 60 s and 10 s, all learned supervised models are close.** At 60 s the Transformer beats RF by +0.015 (paired t-test p = 0.015, Wilcoxon p = 0.0625 at n = 5). It is not better than tuned XGBoost (0.959 vs 0.963). **DL does not clearly beat gradient-boosted trees** on this data, which is a valid finding: the template-count vector already captures most of the information in the first minute.
3. **Imbalance handling.** SMOTE *hurt* RF (0.913 vs 0.945) and focal loss neither helped nor hurt the LSTM (0.956 vs 0.963). Class weighting alone was sufficient.
4. **Unsupervised detectors** are much weaker than supervised ones, but well above chance. The LSTM-AE (0.78 at 60 s) needs no labels, which matters for fault types never seen in training.
5. **Fusion** did not improve ranking at 60 s (−0.005, p = 0.29) and improved it slightly at 10 s (+0.004, not significant). Its real value is **calibration**: the Brier score falls from 0.127 (raw RF) to 0.0019 (isotonic) and 0.0029 (fused). FARS uses absolute thresholds τ, so calibrated probabilities are what make those thresholds meaningful (see ablation (b)).
6. **Explainability.** The top SHAP features for the 60 s model are the count of `Receiving block` events, the number of events, the number of distinct IPs and distinct templates, and `addStoredBlock` updates. In other words, the *shape of the replication pipeline in the first minute*, not just error messages (`reports/figures/shap_importance.png`).

### 6.2 Strict early warning (headline result) and right-censoring

Under the standard protocol, validation PR-AUC at 60 s is about 0.42 for every model, yet test PR-AUC is about 0.96. We traced this gap to the trace boundary. **The 0.963 is an evaluation bias, not predictive skill:**

| | Train | Validation | Test |
|---|---|---|---|
| Faults ending within 60 s of creation | 36 % | 27 % | **80 %** |
| Median fault lifetime | 8,164 s | 5,410 s | **4 s** |
| Fault rate | 3.26 % | 3.45 % | 1.31 % |

The test period is the last ~2 h of a 38.7 h trace, so most test blocks that turned faulty did so almost immediately, inside the observation window. The window then *sees* the fault rather than predicting it. Blocks that would have failed after the recording stopped are labelled normal, which also explains the lower test fault rate.

**Strict early-warning evaluation** therefore scores only sessions still running when the window closes (`lead_seconds > 0`):

| Model (W = 60 s) | Validation PR-AUC (strict) | Test PR-AUC (strict) | Test F1 (strict) |
|---|---|---|---|
| Base rate | 0.034 | 0.005 | — |
| Rule (any WARN) | 0.063 | 0.294 | 0.009 |
| Logistic regression | 0.155 | 0.750 | 0.853 |
| Random forest | 0.165 | 0.746 | 0.854 |
| **XGBoost** | 0.159 | **0.792 ± 0.012** | 0.856 |
| LSTM | 0.147 | 0.777 ± 0.026 | 0.849 |
| Transformer | 0.151 | 0.771 ± 0.021 | 0.853 |
| LSTM autoencoder | 0.120 | 0.495 | 0.647 |
| Fused risk | 0.167 | 0.742 | 0.848 |

*Test: 39,977 still-running blocks, 189 faults. Validation: 50,558 blocks, 1,733 faults.*

- **Test.** 148 of the 189 still-running faults are flagged correctly, a median of **26 min** (IQR 10–41 min) before their sessions end.
- **Validation.** The same models reach only 4–5× the base rate. Fused scores for still-running faults (median 0.036) are almost indistinguishable from normal blocks (0.035).
- **Interpretation.** The fault mix is non-stationary. Whatever made the validation-period faults happen happened *after* the first minute and left no early trace, whereas many test-period faults show their cause in the first minute.
- **Takeaway.** Early-warning skill on HDFS depends strongly on the period evaluated. Anyone evaluating on this trace should report censoring-aware numbers.

### 6.3 Calibration

Isotonic calibration on validation brings the test Brier score from 0.127 to 0.0019 (`reports/figures/reliability.png`). Uncalibrated tree scores are badly over-confident on normal blocks (median raw RF probability 0.33), which would make any fixed τ meaningless.

### 6.4 FARS vs baselines (20 test seeds, test-period risk scores)

| Scheduler | Task interruptions | Wasted compute (min) | Lost node-min | SLA viol. | Migrations |
|---|---|---|---|---|---|
| Random | 131.4 ± 16.5 | 1,859 ± 252 | 7,708 ± 262 | 1.0 % | 0 |
| Round-robin | 130.5 ± 16.4 | 1,789 ± 247 | 7,638 ± 248 | 1.0 % | 0 |
| First-fit | 132.9 ± 14.4 | 1,832 ± 213 | 7,681 ± 207 | 1.0 % | 0 |
| Best-fit | 132.8 ± 17.6 | 1,818 ± 273 | 7,667 ± 295 | 1.0 % | 0 |
| Least-loaded | 129.9 ± 15.2 | 1,791 ± 242 | 7,640 ± 269 | 1.0 % | 0 |
| Reactive | 131.1 ± 18.0 | 1,845 ± 237 | 7,694 ± 236 | 1.0 % | 0 |
| **FARS** | **47.2 ± 13.4** | **651 ± 215** | **6,512 ± 209** | **0.4 %** | 97 |

Against every baseline, FARS reduces:
- task interruptions by **64 %**;
- wasted compute by **64 %**;
- SLA violations by **60 %**;
- lost node-minutes by **15 %**, which includes FARS's own migration and checkpoint costs.

All of these hold at Wilcoxon p ≤ 1e-4 over 20 paired seeds. Mean response time improves by 0.5 %, and energy use is unchanged (−0.4 %).

**Risk-score source (D-014).** The scores above come from the test period, whose still-running faults are much easier to separate than the validation period's (§6.2). We re-ran FARS on the same 20 seeds, with the same τ, workloads and faults, drawing the node risk scores from the validation-period distribution instead:

| Risk scores drawn from | Interruptions | Wasted compute (min) | Lost node-min | SLA viol. | Migrations |
|---|---|---|---|---|---|
| Best baseline | 129.9 | 1,789 | 7,638 | 1.0 % | 0 |
| Test period (§6.4) | 47.2 ± 13.4 (−64 %) | 651 ± 215 | 6,512 ± 209 (−15 %) | 0.4 % | 97 |
| Validation period (pessimistic) | 117.4 ± 15.9 (−10 %) | 1,664 ± 222 | 7,515 ± 203 (−1.6 %) | 0.9 % | 16 |

- **With the weaker scores**, FARS's thresholds are rarely crossed (16 migrations), so it behaves almost like least-loaded with a slight edge.
- **It is still significantly better than every baseline:**
  - interruptions: Wilcoxon p ≤ 6 × 10⁻⁴;
  - lost node-time: p between 6 × 10⁻⁴ and 0.022;
  - SLA violations versus least-loaded: −10 %, p = 0.021.
- **The two rows bracket what the predictor is worth to the scheduler:** a large gain when faults leave an early trace, and a small, safe gain when they do not. The six fault-unaware baselines are statistically indistinguishable from each other. Placement alone cannot avoid failures that strike at random, which is why prediction is needed.

### 6.5 Simulator sanity checks (CLAUDE.md §9)

| Signal given to FARS | Interruptions | Lost node-min |
|---|---|---|
| Least-loaded (no signal) | 129.9 | 7,640 |
| FARS + **random** scores | 148.5 | 28,843 |
| FARS + **real** model scores | 47.2 | 6,512 |
| FARS + **oracle** | 24.1 | 6,203 |

The oracle gives the expected bound. The remaining ~24 interruptions are the 20 % of failures with no precursor. A random predictor makes FARS far *worse* than baselines, because it migrates and checkpoints constantly. The real predictor recovers **79 %** of the oracle's reduction in interruptions: (129.9 − 47.2) / (129.9 − 24.1).

### 6.6 Ablations (CLAUDE.md §8.3)

| Variant (τ re-tuned on validation) | τ_low/τ_high | Interruptions | Lost node-min |
|---|---|---|---|
| FARS (full) | 0.3 / 0.4 | 47.2 ± 13.4 | 6,512 ± 209 |
| (a) without anomaly detector | 0.3 / 0.4 | 48.4 ± 12.9 | 6,515 ± 183 |
| (b) without calibration | 0.5 / 0.6 | 38.9 ± 13.6 | **8,228 ± 211** |
| (c) W = 10 s | 0.5 / 0.6 | 90.7 ± 18.2 | 7,112 ± 259 |
| (c) full session (detection, upper bound) | 0.1 / 0.4 | 24.3 ± 12.1 | 6,256 ± 142 |

- **(a)** The anomaly detector adds nothing measurable at 60 s, consistent with α = 0.9.
- **(b)** Without calibration, FARS interrupts fewer tasks but loses *more* node-time than any baseline. The uncalibrated scores are compressed (every node looks "medium risk"), so validation-tuned thresholds cause excessive checkpointing and migration. **Calibration is what makes FARS pay off.**
- **(c)** A shorter window gives a weaker but still positive benefit: lost node-min is 7 % lower than least-loaded. A longer window (detection) approaches the oracle.
- **(d) Threshold sensitivity** (`reports/figures/fars_threshold_sweep.png`): on the validation seeds, the 27 valid (τ_low, τ_high) pairs span only 7,462–7,612 lost node-min (< 2 %). The validation-period scores carry little signal (§6.2), so the choice of τ is weakly determined. The selected pair (0.3, 0.4) is the grid minimum, and it transferred well to the test seeds, but with weak validation information that is partly fortunate. A deployment should re-tune τ as the predictor's score distribution drifts.

### 6.7 Prediction-noise sensitivity

| Share of scores replaced by noise η | 0 | 0.10 | 0.25 | 0.50 | 1.0 |
|---|---|---|---|---|---|
| Lost node-min | 6,512 | 8,128 | 11,066 | 19,238 | 28,854 |
| Interruptions | 47.2 | 45.2 | 49.0 | 62.8 | 146.3 |
| Migrations | 97 | 1,479 | 3,950 | 6,166 | 1,617 |

Interruptions stay low up to η = 0.25, because true warnings still get through. Lost node-time, however, exceeds the best baseline (7,640) already at η = 0.1, because every false alarm triggers a migration. FARS's net benefit therefore requires a low false-alarm rate. The 60 s model's test FPR is 0.1 % at its validation threshold, but there is no guarantee of that in a period like the validation window, where scores are less informative.

### 6.8 Latency

Every model scores a block in under 0.04 ms when batched (GPU for DL models). End-to-end API latency for one block, including parsing and featurisation, was 29–48 ms on the development laptop. This is well within a 5-minute scheduling tick.

---

## 7. Discussion and Limitations

1. **Block-level labels, node-level scheduling.** HDFS labels faulty *block sessions*, not failing machines. FARS is therefore evaluated in simulation, where node-degradation signals are drawn from the block-level score distributions (D-009). This assumes a degrading node is as recognisable as a faulty block one minute in. The noise ablation shows how the conclusion changes when that assumption is weakened.
2. **Non-stationarity and censoring.** The 38.7 h trace is short, and validation and test periods differ sharply (§6.2). Five seeds measure model variance, not period variance. A multi-week trace (Google 2019, Alibaba) or rolling-origin evaluation would be needed to estimate how stable early-warning skill is.
3. **Synthetic workload.** HDFS logs contain no task trace, so arrivals and durations are synthetic (Poisson / lognormal). Migration is idealised as 30 s, preserving progress, with no network contention.
4. **DL hyperparameters were not searched** (time budget); tuned trees may be slightly favoured.
5. **Single dataset.** Cross-dataset validation (BGL, Google trace) is future work.

**Cost of errors.** A false negative costs the work lost since the last checkpoint plus a restart. A false positive costs one migration (30 s) or extra checkpoints (1 min each). Per event, a false positive is far cheaper. But false positives are far more *numerous* because faults are rare (1–3 %), which is why the noise ablation shows false alarms dominating total cost. The F1-optimal threshold is a reasonable compromise. The cost-weighted objective in `configs/fusion/default.yaml` lets a deployment shift it.

**Deployment path.** In a real cluster, the risk service would read node logs and metrics (for example via Prometheus), and FARS's rules would run as a Kubernetes scheduler extender (placement filter and score) plus a controller that cordons and drains nodes above τ_high. The repository's API (`POST /predict`) and the containerised deployment (`docker compose`) are a first step.

---

## 8. Conclusion and Future Work

Under the standard protocol, ML and DL models score a test PR-AUC of 0.96 at 60 s. We showed that this figure is an evaluation bias caused by right-censoring at the end of the trace. Genuine early warning, measured on blocks still running after the window, reaches a PR-AUC of 0.79 on the test period (warnings a median of 26 min ahead) but only 0.16 on the validation period. Gradient-boosted trees match recurrent and attention models.

Fed through isotonic calibration into the simple, threshold-based FARS scheduler:
- **test-period predictions** cut task interruptions by 64 % and total lost node-time by 15 % against six baselines in simulation, capturing 79 % of a perfect oracle's benefit;
- **validation-period predictions** still give a statistically significant but small gain (−10 % interruptions, −1.6 % lost node-time).

The benefit depends critically on calibration and on a low false-alarm rate. Reporting both score sources, rather than the best one, is what makes the result trustworthy.

**Future work:**
- multi-week traces with rolling-origin evaluation;
- per-node prediction from resource metrics (Google 2019, Backblaze SMART);
- a cost-aware learning objective;
- a DQN/PPO scheduler whose state includes the risk scores (CLAUDE.md §2.1 step 5);
- online drift detection to switch FARS off when the predictor's false-alarm rate rises.

---

## Reproducibility

```bash
make setup && make reproduce      # data → tune → train → eval → simulate
```

- Every number above is listed with its command in `docs/EXPERIMENTS.md`, and every design decision is in `docs/DECISIONS.md` (D-001 … D-013).
- Run records include the git commit, resolved config and seed.
- All randomness is seeded.

## References
See `references.bib` (to be verified by the team).

## Appendix: figures (in `reports/figures/`)
`class_distribution`, `fault_rate_over_time`, `pr_curves_{full,w60,w10}`, `confusion_{full,w60,w10}`, `training_curves`, `reliability`, `fusion_alpha`, `shap_importance`, `lead_time_hist`, `scheduler_{task_interruptions,wasted_compute_min,lost_node_min,mean_response_min}`, `fars_threshold_sweep`, `ablation_noise`. All are generated as PNG (300 dpi) and PDF.
