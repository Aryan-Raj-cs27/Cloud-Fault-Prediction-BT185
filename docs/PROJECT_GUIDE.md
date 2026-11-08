# BT-185 Project Guide: everything about the project, in one place

This guide is for the team. It explains what was built, why each choice was made, what the results mean (including the uncomfortable parts), how to run and deploy it, how to turn it into a research paper, and how to answer viva questions. The formal write-up is `docs/report/report.md`; this guide is the version that explains the reasoning.

**Contents**
1. The project in one paragraph
2. The big picture: how the pieces fit
3. The data
4. The prediction task, and why it is framed this way
5. The models
6. Calibration and risk fusion
7. FARS, the scheduling algorithm
8. The simulator
9. Results, and how to read them honestly
10. How we prevented data leakage
11. The software: code map, tests, quality gates
12. The web console and API
13. Security
14. Running, reproducing, deploying
15. Writing the research paper
16. Viva preparation: likely questions with answers
17. Limitations and future work
18. Glossary

---

## 1. The project in one paragraph

We take real logs from a Hadoop storage cluster (HDFS) in which experts labelled which storage blocks went wrong. We train 14 kinds of models (simple rules, classical machine learning, deep learning and unsupervised anomaly detectors) to predict, **from only the first 60 seconds of a block's life**, whether it will turn out faulty. We combine the best predictor with an anomaly detector into one calibrated **risk score**. We then design **FARS (Fault-Aware Risk-based Scheduler)**. FARS uses risk scores to avoid unhealthy nodes, move work off nodes about to fail, and checkpoint more often when risk is high. We test it in a simulated cloud against six standard schedulers.

**The honest headline:**
- **Prediction:** genuine early warning (blocks still running after 60 s) reaches PR-AUC **0.79 on the test period, but only 0.16 on the validation period**. The often-quoted 0.963 is an evaluation bias (§9.2).
- **Scheduling:** FARS cuts interrupted tasks by **64 %** and lost machine-time by **15 %** when driven by test-period scores. With the harder validation-period scores, the gains are **10 % and 1.6 %**. Both results are statistically significant over 20 paired runs (§9.3).

---

## 2. The big picture: how the pieces fit

```
 HDFS raw log (11.2 M lines)  +  expert labels (575,061 blocks)
          │  parse + mask variables  →  54 event templates
          ▼
 block sessions (one row per block: its events in time order)
          │  chronological split 70/15/15 with a 30-min gap
          ▼
 features from the FIRST W SECONDS only (W = 60 s main, 10 s, or full session)
          │
   ┌──────┴──────────────────────────────┐
   ▼                                     ▼
 supervised models                 unsupervised anomaly detectors
 (LR, RF, XGBoost, LSTM, GRU,      (Isolation Forest, LSTM autoencoder;
  CNN-LSTM, Transformer, ...)       trained on normal blocks only)
   │ isotonic calibration (on validation)  │ normalised score
   └──────────────┬───────────────────────┘
                  ▼
     risk r = α·p + (1−α)·a        (α chosen on validation)
                  ▼
     FARS scheduler: place / avoid / migrate / checkpoint  (τ chosen on validation simulations)
                  ▼
     SimPy cloud simulator (40 nodes, 5,000 tasks, Weibull node failures)
                  ▼
     metrics + statistics → reports/ (tables, figures) → web console
```

The rule that runs through everything: **anything chosen (thresholds, α, τ, hyperparameters) is chosen on the validation split; the test split is used once at the end.**

---

## 3. The data

- **Dataset:** Loghub HDFS_v1, originally from Xu et al. (SOSP 2009). It covers 38.7 hours of a 203-node Hadoop cluster in November 2008: 1.58 GB and 11,175,629 log lines.
- **Labels:** each of 575,061 storage blocks is labelled Normal or Anomaly by experts; 16,838 (2.93 %) are anomalies.
- **What a "block session" is:** HDFS stores files in blocks of about 64 MB. Each block is allocated, written through a pipeline of DataNodes (replicas), confirmed, and later possibly deleted or verified. Every log line mentioning the block id belongs to its session. A typical session has 19 events.
- **Why HDFS instead of the Google trace** (decision D-001): the Google trace is terabytes and needs BigQuery; HDFS is local, labelled and well studied. The cost is that labels are per block, not per machine, which is why the scheduler is evaluated in simulation (§8).
- **Parsing:** each line has a date, time, process id, level (INFO/WARN), component and message. We mask variables (block ids → `<BLK>`, IPs → `<IP>`, numbers → `<NUM>`) to get event *templates*, which gives 54 templates. Example template: `INFO dfs.DataNode$DataXceiver: Receiving block <BLK> src: <IP> dest: <IP>`.
- **EDA** (`notebooks/01_eda.ipynb`): the fault rate changes by hour (0.9–6.4 %) and by node IP (1.6–21 %); classes are heavily imbalanced.

---

## 4. The prediction task, and why it is framed this way

**The easy version (detection).** Look at the whole session and decide if it was faulty. On HDFS this is almost solved: even logistic regression gets PR-AUC 0.998, because faulty sessions contain exception messages that the experts used to label them. That is detection after the fact, not prediction.

**Our version (early warning).** Decide at 60 seconds (or 10 s) after the block is created, using only events up to then. If the block's session continues afterwards, a correct warning comes before the trouble is over. That gap is the **lead time**.

**Label (write it this way in the paper):** *a block b is positive if its Loghub expert label is "Anomaly". The prediction is made at t_b = first_ts + W using only events with timestamp ≤ t_b; W ∈ {10 s, 60 s}, plus the full session as a detection reference. Lead time = last_ts − t_b.*

**Why PR-AUC and not accuracy:** with 1–3 % faults, a model that always says "normal" is 97–99 % accurate and useless. PR-AUC measures how well faults are ranked above normal blocks. Its chance level equals the fault rate. Among still-running test blocks that is 0.005, so our strict 0.79 is far above chance.

---

## 5. The models

We built a "ladder", from trivial to complex, so every complex model has to beat simpler ones:

| Rung | Model | What it sees | Notes |
|---|---|---|---|
| 1 | Always-0, random at base rate | nothing | chance reference |
| 1 | Rule: any WARN line in window | count of WARN lines | the "obvious" heuristic |
| 2 | Logistic regression | template counts + statistics | linear baseline |
| 3 | Random forest, XGBoost | same | tuned with Optuna on validation |
| 3 | RF + SMOTE | same | tests oversampling vs class weights |
| 4 | LSTM, GRU | the sequence of the first 64 templates | 2 layers, 64 hidden |
| 4 | LSTM + focal loss | sequence | tests focal loss vs class weights |
| 5 | CNN-LSTM (causal) | sequence | convolution picks local patterns |
| 5 | Transformer encoder | sequence | self-attention, 2 layers, 4 heads |
| 6 | Isolation Forest | counts | unsupervised, normal data only |
| 6 | LSTM autoencoder | sequence | trained to reconstruct normal sequences; high error = anomaly |
| 7 | Fused risk | best supervised + best anomaly | calibrated, see §6 |

**Training details:**
- Class imbalance is handled with class weights.
- The deep models use AdamW, a learning-rate scheduler, gradient clipping, and early stopping on validation PR-AUC.
- Every model is trained with 5 seeds for each of 3 windows: 210 training runs, about 1 minute each on the RTX 4050.

**Why unsupervised models:** supervised models only recognise fault types seen in training. An autoencoder trained only on normal behaviour flags anything unusual, including new fault types.

---

## 6. Calibration and risk fusion

**Calibration.** A model's "probability" is often not a real probability. Our raw random forest gives normal blocks a median score of 0.33. Isotonic regression, fitted on validation, maps scores to observed frequencies. The test Brier score (a calibration error measure) dropped from 0.127 to 0.0019.

**Why the scheduler needs calibrated scores.** FARS compares risk to fixed thresholds (τ). If 0.4 means "40 % chance of failure" today but "5 %" for another model, the thresholds mean nothing. The ablation proves it: without calibration, FARS loses more machine-time than doing nothing smart at all.

**Fusion:** r = α · p_calibrated + (1 − α) · anomaly_score. α is chosen on validation. At 60 s, α = 0.9 (mostly the supervised model) with random forest + LSTM autoencoder.

---

## 7. FARS, the scheduling algorithm

**FARS = Fault-Aware Risk-based Scheduler.** Every 5 minutes, for each node:

1. **Smooth** the risk (EWMA with factor 0.5), so one noisy score doesn't trigger action.
2. **High risk (≥ τ_high = 0.4):** migrate running tasks to healthy nodes (30 s each, progress kept); checkpoint what can't move.
3. **Medium risk (τ_low = 0.3 to τ_high):** don't place new tasks there; checkpoint its tasks every 10 min instead of 30.
4. **New task placement:** choose the node maximising `1.0 × free capacity − 2.0 × risk − 0.5 × migration cost`.

**Why rules instead of reinforcement learning:** rules are interpretable, need no training, and give a clear baseline. CLAUDE.md says to try deep RL only after the rule-based version works; that is listed as future work.

**Baselines it is compared against:**
- Random, Round-Robin;
- First-Fit, Best-Fit (bin-packing);
- Least-Loaded;
- **Reactive**, which restarts after a failure and avoids that node for an hour. This is "what most systems do today".

---

## 8. The simulator

HDFS logs have no record of which *machines* failed, so we cannot replay real node failures. Instead (decision D-009):

- **The cluster:** 40 nodes and 5,000 tasks. Arrivals are random, at about 60 % utilisation, and tasks run 5–600 min.
- **Node failures:** they follow a Weibull process, the standard model from HPC failure studies, and a node takes 30 min to repair.
- **Precursors:** 80 % of failures have a hidden 15–60 min "degradation" period before them; 20 % come with no warning at all.
- **Risk scores:** each tick, FARS sees a risk score per node. While a node is degrading, the score is drawn from the **real model's scores for real faulty blocks**; otherwise it is drawn from the real scores for normal blocks. So the simulator inherits exactly the predictor's real strengths and mistakes.
- **Conservative scoring:** a degradation episode gets one score that persists, so if the model "missed" it, it stays missed.
- **Fair comparison:** every scheduler sees the same workload and the same failures for a given seed.
- **Sanity checks:** with a perfect (oracle) predictor FARS should almost eliminate avoidable losses; with random scores it should not help. Both pass, and both are automated tests.

---

## 9. Results, and how to read them honestly

### 9.1 Prediction under the standard protocol (test split, mean over 5 seeds)

*Not the headline. Read §9.2 first: these scores include faults that finished inside the window.*

| Setting | Best models | Test PR-AUC |
|---|---|---|
| Full session (detection) | XGBoost, all DL | ≈ 1.000 |
| 60 s window | XGBoost 0.963, LSTM 0.963, Transformer 0.959, LR 0.953, RF 0.945 | chance = 0.013 |
| 10 s window | Fused 0.834, Transformer 0.831, XGBoost 0.827 | chance = 0.013 |

**What to say:**
- Learned models are far above simple rules: the WARN rule gets only 0.108.
- **Deep learning does not clearly beat tuned XGBoost.** The template counts in the first minute already carry most of the information. We report this honestly, as CLAUDE.md asks.
- SMOTE hurt and focal loss didn't help; class weights were enough.
- The anomaly detector alone reaches 0.78 with no labels.

### 9.2 The important twist: the trace ends (decision D-013)

At 60 s, validation PR-AUC is about 0.42 but test PR-AUC is about 0.96. We investigated:

- The test period is the **last 2 hours** of the 38.7-hour recording.
- Blocks created then had little time to live: **80 % of test faults ended within 60 seconds** (median lifetime 4 s). In validation it's 27 % (median 1.5 h).
- So for most test faults, the 60 s window *sees the whole fault* rather than predicting it. Blocks that would have failed after the recording stopped are labelled normal.
- This is called **right-censoring**. It is a property of the dataset, not a bug or leakage.

**What we did:** we added a *strict early-warning* evaluation that scores only blocks still running after the window.

| W = 60 s, strict | Validation PR-AUC | Test PR-AUC |
|---|---|---|
| XGBoost | 0.159 (base 0.034) | 0.792 (base 0.005) |
| Fused | 0.167 | 0.742 |

- Warnings for those still-running test faults came a **median of 26 minutes** early (IQR 10–41 min).
- On validation, still-running faults were barely predictable.

**So which number is the result?** The strict one. Say: *"Early-warning PR-AUC is 0.79 on the test period and 0.16 on the validation period. The standard protocol's 0.963 is inflated by right-censoring."* Never quote 0.963 on its own.

**How to present it:** as a contribution, not a weakness. "Standard evaluation on this benchmark overstates early-warning skill; here is a censoring-aware protocol." Examiners respect a team that found its own optimistic bias.

### 9.3 Scheduling (20 test seeds)

FARS is run twice on identical workloads and failures. The only difference is where the node risk scores come from (still-running blocks in both cases):

| | Best baseline | FARS, test-period scores | FARS, validation-period scores |
|---|---|---|---|
| Interrupted tasks | 129.9 | 47.2 (**−64 %**) | 117.4 (**−10 %**) |
| Lost node-minutes (all costs) | 7,638 | 6,512 (**−15 %**) | 7,515 (**−1.6 %**) |
| SLA violations | 1.0 % | 0.4 % | 0.9 % |
| Migrations | 0 | 97 | 16 |

Both FARS rows are significantly better than every baseline (Wilcoxon over 20 paired seeds):
- test-period scores: p < 0.0001;
- validation-period scores: interruptions p ≤ 0.0006, lost time p ≤ 0.022.

**How to say it:** "FARS gives a large gain when the predictor sees faults coming and a small but safe gain when it mostly doesn't. It never does worse than the baselines in either case."

**Why −64 % but only −15 %?** "Lost node-minutes" includes FARS's own costs (migrations and extra checkpoints) and the unavoidable checkpoint overhead every scheduler pays.

**Sanity and ablations:**
- **Oracle:** 24 interruptions. FARS with the real model gets 79 % of the oracle's improvement.
- **Random scores:** much worse than the baselines.
- **No calibration:** worse than the baselines in total cost.
- **No anomaly detector:** no change.
- **10 s window:** weaker, still positive.

**Noise:** if just 10 % of scores are random, FARS's total cost exceeds the baselines because false alarms cause migrations. **FARS only pays off when the predictor rarely cries wolf.**

---

## 10. How we prevented data leakage

Every item below is enforced by a test in `tests/unit/test_leakage.py` or `test_data.py`:
1. **Chronological split** by block start time, with a 30-minute gap between splits (`max(train) + gap < min(val)`).
2. The **vocabulary** of templates is learned from training data only; unseen templates become `<UNK>`.
3. **Features use only events inside the window.** The test changes later events and checks that the features don't change.
4. **SMOTE** only on training data. **Calibration, α, thresholds and τ** only on validation.
5. The **test set is read by one script** (`scripts/evaluate.py`), once.
6. Results that looked "too good" were investigated twice (D-010 full session, D-013 censoring).

---

## 11. The software: code map, tests, quality gates

| Path | What it does |
|---|---|
| `src/faultaware/data/` | parse logs, build sessions, chronological split, pandera schemas |
| `src/faultaware/features/` | windowed features, vocabulary, cached feature store |
| `src/faultaware/models/` | all 14 models behind one `FaultPredictor` interface |
| `src/faultaware/training/` | training loop, early stopping, focal loss, run orchestration |
| `src/faultaware/fusion/` | calibration and risk fusion |
| `src/faultaware/scheduler/` | FARS and the six baselines |
| `src/faultaware/simulator/` | SimPy cluster, fault model, risk signals, experiments |
| `src/faultaware/eval/` | metrics, statistics, tables, figures |
| `src/faultaware/serving.py` | turns raw log lines into a risk prediction (used by the API) |
| `scripts/` | command-line steps: download, make_dataset, tune, train, evaluate, run_simulation, package |
| `configs/` | every hyperparameter (YAML); nothing is hard-coded |
| `app/` | FastAPI server, authentication, security |
| `frontend/` | React console |
| `tests/` | 123 tests: unit, leakage, data validation, simulator sanity, API, security |

**Quality gates (`make check`):** ruff (lint and format), mypy in strict mode, and pytest with ≥ 80 % coverage (currently 93 %). CI runs the same on GitHub for every push.

---

## 12. The web console and API

**Pages:**
- **Overview:** headline numbers and the scheduler comparison.
- **Live inference:** replays real held-out test blocks through the model, or lets you paste raw log lines. Shows the risk, each component of the risk, and whether it matched the expert label.
- **Models:** the full comparison, confusion matrix, calibration, SHAP explanation, strict early-warning table and significance tests, with a 60 s / 10 s / full switch.
- **FARS scheduling:** the comparison against baselines, a node-by-time risk heat-map replay (FARS vs least-loaded), sanity checks, ablations and the noise curve.
- **Data & method:** pipeline, safeguards, simulator assumptions and team.

**Interface:**
- Light/dark toggle, collapsible sidebar, responsive on phones.
- All charts use a colour-blind-safe palette.
- Motion is reduced automatically when the operating system asks for it.

**API (all under `/api`):** `GET /health`, `POST /auth/login`, `POST /auth/logout`, `GET /auth/me`, `GET /model-info`, `POST /predict`, `GET /samples/next`, `GET /metrics`, `GET /simulation`. Everything except health and login requires a session.

---

## 13. Security

- **One account, no sign-up.** It is created on the server with `scripts/create_user.py`, and only a scrypt hash of the password is stored.
- **Sessions:** a random 256-bit token in an `HttpOnly`, `SameSite=Strict` cookie, which also gets `Secure` and the `__Host-` prefix over HTTPS. It expires after 1 h idle or 8 h total, and logout revokes it.
- **Brute force:** 5 wrong passwords in 15 minutes lock that client out for 15 minutes.
- **Cross-site attacks:** state-changing requests from another website are rejected by an Origin check, and strict security headers are set (CSP, no framing, no sniffing).
- **Transport:** the Docker deployment puts Caddy in front, which obtains HTTPS certificates automatically.

**About "end-to-end encrypted":** for a website, the correct and standard protection is HTTPS (encryption in transit) plus hashed passwords and secure sessions. "End-to-end encryption" in the WhatsApp sense would mean the server can't read the data, which is impossible here because the server must read logs to score them. Say "encrypted in transit with TLS; passwords hashed with scrypt" in the report.

---

## 14. Running, reproducing, deploying

**Run the demo locally (Windows):**
```powershell
.\make.ps1 api         # terminal 1 → API on 127.0.0.1:8000
.\make.ps1 frontend    # terminal 2 → open http://localhost:5173
```
Log in as `admin`. Change the password with `.\make.ps1 user`.

**Reproduce every number from scratch** (several GPU hours):
```powershell
.\make.ps1 setup
.\make.ps1 reproduce   # data → tune → train → eval → simulate → demo samples
```

**Deploy from the git repository:**
1. On the training laptop, run `python scripts/package_artifacts.py`. This creates `release/faultaware-artifacts.zip` plus a `.sha256` file. Upload both to a GitHub Release; they contain no code and no data, only trained models and results.
2. On the server:
   ```bash
   git clone <repo-url> faultaware && cd faultaware
   python3 scripts/install_artifacts.py faultaware-artifacts.zip
   python3 scripts/create_user.py --username admin
   DOMAIN=your.domain docker compose up -d --build
   ```
3. Without Docker, run `make serve`. It serves the site and API on port 8000. Put it behind HTTPS and set `FAULTAWARE_COOKIE_SECURE=1`.

**Never commit:** `data/`, `models/`, `secrets/`, `.env`, `release/`. The `.gitignore` already blocks them.

---

## 15. Writing the research paper

**Suggested title:** *"Early Warning Is Not Detection: Censoring-Aware Fault Prediction on HDFS Logs and Its Value for Fault-Aware Cloud Scheduling"* (or keep the project title).

**Structure (IEEE conference format, 6–8 pages):**

| Section | Content | Source in the repository |
|---|---|---|
| Abstract | task, method, key numbers: strict PR-AUC 0.79 test / 0.16 val; FARS −64 % / −10 % interruptions (test / val scores); 0.963 named as a bias | report.md abstract |
| I. Introduction | reactive vs proactive scheduling; contributions list | report §1 |
| II. Related work | log anomaly detection; failure prediction; fault-tolerant scheduling | report §2 (add verified papers) |
| III. Problem formulation | early-warning label, lead time, scheduler objective | report §3 |
| IV. Data and protocol | HDFS, split with gap, leakage safeguards, censoring | report §4, §6.2, D-013 |
| V. Method | model ladder (table), calibration and fusion, FARS (algorithm box), simulator | report §5 |
| VI. Experiments | prediction table; strict table; scheduler table; sanity; ablations; noise | `reports/tables/*.md` |
| VII. Discussion | DL vs trees; censoring; cost of false alarms; deployment | report §7 |
| VIII. Conclusion | | report §8 |

**Figures to include (already generated, 300 dpi, in `reports/figures/`):**
- `pr_curves_w60.png`, the prediction comparison;
- `reliability.png`, why calibration matters;
- `scheduler_task_interruptions.png`, the headline system result;
- `ablation_noise.png`, the sensitivity analysis;
- optionally `lead_time_hist.png` and `shap_importance.png`.

**Writing FARS as an algorithm box:**
```
Algorithm 1  FARS tick (every Δt)
Input: nodes N, risk r_i, thresholds τ_low < τ_high, smoothing β
for each node i in N:
    r̄_i ← β·r_i + (1−β)·r̄_i
    if r̄_i ≥ τ_high:  migrate tasks on i to argmax_j score_j ; checkpoint unmovable tasks
    elif r̄_i ≥ τ_low: mark i unavailable for placement ; checkpoint interval ← short
    else:             mark i available ; checkpoint interval ← default
On task arrival t:  place on argmax_{j feasible, available} (w1·free_j − w2·r̄_j − w3·mig_j)
```

**Rules for the paper:**
- Report mean ± std and p-values exactly as in the tables.
- Never round in your favour.
- Say "in simulation" whenever you state FARS gains.
- Keep the honest findings (DL ≈ XGBoost; censoring; noise sensitivity): examiners and reviewers trust them.
- Headline the **strict** numbers. Show 0.963 only next to its explanation as an evaluation bias.
- For FARS, always give both score sources (−64 % and −10 % interruptions).
- **Verify every reference** in `docs/report/references.bib` on Google Scholar. They were written from memory by an AI assistant and each is marked `VERIFY`. Fix authors, venues, years and pages.
- Don't add citations you haven't read.

---

## 16. Viva preparation: likely questions with answers

**Q: What exactly is your label, horizon and lookback window?**
A: The label is the Loghub expert label per HDFS block. We predict at 60 s after the block's creation (10 s secondary) using only events up to then. The horizon is the remaining life of the block after the window, which gives the lead time. Full-session results are a detection reference.

**Q: How did you prevent data leakage? Show the test.**
A: We used a chronological split with a 30-min gap, a train-only vocabulary, window-only features, SMOTE on train only, all tuning on validation, and one test pass. Show `tests/unit/test_leakage.py`. One test mutates events after the window and checks that the features are unchanged.

**Q: Your full-session F1 is 0.998. Isn't that leakage?**
A: We checked (D-010). The leakage tests pass, and a model trained on shuffled labels gets nowhere near it. The signal is exception messages that the experts used for labelling, so full-session HDFS detection is a known saturated benchmark. That's why our main task is early warning.

**Q: Why is test so much better than validation?**
A: The trace ends 2 hours into the test period, so test blocks' lifetimes are truncated: 80 % of test faults end within 60 s. Under the standard protocol the model therefore partly sees faults rather than predicting them, which gives the 0.963. We call that an evaluation bias. Our headline is the strict evaluation on still-running blocks: 0.79 on test and 0.16 on validation.

**Q: Does FARS only work because the test scores are easy?**
A: We tested exactly that. Driving FARS with the harder validation-period scores (same seeds, same τ) still beats every baseline significantly, but by much less: −10 % interruptions instead of −64 %, and −1.6 % lost time instead of −15 %. We report both.

**Q: Why PR-AUC and not accuracy?**
A: Faults are 1–3 % of blocks. Always predicting "normal" is 98 % accurate. PR-AUC focuses on the fault class, and its chance level equals the fault rate.

**Q: Why does FARS need calibrated probabilities?**
A: It uses fixed thresholds τ. Without calibration, the scores are compressed and every node looks medium-risk, so FARS over-checkpoints. The ablation shows total cost rising to 8,228 lost node-min, worse than the baselines.

**Q: What does a false positive cost compared with a false negative?**
A: A false positive costs one migration (30 s) or extra checkpoints. A false negative costs all work since the last checkpoint plus a restart. Per event a false negative is worse, but false positives are much more frequent because faults are rare. The noise ablation shows false alarms dominate once more than about 10 % of scores are wrong.

**Q: What happens when the predictor is wrong more often?**
A: Interruptions stay low up to 25 % noise, but total lost time exceeds the baselines from 10 % noise, because of needless migrations. FARS needs a low false-alarm rate.

**Q: Did deep learning beat classical ML?**
A: No. On the strict early-warning measure, XGBoost scores 0.792, LSTM 0.777 and Transformer 0.771; under the standard protocol they are within 0.004 of each other. Template counts in the first minute carry most of the signal, so sequence order adds little. We report it honestly. DL hyperparameters were not searched, which is a stated limitation.

**Q: What are the limitations of the simulator?**
A: The workload is synthetic, and node-level signals are drawn from block-level score distributions. Migration is idealised (30 s, no network contention). Failures are Weibull with 80 % predictable. The sanity checks bound the results, and the assumptions are listed on the console's Method page and in the report.

**Q: What is novel compared with existing work?**
A: (1) Early-warning framing of HDFS with explicit lead time. (2) Identifying and correcting the right-censoring bias at the trace end. (3) Closing the loop from a *measured* predictor to a scheduler, with calibration, paired statistics over 20 seeds, oracle/random bounds, and a noise-sensitivity curve. Most fault-aware scheduling papers assume a synthetic predictor.

**Q: How would this deploy in a real cloud?**
A: Node logs and metrics would flow into the risk service (for example via Prometheus/Fluentd). FARS rules would become a Kubernetes scheduler extender (filter and score nodes by risk) plus a controller that cordons and drains nodes above τ_high. Our Docker deployment and `/predict` API are the starting point.

**Q: Why HDFS and not the Google trace?**
A: It has expert labels, real timestamps, and fits on a laptop. The Google trace is terabytes. The trade-off is block-level rather than node-level labels, which we handle with the simulator.

**Q: Why 5 seeds for models but 20 for the scheduler?**
A: Each model run takes minutes, and 5 seeds is the CLAUDE.md minimum. Simulator runs are cheap, so 20 seeds gives much more statistical power for the main system-level claim.

**Q: Why isotonic rather than Platt calibration?**
A: Isotonic regression is non-parametric and works well with large validation sets (69 k blocks here). Tree scores aren't sigmoid-shaped, so Platt scaling fits poorly.

---

## 17. Limitations and future work

**Limitations:**
- Block-level labels and node-level scheduling in simulation.
- A short (38.7 h) single trace with strong period shift.
- A synthetic workload.
- Idealised migration.
- No DL hyperparameter search.

**Future work:**
- A multi-week trace with rolling evaluation.
- Per-node prediction from resource metrics (Google 2019, Backblaze).
- A DQN/PPO scheduler.
- Drift detection to switch FARS off when its false-alarm rate rises.
- A cost-aware training objective.

---

## 18. Glossary

- **PR-AUC:** area under the precision-recall curve. It measures ranking quality for the rare class; chance equals the positive rate.
- **ROC-AUC:** area under the ROC curve. It is less informative under heavy imbalance.
- **F1:** the harmonic mean of precision and recall at a chosen threshold.
- **FPR:** false positive rate, the share of normal blocks wrongly flagged.
- **Brier score:** mean squared error of probabilities; lower means better calibrated.
- **Calibration:** making scores match real frequencies (isotonic regression here).
- **Lead time:** how long before the end of a faulty session the warning came.
- **Right-censoring:** the end of observation cuts off events that would have happened later.
- **SMOTE:** synthetic oversampling of the minority class.
- **Focal loss:** a loss that down-weights easy examples.
- **SHAP:** a method that attributes a prediction to its input features.
- **Wilcoxon signed-rank test:** a paired, non-parametric significance test.
- **Weibull distribution:** a standard distribution for time-to-failure.
- **EWMA:** exponentially weighted moving average.
- **Checkpoint:** a saved task state, so a restart doesn't lose everything.
- **Migration:** moving a running task to another node.
- **Lost node-minutes:** wasted compute + checkpoint overhead + migration time; FARS's tuning objective.
- **Oracle:** a perfect predictor used as an upper bound.
