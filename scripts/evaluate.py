"""Final evaluation on the held-out TEST split (run once, after all tuning is finished).

Produces, for every observation window:
  * reports/tables/prediction_<window>.{csv,md}   test metrics, mean ± std over seeds
  * reports/tables/significance_<window>.{csv,md} paired tests between key models
  * fusion artefacts (models/fusion/<window>/) — calibrator + alpha fitted on VALIDATION
  * score distributions consumed by the simulator (models/fusion/score_distributions*.npz)
plus figures in reports/figures and reports/results_summary.json (served by the API).

Usage:
    python scripts/evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from faultaware.eval import plots
from faultaware.eval.metrics import (
    alive_mask,
    best_threshold,
    bootstrap_ci,
    confusion_counts,
    evaluate_early_warning,
    evaluate_scores,
)
from faultaware.eval.stats import fmt_mean_std, mean_std, paired_tests
from faultaware.eval.tables import write_table
from faultaware.features.build import tabular_feature_names
from faultaware.features.store import get_features, load_bundle, window_tag
from faultaware.fusion.risk import RiskFusion
from faultaware.training.runner import artefact_dir, preds_path
from faultaware.utils.config import load_config, project_path
from faultaware.utils.logging import get_logger

log = get_logger("evaluate")

TRIVIAL = ["always_zero", "random_rate", "rule_warn"]
CLASSICAL = ["logreg", "random_forest", "xgboost", "random_forest_smote"]
DEEP = ["lstm", "gru", "cnn_lstm", "transformer", "lstm_focal"]
ANOMALY = ["iforest", "lstm_ae"]
FAMILY = {
    **dict.fromkeys(TRIVIAL, "trivial"),
    **dict.fromkeys(CLASSICAL, "classical ML"),
    **dict.fromkeys(DEEP, "deep learning"),
    **dict.fromkeys(ANOMALY, "unsupervised"),
}
METRICS = ["pr_auc", "roc_auc", "f1", "precision", "recall", "fpr", "brier"]


def load_preds(cfg: Any, tag: str, model: str, seeds: list[int]) -> list[dict[str, Any]]:
    """Load stored val/test scores for each available seed."""
    out = []
    for seed in seeds:
        path = preds_path(cfg, tag, model, seed)
        if path.is_file():
            with np.load(path) as z:
                out.append({k: z[k] for k in z.files} | {"seed": seed})
    return out


def evaluate_window(
    cfg: Any, bundle: Any, window: float | None, seeds: list[int], figs: Path, tables: Path
) -> dict[str, Any]:
    """Evaluate all models + fusion for one observation window."""
    tag = window_tag(window)
    val = get_features(cfg, bundle, "val", window)
    test = get_features(cfg, bundle, "test", window)
    rows, per_seed, seed0 = [], {}, {}
    for model in TRIVIAL + CLASSICAL + DEEP + ANOMALY:
        runs = load_preds(cfg, tag, model, seeds)
        if not runs:
            continue
        mets = [evaluate_scores(test.y, r["test"], float(r["threshold"])) for r in runs]
        val_ap = [evaluate_scores(val.y, r["val"], float(r["threshold"]))["pr_auc"] for r in runs]
        per_seed[model] = {"test": mets, "val_pr_auc": val_ap, "runs": runs}
        seed0[model] = runs[0]
        rows.append(
            _row(model, FAMILY[model], mets, val_ap, [float(r["latency_ms"]) for r in runs])
        )

    def best_of(names: list[str]) -> str:
        avail = [n for n in names if n in per_seed]
        return max(avail, key=lambda n: np.mean(per_seed[n]["val_pr_auc"]))

    best_sup = (
        cfg.fusion.supervised_model
        if cfg.fusion.supervised_model != "auto"
        else best_of(CLASSICAL + DEEP)
    )
    best_anom = cfg.fusion.anomaly_model if cfg.fusion.anomaly_model != "auto" else best_of(ANOMALY)
    best_ml, best_dl = best_of(CLASSICAL), best_of(DEEP)
    log.info(
        "[%s] best supervised=%s anomaly=%s ML=%s DL=%s", tag, best_sup, best_anom, best_ml, best_dl
    )

    # ---- fusion per seed (fitted on validation only)
    fused_mets, cal_mets, fused_test, fusions = [], [], {}, {}
    sup_runs, anom_runs = per_seed[best_sup]["runs"], per_seed[best_anom]["runs"]
    for rs, ra in zip(sup_runs, anom_runs, strict=False):
        fusion = RiskFusion(best_sup, best_anom, cfg.fusion.calibration).fit(
            rs["val"], ra["val"], val.y, list(cfg.fusion.alpha_grid)
        )
        fval = fusion(rs["val"], ra["val"])
        fusion.threshold = best_threshold(val.y, fval, cfg.fusion.threshold_objective)
        ftest = fusion(rs["test"], ra["test"])
        fused_mets.append(evaluate_scores(test.y, ftest, fusion.threshold))
        cval = fusion.calibrator(rs["val"])
        cal_mets.append(
            evaluate_scores(test.y, fusion.calibrator(rs["test"]), best_threshold(val.y, cval))
        )
        fused_test[rs["seed"]] = (fval, ftest)
        fusions[rs["seed"]] = fusion
    rows.append(
        _row(
            "fused",
            "fusion",
            fused_mets,
            [max(f.alpha_curve.values()) for f in fusions.values()],
            [np.nan],
        )
    )
    rows.append(_row(f"{best_sup}+calibration", "fusion", cal_mets, [np.nan], [np.nan]))
    table = pd.DataFrame(rows)
    write_table(table, tables / f"prediction_{tag}")
    early = (
        _early_warning_table(per_seed, fused_mets_inputs(fusions, sup_runs, anom_runs), val, test)
        if window is not None
        else None
    )
    if early is not None:
        write_table(early, tables / f"early_warning_{tag}")

    # ---- significance (paired over seeds, test PR-AUC)
    def aps(name: str) -> list[float]:
        return [m["pr_auc"] for m in per_seed[name]["test"]]

    comparisons = [
        (best_dl, best_ml),
        ("fused", best_sup),
        (best_sup, "rule_warn"),
        (best_anom, "random_rate"),
    ]
    sig_rows = []
    for a, b in comparisons:
        xa = [m["pr_auc"] for m in fused_mets] if a == "fused" else aps(a)
        xb = aps(b)
        n = min(len(xa), len(xb))
        test_res = paired_tests(xa[:n], xb[:n])
        lo_a, hi_a = bootstrap_ci(
            test.y, _scores_for(a, seed0, fused_test, sup_runs), _ap, n_boot=300, seed=0
        )
        sig_rows.append(
            {
                "A": a,
                "B": b,
                "metric": "test PR-AUC",
                "n_seeds": n,
                "mean_A": mean_std(xa)[0],
                "mean_B": mean_std(xb)[0],
                **test_res,
                "A_seed0_95CI": f"[{lo_a:.3f}, {hi_a:.3f}]",
            }
        )
    write_table(pd.DataFrame(sig_rows), tables / f"significance_{tag}")

    # ---- persist fusion (seed 0) + simulator score distributions
    first_seed = sup_runs[0]["seed"]
    fusion0 = fusions[first_seed]
    fdir = project_path(cfg.paths.models_dir) / "fusion" / tag
    fusion0.save(fdir / "fusion.joblib")
    (fdir / "fusion.json").write_text(
        json.dumps(
            {
                "supervised_model": best_sup,
                "anomaly_model": best_anom,
                "alpha": fusion0.alpha,
                "threshold": fusion0.threshold,
                "calibration": fusion0.calibration,
                "alpha_curve": fusion0.alpha_curve,
                "observe_seconds": window,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    dist: dict[str, np.ndarray] = {}
    rs0 = sup_runs[0]
    for split, fs, raw in (("val", val, rs0["val"]), ("test", test, rs0["test"])):
        fused = fused_test[first_seed][0 if split == "val" else 1]
        cal = fusion0.calibrator(raw)
        # The simulator's precursor signal must be a *prediction*: with a finite window, keep only
        # sessions still alive after it (blocks that already ended were detected, not predicted).
        keep = alive_mask(fs.lead_seconds) if window is not None else np.ones(len(fs.y), bool)
        y = fs.y
        for variant, s in (("fused", fused), ("supervised_cal", cal), ("supervised_raw", raw)):
            dist[f"{split}_{variant}_pos"] = s[(y == 1) & keep].astype(np.float32)
            dist[f"{split}_{variant}_neg"] = s[(y == 0) & keep].astype(np.float32)
    dist_name = (
        "score_distributions.npz"
        if window == cfg.data.default_observation
        else f"score_distributions_{tag}.npz"
    )
    np.savez_compressed(fdir.parent / dist_name, **dist)

    # ---- figures
    curve_models = [
        m
        for m in (
            "rule_warn",
            "logreg",
            "xgboost",
            "random_forest",
            "lstm",
            "transformer",
            "lstm_ae",
        )
        if m in seed0
    ]
    scores = {m: seed0[m]["test"] for m in curve_models} | {"fused": fused_test[first_seed][1]}
    plots.pr_curves(
        test.y, scores, f"Precision-recall, test split ({tag})", figs / f"pr_curves_{tag}"
    )
    mats = {}
    for m in [*curve_models, "fused"]:
        s = scores[m]
        thr = fusion0.threshold if m == "fused" else float(seed0[m]["threshold"])
        c = confusion_counts(test.y, s >= thr)
        mats[m] = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    plots.confusion_matrices(
        mats, f"Confusion matrices, test ({tag}, validation thresholds)", figs / f"confusion_{tag}"
    )
    summary = {
        "window": tag,
        "observe_seconds": window,
        "best_supervised": best_sup,
        "best_anomaly": best_anom,
        "best_classical": best_ml,
        "best_deep": best_dl,
        "fusion": {"alpha": fusion0.alpha, "threshold": fusion0.threshold},
        "table": table.to_dict(orient="records"),
        "early_warning": None if early is None else early.to_dict(orient="records"),
        "significance": sig_rows,
        "confusion": {m: mats[m].tolist() for m in mats},
    }
    if window == cfg.data.default_observation:
        _primary_figures(
            cfg,
            bundle,
            test,
            val,
            seed0,
            best_sup,
            fusion0,
            fused_test[first_seed][1],
            figs,
            summary,
        )
    return summary


def _ap(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, s))


def _scores_for(
    name: str, seed0: dict[str, Any], fused_test: dict[int, Any], sup_runs: list[dict[str, Any]]
) -> np.ndarray:
    if name == "fused":
        return fused_test[sup_runs[0]["seed"]][1]
    return seed0[name]["test"]


def fused_mets_inputs(
    fusions: dict[int, Any], sup_runs: list[dict[str, Any]], anom_runs: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Per-seed fused val/test scores and threshold, shaped like a stored prediction run."""
    out = []
    for rs, ra in zip(sup_runs, anom_runs, strict=False):
        fusion = fusions[rs["seed"]]
        out.append(
            {
                "val": fusion(rs["val"], ra["val"]),
                "test": fusion(rs["test"], ra["test"]),
                "threshold": fusion.threshold,
            }
        )
    return out


def _early_warning_table(
    per_seed: dict[str, Any], fused_runs: list[dict[str, Any]], val: Any, test: Any
) -> pd.DataFrame:
    """Metrics on sessions still alive after the window (genuine prediction, D-013)."""
    rows = []
    runs_by_model = {m: v["runs"] for m, v in per_seed.items()} | {"fused": fused_runs}
    for model, runs in runs_by_model.items():
        tm = [
            evaluate_early_warning(test.y, r["test"], test.lead_seconds, float(r["threshold"]))
            for r in runs
        ]
        vm = [
            evaluate_early_warning(val.y, r["val"], val.lead_seconds, float(r["threshold"]))
            for r in runs
        ]
        rows.append(
            {
                "model": model,
                "family": FAMILY.get(model, "fusion"),
                "val_pr_auc": fmt_mean_std([m["pr_auc"] for m in vm]),
                "test_pr_auc": fmt_mean_std([m["pr_auc"] for m in tm]),
                "test_f1": fmt_mean_std([m["f1"] for m in tm]),
                "test_precision": fmt_mean_std([m["precision"] for m in tm]),
                "test_recall": fmt_mean_std([m["recall"] for m in tm]),
                "test_pr_auc_mean": mean_std([m["pr_auc"] for m in tm])[0],
                "val_alive": f"{int(vm[0]['n_alive'])} ({int(vm[0]['n_alive_faults'])} faults)",
                "test_alive": f"{int(tm[0]['n_alive'])} ({int(tm[0]['n_alive_faults'])} faults)",
            }
        )
    return pd.DataFrame(rows)


def _row(
    model: str, family: str, mets: list[dict[str, float]], val_ap: list[float], latency: list[float]
) -> dict[str, Any]:
    row: dict[str, Any] = {"model": model, "family": family, "seeds": len(mets)}
    for k in METRICS:
        row[f"test_{k}"] = fmt_mean_std([m[k] for m in mets])
    row["test_pr_auc_mean"] = mean_std([m["pr_auc"] for m in mets])[0]
    row["test_f1_mean"] = mean_std([m["f1"] for m in mets])[0]
    finite_val = [v for v in val_ap if np.isfinite(v)]
    row["val_pr_auc"] = fmt_mean_std(finite_val) if finite_val else "n/a"
    finite_lat = [v for v in latency if np.isfinite(v)]
    row["latency_ms_per_sample"] = round(float(np.mean(finite_lat)), 4) if finite_lat else "n/a"
    return row


def _primary_figures(
    cfg: Any,
    bundle: Any,
    test: Any,
    val: Any,
    seed0: dict[str, Any],
    best_sup: str,
    fusion: RiskFusion,
    fused: np.ndarray,
    figs: Path,
    summary: dict[str, Any],
) -> None:
    curves = {
        m: (seed0[m]["train_loss"], seed0[m]["val_curve"])
        for m in ("lstm", "gru", "cnn_lstm", "transformer")
        if m in seed0
    }
    if curves:
        plots.training_curves(curves, figs / "training_curves")
    raw = seed0[best_sup]["test"]
    plots.reliability(
        test.y,
        {
            f"{best_sup} (raw)": raw,
            f"{best_sup} (isotonic)": fusion.calibrator(raw),
            "fused risk": fused,
        },
        figs / "reliability",
    )
    from faultaware.eval.metrics import ranking_metrics

    summary["calibration"] = {
        "brier_raw": ranking_metrics(test.y, raw)["brier"],
        "brier_calibrated": ranking_metrics(test.y, fusion.calibrator(raw))["brier"],
        "brier_fused": ranking_metrics(test.y, fused)["brier"],
    }
    tp = (test.y == 1) & (fused >= fusion.threshold)
    # Lead time is only meaningful for faults still running after the window (D-013).
    alive_tp = tp & alive_mask(test.lead_seconds)
    lead_h = test.lead_seconds[alive_tp] / 3600.0
    plots.histogram(
        lead_h,
        "Lead time before block session ends (hours)",
        "Lead time of correct early warnings (test, 60 s window, alive blocks)",
        figs / "lead_time_hist",
    )
    summary["lead_time_hours"] = {
        "n_true_positives": int(tp.sum()),
        "n_true_positives_alive": int(alive_tp.sum()),
        "median": float(np.median(lead_h)) if lead_h.size else 0.0,
        "p25": float(np.percentile(lead_h, 25)) if lead_h.size else 0.0,
        "p75": float(np.percentile(lead_h, 75)) if lead_h.size else 0.0,
    }
    plots.line_sweep(
        list(fusion.alpha_curve),
        {"fused": (list(fusion.alpha_curve.values()), [0.0] * len(fusion.alpha_curve))},
        "alpha (weight of supervised probability)",
        "Validation PR-AUC",
        "Fusion weight selection (validation)",
        figs / "fusion_alpha",
    )
    _shap(cfg, bundle, test, figs, summary)


def _shap(cfg: Any, bundle: Any, test: Any, figs: Path, summary: dict[str, Any]) -> None:
    """SHAP global importance + example explanations of real predicted failures (XGBoost)."""
    import shap

    from faultaware.models.classical import XGBoostModel

    adir = artefact_dir(cfg, window_tag(cfg.data.default_observation), "xgboost")
    if not adir.is_dir():
        return
    model = XGBoostModel.load(adir)
    names = tabular_feature_names(bundle.vocab)
    short = [_short(n) for n in names]
    rng = np.random.default_rng(0)
    pos = np.flatnonzero(test.y == 1)
    idx = np.r_[rng.choice(len(test.y), 3000, replace=False), pos[:500]]
    X = test.tabular[idx]
    explainer = shap.TreeExplainer(model.estimator)
    values = explainer.shap_values(X)
    imp = np.abs(values).mean(0)
    order = np.argsort(-imp)[:15]
    plots.bar_with_error(
        [short[i] for i in order][::-1],
        imp[order][::-1],
        [0.0] * len(order),
        "mean |SHAP value|",
        "XGBoost global feature importance (SHAP, test sample)",
        figs / "shap_importance",
    )
    proba = model.predict_proba(_sub(test, idx))
    examples = []
    thr = float(np.load(preds_path(cfg, adir.parent.name, "xgboost", 0))["threshold"])
    tp_rows = [i for i in range(len(idx)) if test.y[idx[i]] == 1 and proba[i] >= thr][:3]
    for i in tp_rows:
        top = np.argsort(-np.abs(values[i]))[:5]
        examples.append(
            {
                "probability": float(proba[i]),
                "top_features": [
                    {"feature": names[j], "value": float(X[i, j]), "shap": float(values[i, j])}
                    for j in top
                ],
            }
        )
    summary["shap"] = {
        "global_top": [{"feature": names[i], "mean_abs_shap": float(imp[i])} for i in order[:10]],
        "examples": examples,
    }


def _short(name: str) -> str:
    text = name.replace("count[", "").rstrip("]")
    for prefix in ("INFO ", "WARN "):
        text = text.replace(prefix, "")
    return text if len(text) <= 48 else text[:45] + "..."


def _sub(fs: Any, idx: np.ndarray) -> Any:
    from faultaware.features.build import FeatureSet

    return FeatureSet(
        fs.counts[idx],
        fs.stats[idx],
        fs.sequences[idx],
        fs.lengths[idx],
        fs.y[idx],
        fs.lead_seconds[idx],
    )


def eda_figures(cfg: Any, bundle: Any, figs: Path) -> dict[str, Any]:
    """Class distribution per split and fault rate over time."""
    s = bundle.sessions
    counts = {
        sp: (int((s.split == sp).sum()), int(s.loc[s.split == sp, "label"].sum()))
        for sp in ("train", "val", "test")
    }
    plots.class_distribution(counts, figs / "class_distribution")
    t0 = s.first_ts.min()
    hour = ((s.first_ts - t0) // 3600).astype(int)
    grp = s.groupby(hour)["label"].agg(["mean", "size"])
    split_info = json.loads(
        (project_path(cfg.data.sessions_file).parent / "split_summary.json").read_text("utf-8")
    )
    bounds = [
        (split_info["boundaries"]["train_end"] - t0) / 3600,
        (split_info["boundaries"]["val_end"] - t0) / 3600,
    ]
    plots.rate_over_time(
        grp.index.to_numpy(),
        grp["mean"].to_numpy(),
        grp["size"].to_numpy(),
        bounds,
        figs / "fault_rate_over_time",
    )
    return {"split_counts": counts, "split_summary": split_info}


def censoring_stats(sessions: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Fault lifetimes per split: the trace end truncates long-lived blocks in test (D-013)."""
    out = {}
    for split, grp in sessions.groupby("split"):
        dur = (grp["last_ts"] - grp["first_ts"]).to_numpy()
        faults = grp["label"].to_numpy() == 1
        out[str(split)] = {
            "fault_rate": float(faults.mean()),
            "faults_ending_within_60s": float(np.mean(dur[faults] <= 60)),
            "faults_ending_within_10s": float(np.mean(dur[faults] <= 10)),
            "normals_ending_within_60s": float(np.mean(dur[~faults] <= 60)),
            "fault_median_lifetime_s": float(np.median(dur[faults])),
        }
    return out


def main() -> None:
    """CLI entry point."""
    cfg = load_config()
    bundle = load_bundle(cfg)
    seeds = list(cfg.seeds_for_reporting)
    figs = project_path(cfg.paths.figures_dir)
    tables = project_path(cfg.paths.tables_dir)
    results: dict[str, Any] = {"data": eda_figures(cfg, bundle, figs), "windows": {}}
    for window in cfg.data.observation_windows:
        results["windows"][window_tag(window)] = evaluate_window(
            cfg, bundle, window, seeds, figs, tables
        )
    results["primary_window"] = window_tag(cfg.data.default_observation)
    results["censoring"] = censoring_stats(bundle.sessions)
    out = project_path(cfg.paths.reports_dir) / "results_summary.json"
    out.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
