"""Prediction metrics for imbalanced fault detection (CLAUDE.md §8.1).

Accuracy is deliberately not reported as a headline metric (§7.2).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def confusion_counts(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    """Return TP/FP/FN/TN counts for binary arrays of equal shape ``(n,)``."""
    y = np.asarray(y).astype(bool)
    pred = np.asarray(pred).astype(bool)
    return {
        "tp": int(np.sum(y & pred)),
        "fp": int(np.sum(~y & pred)),
        "fn": int(np.sum(y & ~pred)),
        "tn": int(np.sum(~y & ~pred)),
    }


def _safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def threshold_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    """Precision, recall, F1 and FPR of the failure class at ``score >= threshold``."""
    c = confusion_counts(y, score >= threshold)
    precision = _safe_div(c["tp"], c["tp"] + c["fp"])
    recall = _safe_div(c["tp"], c["tp"] + c["fn"])
    return {
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "fpr": _safe_div(c["fp"], c["fp"] + c["tn"]),
        **{k: float(v) for k, v in c.items()},
    }


def ranking_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """PR-AUC (primary), ROC-AUC and Brier score. Degenerate inputs give NaN AUCs."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return {
            "pr_auc": float("nan"),
            "roc_auc": float("nan"),
            "brier": float(brier_score_loss(y, score, pos_label=1)),
        }
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "brier": float(brier_score_loss(y, score)),
    }


def best_threshold(
    y: np.ndarray,
    score: np.ndarray,
    objective: str = "f1",
    fp_cost: float = 1.0,
    fn_cost: float = 10.0,
) -> float:
    """Choose a decision threshold on VALIDATION data.

    Args:
        y: Validation labels ``(n,)``.
        score: Validation scores ``(n,)``.
        objective: ``"f1"`` (maximise F1) or ``"cost"`` (minimise ``fp_cost·FP + fn_cost·FN``).
        fp_cost: Cost of a false alarm.
        fn_cost: Cost of a missed fault.

    Returns:
        The threshold (predict fault iff ``score >= threshold``).
    """
    y = np.asarray(y).astype(np.int64)
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(-score, kind="stable")
    s_sorted, y_sorted = score[order], y[order]
    # Candidate thresholds = distinct score values; predicting the top-k as positive.
    distinct = np.r_[np.flatnonzero(np.diff(s_sorted) != 0), len(s_sorted) - 1]
    tp = np.cumsum(y_sorted)[distinct]
    fp = (distinct + 1) - tp
    pos = y.sum()
    fn = pos - tp
    if objective == "f1":
        f1 = np.where(tp > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        best = int(np.argmax(f1))
    elif objective == "cost":
        best = int(np.argmin(fp_cost * fp + fn_cost * fn))
    else:
        raise ValueError(f"unknown objective {objective!r}")
    return float(s_sorted[distinct[best]])


def evaluate_scores(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    """All prediction metrics at a fixed (validation-chosen) threshold."""
    return {
        **ranking_metrics(y, score),
        **threshold_metrics(y, score, threshold),
        "threshold": float(threshold),
    }


def bootstrap_ci(
    y: np.ndarray,
    score: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of ``metric(y, score)``."""
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(metric(y[idx], score[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def reliability_curve(
    y: np.ndarray, prob: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reliability-diagram data with quantile bins.

    Returns:
        ``(mean_predicted, observed_rate, counts)``, each of shape ``(<= n_bins,)``.
    """
    edges = np.unique(np.quantile(prob, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return np.array([prob.mean()]), np.array([y.mean()]), np.array([len(y)])
    bins = np.clip(np.searchsorted(edges, prob, side="right") - 1, 0, len(edges) - 2)
    mp, obs, cnt = [], [], []
    for b in range(len(edges) - 1):
        m = bins == b
        if m.any():
            mp.append(prob[m].mean())
            obs.append(y[m].mean())
            cnt.append(m.sum())
    return np.array(mp), np.array(obs), np.array(cnt)


def alive_mask(lead_seconds: np.ndarray) -> np.ndarray:
    """Sessions still running when the observation window closes.

    A block whose whole life fits inside the window is *detected*, not *predicted*. Restricting
    evaluation to ``lead_seconds > 0`` measures genuine early warning and removes the
    right-censoring bias at the end of a finite trace (see docs/DECISIONS.md D-013).

    Args:
        lead_seconds: ``last_ts - (first_ts + W)`` clipped at 0, shape ``(n,)``.

    Returns:
        Boolean mask of shape ``(n,)``.
    """
    return np.asarray(lead_seconds) > 0


def evaluate_early_warning(
    y: np.ndarray, score: np.ndarray, lead_seconds: np.ndarray, threshold: float
) -> dict[str, float]:
    """:func:`evaluate_scores` on the sessions still alive after the window, plus their counts."""
    mask = alive_mask(lead_seconds)
    y_sub, s_sub = np.asarray(y)[mask], np.asarray(score)[mask]
    return {
        **evaluate_scores(y_sub, s_sub, threshold),
        "n_alive": float(mask.sum()),
        "n_alive_faults": float(np.sum(y_sub)),
    }
