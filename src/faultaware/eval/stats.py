"""Statistical comparison helpers (CLAUDE.md §8.3)."""

from __future__ import annotations

import numpy as np
from scipy import stats


def mean_std(values: np.ndarray | list[float]) -> tuple[float, float]:
    """Mean and sample standard deviation (ddof=1; 0 for a single value)."""
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def fmt_mean_std(values: np.ndarray | list[float], digits: int = 3) -> str:
    """Format as ``"mean ± std"``."""
    m, s = mean_std(values)
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def paired_tests(a: np.ndarray | list[float], b: np.ndarray | list[float]) -> dict[str, float]:
    """Paired comparison of two methods evaluated on the same seeds.

    Args:
        a: Metric of method A per seed, shape ``(n,)``.
        b: Metric of method B per seed (same seeds, same order).

    Returns:
        Mean difference ``a - b``, Wilcoxon signed-rank p-value and paired t-test p-value
        (two-sided). With ``n = 5`` the smallest attainable Wilcoxon p is 0.0625, which is why
        both tests are reported.
    """
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("paired samples must have equal shape")
    diff = x - y
    if np.allclose(diff, 0):
        return {"mean_diff": 0.0, "wilcoxon_p": 1.0, "ttest_p": 1.0}
    w_p = float(stats.wilcoxon(x, y, zero_method="zsplit").pvalue)
    t_p = float(stats.ttest_rel(x, y).pvalue) if np.std(diff) > 0 else 0.0
    return {"mean_diff": float(diff.mean()), "wilcoxon_p": w_p, "ttest_p": t_p}
