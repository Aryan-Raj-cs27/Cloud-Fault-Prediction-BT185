"""Report figures (PNG at 300 dpi + PDF), one consistent style.

Style rules: validated categorical palette assigned in a FIXED order per entity (colour follows
the model, never its rank), a single y-axis per chart, 2 px lines, recessive grid, legends for
>= 2 series. Status colours are not reused for series.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import precision_recall_curve

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Fixed entity -> slot mapping so a model keeps its colour across every figure.
ENTITY_SLOT = {
    "rule_warn": 0,
    "logreg": 1,
    "xgboost": 2,
    "random_forest": 3,
    "lstm": 4,
    "gru": 5,
    "cnn_lstm": 5,
    "transformer": 6,
    "lstm_ae": 7,
    "iforest": 7,
    "fused": 0,
    "least_loaded": 1,
    "reactive": 2,
    "fars": 0,
    "best_fit": 3,
    "round_robin": 4,
    "random": 5,
    "first_fit": 6,
}


def color_of(entity: str) -> str:
    """Colour of a model/scheduler (stable across figures)."""
    return PALETTE[ENTITY_SLOT.get(entity, abs(hash(entity)) % len(PALETTE))]


def _style(ax: Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=11, fontweight="bold")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)


def save(fig: Figure, out: Path) -> None:
    """Write ``out.png`` (300 dpi) and ``out.pdf``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def pr_curves(y: np.ndarray, scores: dict[str, np.ndarray], title: str, out: Path) -> None:
    """Precision-recall curves; at most 8 series (one per model family)."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, s in list(scores.items())[:8]:
        p, r, _ = precision_recall_curve(y, s)
        from sklearn.metrics import average_precision_score

        ax.plot(
            r,
            p,
            color=color_of(name),
            linewidth=2,
            label=f"{name} (AP {average_precision_score(y, s):.3f})",
        )
    ax.axhline(y.mean(), color=INK2, linewidth=1, linestyle=":", label="base rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _style(ax, "Recall (failure class)", "Precision (failure class)", title)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    save(fig, out)


def confusion_matrices(mats: dict[str, np.ndarray], title: str, out: Path) -> None:
    """Grid of 2x2 confusion matrices (rows = truth, cols = prediction), sequential blue."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    n = len(mats)
    cols = min(4, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.5 * rows), squeeze=False)
    for ax, (name, m) in zip(axes.flat, mats.items(), strict=False):
        norm = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for (i, j), v in np.ndenumerate(m):
            ax.text(
                j,
                i,
                f"{int(v):,}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if norm[i, j] > 0.55 else INK,
            )
        ax.set_xticks([0, 1], ["normal", "fault"], fontsize=7)
        ax.set_yticks([0, 1], ["normal", "fault"], fontsize=7)
        ax.set_title(name, fontsize=9, color=INK)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(title, x=0.02, ha="left", fontsize=11, color=INK)
    save(fig, out)


def training_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], out: Path) -> None:
    """Train loss and validation PR-AUC per epoch: two panels (never a dual axis)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, (loss, val) in curves.items():
        a1.plot(np.arange(1, len(loss) + 1), loss, color=color_of(name), linewidth=2, label=name)
        a2.plot(
            np.arange(1, len(val) + 1),
            val,
            color=color_of(name),
            linewidth=2,
            label=name,
            marker="o",
            markersize=3,
        )
    _style(a1, "Epoch", "Training loss", "Training loss")
    _style(a2, "Epoch", "Validation PR-AUC", "Validation PR-AUC (early-stopping metric)")
    a2.legend(fontsize=7, frameon=False)
    save(fig, out)


def reliability(y: np.ndarray, probs: dict[str, np.ndarray], out: Path) -> None:
    """Reliability diagram (quantile bins)."""
    from faultaware.eval.metrics import reliability_curve

    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ax.plot([0, 1], [0, 1], color=INK2, linestyle=":", linewidth=1, label="perfect")
    for i, (name, p) in enumerate(probs.items()):
        mp, obs, _ = reliability_curve(y, p, 10)
        ax.plot(mp, obs, color=PALETTE[i], linewidth=2, marker="o", markersize=4, label=name)
    _style(ax, "Mean predicted probability", "Observed fault rate", "Reliability (test)")
    ax.legend(fontsize=7, frameon=False)
    save(fig, out)


def histogram(values: np.ndarray, xlabel: str, title: str, out: Path, bins: int = 30) -> None:
    """Single-series histogram."""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(values, bins=bins, color=PALETTE[0], edgecolor=SURFACE, linewidth=1)
    _style(ax, xlabel, "Count", title)
    save(fig, out)


def bar_with_error(
    labels: Sequence[str],
    means: Sequence[float],
    stds: Sequence[float],
    ylabel: str,
    title: str,
    out: Path,
    highlight: str | None = None,
) -> None:
    """Bars with ±1 std error bars; the highlighted entity uses slot 1, others gray-blue."""
    fig, ax = plt.subplots(figsize=(max(5, 0.8 * len(labels) + 2), 3.8))
    colors = [PALETTE[0] if lab == highlight else "#9ec5f4" for lab in labels]
    x = np.arange(len(labels))
    ax.bar(
        x,
        means,
        yerr=stds,
        color=colors,
        width=0.6,
        capsize=3,
        error_kw={"ecolor": INK2, "elinewidth": 1},
    )
    ax.set_xticks(x, labels, rotation=20, ha="right", fontsize=8)
    _style(ax, "", ylabel, title)
    ax.xaxis.grid(False)
    save(fig, out)


def line_sweep(
    x: Sequence[float],
    series: dict[str, tuple[Sequence[float], Sequence[float]]],
    xlabel: str,
    ylabel: str,
    title: str,
    out: Path,
) -> None:
    """Line chart of a metric vs a swept parameter, mean ± std band per series."""
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for name, (m, s) in series.items():
        m_arr, s_arr = np.asarray(m), np.asarray(s)
        ax.plot(x, m_arr, color=color_of(name), linewidth=2, marker="o", markersize=4, label=name)
        ax.fill_between(
            x, m_arr - s_arr, m_arr + s_arr, color=color_of(name), alpha=0.15, linewidth=0
        )
    _style(ax, xlabel, ylabel, title)
    if len(series) > 1:
        ax.legend(fontsize=7, frameon=False)
    save(fig, out)


def heatmap(
    matrix: np.ndarray,
    xticks: Sequence[float],
    yticks: Sequence[float],
    xlabel: str,
    ylabel: str,
    title: str,
    out: Path,
    fmt: str = "{:.0f}",
) -> None:
    """Sequential heatmap with value labels (NaN cells blank)."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(xticks)), [str(v) for v in xticks])
    ax.set_yticks(range(len(yticks)), [str(v) for v in yticks])
    finite = matrix[np.isfinite(matrix)]
    mid = (finite.min() + finite.max()) / 2 if finite.size else 0
    for (i, j), v in np.ndenumerate(matrix):
        if np.isfinite(v):
            ax.text(
                j,
                i,
                fmt.format(v),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if v > mid else INK,
            )
    fig.colorbar(im, ax=ax, fraction=0.04)
    _style(ax, xlabel, ylabel, title)
    ax.grid(False)
    save(fig, out)


def class_distribution(counts: dict[str, tuple[int, int]], out: Path) -> None:
    """Fault rate per split with absolute counts annotated."""
    fig, ax = plt.subplots(figsize=(5, 3.6))
    names = list(counts)
    rates = [100 * f / max(1, n) for n, f in counts.values()]
    ax.bar(names, rates, color=PALETTE[0], width=0.55)
    for i, (n, f) in enumerate(counts.values()):
        ax.text(i, rates[i], f"{f:,} / {n:,}", ha="center", va="bottom", fontsize=8, color=INK2)
    _style(ax, "Split (chronological)", "Anomalous blocks (%)", "Class balance per split")
    save(fig, out)


def rate_over_time(
    hours: np.ndarray, rate: np.ndarray, volume: np.ndarray, boundaries: Sequence[float], out: Path
) -> None:
    """Fault rate and block volume over trace time — two stacked panels, shared x."""
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    a1.plot(hours, 100 * rate, color=PALETTE[0], linewidth=2)
    a2.bar(hours, volume, width=0.9, color="#9ec5f4")
    for b in boundaries:
        for ax in (a1, a2):
            ax.axvline(b, color=INK2, linestyle="--", linewidth=1)
    _style(a1, "", "Anomalous blocks (%)", "Fault rate over time (dashed = split boundaries)")
    _style(a2, "Hours since trace start", "New blocks / hour", "Block volume")
    save(fig, out)
