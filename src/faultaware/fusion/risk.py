"""Calibrated fault-risk score (CLAUDE.md §2.1 step 2).

``r = alpha * p_cal + (1 - alpha) * a`` where ``p_cal`` is the calibrated supervised failure
probability and ``a`` the normalised anomaly score. The calibrator, ``alpha`` and the decision
threshold are all fitted on the VALIDATION split; the test split is never used here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


class Calibrator:
    """Maps raw scores to calibrated probabilities.

    Args:
        method: ``"isotonic"``, ``"sigmoid"`` (Platt scaling) or ``"none"``.
    """

    def __init__(self, method: str = "isotonic") -> None:
        if method not in {"isotonic", "sigmoid", "none"}:
            raise ValueError(f"unknown calibration method {method!r}")
        self.method = method
        self._iso: IsotonicRegression | None = None
        self._lr: LogisticRegression | None = None

    def fit(self, score: np.ndarray, y: np.ndarray) -> Calibrator:
        """Fit on validation scores ``(n,)`` and labels ``(n,)``."""
        if self.method == "isotonic":
            self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._iso.fit(score, y)
        elif self.method == "sigmoid":
            self._lr = LogisticRegression().fit(_logit(score).reshape(-1, 1), y)
        return self

    def __call__(self, score: np.ndarray) -> np.ndarray:
        score = np.asarray(score, dtype=np.float64)
        if self.method == "isotonic":
            assert self._iso is not None
            return np.clip(self._iso.predict(score), 0.0, 1.0)
        if self.method == "sigmoid":
            assert self._lr is not None
            return self._lr.predict_proba(_logit(score).reshape(-1, 1))[:, 1]
        return np.clip(score, 0.0, 1.0)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fuse(p_cal: np.ndarray, anomaly: np.ndarray, alpha: float) -> np.ndarray:
    """Convex combination of calibrated probability and anomaly score.

    Args:
        p_cal: Calibrated probabilities in ``[0, 1]``, shape ``(n,)``.
        anomaly: Normalised anomaly scores in ``[0, 1]``, shape ``(n,)``.
        alpha: Weight of the supervised probability, in ``[0, 1]``.

    Returns:
        Risk scores in ``[0, 1]``, shape ``(n,)``.

    Raises:
        ValueError: If ``alpha`` is outside ``[0, 1]`` or shapes differ.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    p_cal, anomaly = np.asarray(p_cal, float), np.asarray(anomaly, float)
    if p_cal.shape != anomaly.shape:
        raise ValueError("shape mismatch")
    return np.clip(alpha * p_cal + (1.0 - alpha) * anomaly, 0.0, 1.0)


@dataclass
class RiskFusion:
    """Fitted fusion: calibrator + alpha (+ the names of the fused models)."""

    supervised_model: str
    anomaly_model: str
    calibration: str = "isotonic"
    alpha: float = 1.0
    threshold: float = 0.5
    calibrator: Calibrator = field(default_factory=Calibrator)
    alpha_curve: dict[float, float] = field(default_factory=dict)

    def fit(
        self, p_val: np.ndarray, a_val: np.ndarray, y_val: np.ndarray, alpha_grid: list[float]
    ) -> RiskFusion:
        """Fit calibrator and choose ``alpha`` maximising validation PR-AUC.

        Ties are broken toward larger ``alpha`` (simpler, supervised-dominant model).
        """
        self.calibrator = Calibrator(self.calibration).fit(p_val, y_val)
        p_cal = self.calibrator(p_val)
        best_alpha, best = 1.0, -1.0
        for alpha in sorted(alpha_grid, reverse=True):
            score = float(average_precision_score(y_val, fuse(p_cal, a_val, alpha)))
            self.alpha_curve[float(alpha)] = score
            if score > best + 1e-6:
                best_alpha, best = float(alpha), score
        self.alpha = best_alpha
        return self

    def __call__(self, p_raw: np.ndarray, anomaly: np.ndarray) -> np.ndarray:
        """Risk scores from raw supervised probabilities and anomaly scores."""
        return fuse(self.calibrator(p_raw), anomaly, self.alpha)

    def save(self, path: Path) -> None:
        """Persist with joblib (trusted, project-produced artefact)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> RiskFusion:
        """Load a fusion saved with :meth:`save`."""
        obj = joblib.load(path)
        if not isinstance(obj, RiskFusion):
            raise TypeError("not a RiskFusion artefact")
        return obj
