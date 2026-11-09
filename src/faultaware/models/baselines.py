"""Trivial baselines (model ladder step 1, CLAUDE.md §7.3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from faultaware.features.build import STAT_NAMES, FeatureSet
from faultaware.models.base import register


class _Baseline:
    """Shared save/load for parameter-light baselines."""

    def __init__(
        self, name: str, params: dict[str, Any] | None = None, seed: int = 0, **_: Any
    ) -> None:
        self.name = name
        self.params = dict(params or {})
        self.seed = seed
        self.state: dict[str, float] = {}

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "baseline.json").write_text(
            json.dumps({"name": self.name, "seed": self.seed, "state": self.state}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> _Baseline:
        data = json.loads((path / "baseline.json").read_text(encoding="utf-8"))
        obj = cls(name=data["name"], seed=data["seed"])
        obj.state = data["state"]
        return obj


@register("always_zero")
class AlwaysZero(_Baseline):
    """Predicts 'no fault' for every sample (score 0)."""

    def fit(
        self, X_train: FeatureSet, y_train: np.ndarray, X_val: FeatureSet, y_val: np.ndarray
    ) -> AlwaysZero:
        return self

    def predict_proba(self, X: FeatureSet) -> np.ndarray:
        return np.zeros(len(X.y), dtype=np.float64)


@register("random_rate")
class RandomRate(_Baseline):
    """Random scores; at threshold 1 - base_rate it alarms at the training fault rate."""

    def fit(
        self, X_train: FeatureSet, y_train: np.ndarray, X_val: FeatureSet, y_val: np.ndarray
    ) -> RandomRate:
        self.state["base_rate"] = float(np.mean(y_train))
        return self

    def predict_proba(self, X: FeatureSet) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.random(len(X.y))


@register("rule_warn")
class RuleWarn(_Baseline):
    """Simple threshold rule: flag a block if it logged any WARN/ERROR line.

    The score is ``min(1, n_warn / cap)`` so that ranking metrics remain defined; with the
    default threshold of 0.5·(1/cap) it is equivalent to ``n_warn >= 1``.
    """

    def fit(
        self, X_train: FeatureSet, y_train: np.ndarray, X_val: FeatureSet, y_val: np.ndarray
    ) -> RuleWarn:
        self.state["cap"] = float(max(1.0, np.percentile(X_train.stats[:, _WARN], 99)))
        return self

    def predict_proba(self, X: FeatureSet) -> np.ndarray:
        cap = self.state.get("cap", 1.0)
        return np.clip(X.stats[:, _WARN] / cap, 0.0, 1.0).astype(np.float64)


_WARN = STAT_NAMES.index("n_warn")
