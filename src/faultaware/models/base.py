"""Common model interface (CLAUDE.md §6.2) and model registry."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from faultaware.features.build import FeatureSet


@runtime_checkable
class FaultPredictor(Protocol):
    """Interface every failure predictor / anomaly detector implements.

    ``X`` arguments are :class:`FeatureSet` objects so that tabular and sequence models share
    one signature; each model picks the representation it needs.
    """

    name: str

    def fit(
        self, X_train: FeatureSet, y_train: np.ndarray, X_val: FeatureSet, y_val: np.ndarray
    ) -> FaultPredictor:
        """Train on the train split; the val split may be used for early stopping only."""
        ...

    def predict_proba(self, X: FeatureSet) -> np.ndarray:
        """Return scores of shape ``(n_samples,)`` in ``[0, 1]`` (higher = more likely fault)."""
        ...

    def save(self, path: Path) -> None:
        """Persist the fitted model to ``path`` (a directory)."""
        ...

    @classmethod
    def load(cls, path: Path) -> FaultPredictor:
        """Load a model saved with :meth:`save`."""
        ...


_REGISTRY: dict[str, Callable[..., FaultPredictor]] = {}


def register(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Class decorator adding a model factory to the registry."""

    def deco(cls: Callable[..., Any]) -> Callable[..., Any]:
        _REGISTRY[name] = cls
        return cls

    return deco


def create_model(name: str, params: dict[str, Any], seed: int, **kwargs: Any) -> FaultPredictor:
    """Instantiate a registered model.

    Args:
        name: Registry key (e.g. ``"xgboost"``).
        params: Model hyperparameters from ``configs/model/models.yaml``.
        seed: Random seed.
        **kwargs: Extra constructor arguments (e.g. ``train_cfg``, ``vocab_size``).

    Returns:
        An unfitted model.

    Raises:
        KeyError: If ``name`` is not registered.
    """
    import faultaware.models.anomaly
    import faultaware.models.baselines
    import faultaware.models.classical
    import faultaware.models.deep  # noqa: F401  (populate registry)

    base = params.get("base", name)
    if base not in _REGISTRY:
        raise KeyError(f"unknown model {base!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[base](name=name, params=params, seed=seed, **kwargs)


def model_class(name: str) -> Callable[..., FaultPredictor]:
    """Return the registered class for ``name`` (to call ``.load``)."""
    import faultaware.models.anomaly
    import faultaware.models.baselines
    import faultaware.models.classical
    import faultaware.models.deep  # noqa: F401

    return _REGISTRY[name]


def check_proba(p: np.ndarray, n: int) -> np.ndarray:
    """Validate a score vector: shape ``(n,)``, finite, within ``[0, 1]``."""
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if p.shape != (n,):
        raise ValueError(f"expected shape ({n},), got {p.shape}")
    if not np.all(np.isfinite(p)) or p.min(initial=0.0) < 0 or p.max(initial=0.0) > 1:
        raise ValueError("scores must be finite and in [0, 1]")
    return p
