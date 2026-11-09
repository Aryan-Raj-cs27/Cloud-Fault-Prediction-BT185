"""Classical ML predictors on tabular count features (model ladder steps 2–3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from faultaware.features.build import FeatureSet
from faultaware.models.base import register


def _resample(
    X: np.ndarray, y: np.ndarray, imbalance: str, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to the TRAINING data only (never called on val/test)."""
    if imbalance != "smote":
        return X, y
    from imblearn.over_sampling import SMOTE

    return SMOTE(random_state=seed).fit_resample(X, y)


class _SklearnModel:
    """Wrapper giving scikit-learn style estimators the FaultPredictor interface."""

    def __init__(self, name: str, params: dict[str, Any], seed: int, **_: Any) -> None:
        self.name = name
        self.params = dict(params)
        self.seed = seed
        self.estimator: Any = None

    def _build(self, pos_weight: float) -> Any:
        raise NotImplementedError

    def fit(
        self, X_train: FeatureSet, y_train: np.ndarray, X_val: FeatureSet, y_val: np.ndarray
    ) -> _SklearnModel:
        imbalance = self.params.get("imbalance", "class_weight")
        X, y = _resample(X_train.tabular, y_train, imbalance, self.seed)
        pos = max(1, int(y.sum()))
        pos_weight = (len(y) - pos) / pos if imbalance == "class_weight" else 1.0
        self.estimator = self._build(pos_weight)
        self._fit(X, y, X_val.tabular, y_val)
        return self

    def _fit(self, X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray) -> None:
        self.estimator.fit(X, y)

    def predict_proba(self, X: FeatureSet) -> np.ndarray:
        return self.estimator.predict_proba(X.tabular)[:, 1].astype(np.float64)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.estimator, path / "estimator.joblib")
        (path / "meta.json").write_text(
            json.dumps({"name": self.name, "params": self.params, "seed": self.seed}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> _SklearnModel:
        # joblib uses pickle: only load artefacts produced by this project's own pipeline.
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        obj = cls(name=meta["name"], params=meta["params"], seed=meta["seed"])
        obj.estimator = joblib.load(path / "estimator.joblib")
        return obj


@register("logreg")
class LogRegModel(_SklearnModel):
    """Logistic regression on ``log1p`` counts, standardised (scaler fitted on train only)."""

    def _build(self, pos_weight: float) -> Any:
        weights = {0: 1.0, 1: pos_weight} if pos_weight != 1.0 else None
        return Pipeline(
            [
                ("log1p", FunctionTransformer(np.log1p)),
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=self.params.get("C", 1.0),
                        max_iter=self.params.get("max_iter", 2000),
                        class_weight=weights,
                        random_state=self.seed,
                    ),
                ),
            ]
        )


@register("random_forest")
class RandomForestModel(_SklearnModel):
    """Random forest with balanced class weights."""

    def _build(self, pos_weight: float) -> Any:
        return RandomForestClassifier(
            n_estimators=self.params.get("n_estimators", 300),
            max_depth=self.params.get("max_depth"),
            min_samples_leaf=self.params.get("min_samples_leaf", 1),
            max_features=self.params.get("max_features", "sqrt"),
            class_weight="balanced_subsample" if pos_weight != 1.0 else None,
            n_jobs=self.params.get("n_jobs", 8),
            random_state=self.seed,
        )


@register("xgboost")
class XGBoostModel(_SklearnModel):
    """Gradient-boosted trees; early stopping on the validation split (logloss)."""

    def _build(self, pos_weight: float) -> Any:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=self.params.get("n_estimators", 400),
            max_depth=self.params.get("max_depth", 6),
            learning_rate=self.params.get("learning_rate", 0.1),
            subsample=self.params.get("subsample", 0.9),
            colsample_bytree=self.params.get("colsample_bytree", 0.9),
            min_child_weight=self.params.get("min_child_weight", 1.0),
            reg_lambda=self.params.get("reg_lambda", 1.0),
            scale_pos_weight=pos_weight,
            n_jobs=self.params.get("n_jobs", 8),
            random_state=self.seed,
            tree_method="hist",
            eval_metric="aucpr",
            early_stopping_rounds=self.params.get("early_stopping_rounds", 30),
        )

    def _fit(self, X: np.ndarray, y: np.ndarray, Xv: np.ndarray, yv: np.ndarray) -> None:
        self.estimator.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
