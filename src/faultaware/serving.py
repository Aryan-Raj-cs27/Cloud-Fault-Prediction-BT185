"""Online inference: raw HDFS log lines of one block -> calibrated fault risk.

Uses exactly the training-time featurisation (train-fitted vocabulary from
``models/serving/feature_spec.json``), the seed-0 artefacts of the fused models and the
validation-fitted fusion (``models/fusion/<window>/``).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from faultaware.data.parse import parse_line
from faultaware.features.build import FeatureSet, Vocabulary, build_features
from faultaware.features.store import window_tag
from faultaware.fusion.risk import RiskFusion
from faultaware.models.base import FaultPredictor, model_class


@dataclass
class FeatureSpec:
    """Serving-time featurisation parameters (fitted on train)."""

    vocab: Vocabulary
    warn_level_ids: set[int]
    levels: list[str]
    max_len: int
    template_names: dict[int, str]

    @classmethod
    def load(cls, path: Path) -> FeatureSpec:
        """Read ``feature_spec.json``."""
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            Vocabulary.from_dict(raw["vocab"]),
            set(raw["warn_level_ids"]),
            list(raw["levels"]),
            int(raw["max_len"]),
            {int(k): v for k, v in raw["template_names"].items()},
        )

    def template_id(self, template: str) -> int:
        """Global template id of a template string (``-1`` if never seen during parsing)."""
        if not hasattr(self, "_by_text"):
            self._by_text = {v: k for k, v in self.template_names.items()}
        return self._by_text.get(template, -1)


def lines_to_session(lines: list[str], spec: FeatureSpec) -> pd.DataFrame:
    """Turn raw log lines (one block) into a one-row session table.

    Raises:
        ValueError: If no line matches the HDFS format.
    """
    parsed = [p for p in (parse_line(line) for line in lines) if p is not None]
    if not parsed:
        raise ValueError("no valid HDFS log lines")
    order = sorted(range(len(parsed)), key=lambda i: (parsed[i].timestamp, i))
    parsed = [parsed[i] for i in order]
    ts = np.array([p.timestamp for p in parsed], dtype=np.int64)
    offsets = np.maximum.accumulate(np.maximum(ts - ts[0], 0)).astype(np.int32)
    ip_codes: dict[str, int] = {}
    level_index = {name: i for i, name in enumerate(spec.levels)}
    return pd.DataFrame(
        {
            "block_id": [parsed[0].blocks[0] if parsed[0].blocks else 0],
            "label": [0],
            "first_ts": [int(ts[0])],
            "last_ts": [int(ts.max())],
            "first_line": [0],
            "n_events": [len(parsed)],
            "templates": [np.array([spec.template_id(p.template) for p in parsed], np.int32)],
            "offsets": [offsets],
            "ips": [np.array([ip_codes.setdefault(p.ip, len(ip_codes)) for p in parsed], np.int32)],
            "levels": [np.array([level_index.get(p.level, -1) for p in parsed], np.int8)],
        }
    )


@dataclass
class WindowModels:
    """The fused predictor of one observation window."""

    observe_seconds: float | None
    supervised: FaultPredictor
    anomaly: FaultPredictor
    fusion: RiskFusion
    meta: dict[str, Any]


class RiskService:
    """Loads artefacts once and answers predictions.

    Args:
        models_dir: Project ``models/`` directory.
        tau: ``(tau_low, tau_high)`` FARS thresholds used to name risk levels.
    """

    def __init__(self, models_dir: Path, tau: tuple[float, float] = (0.3, 0.6)) -> None:
        self.models_dir = models_dir
        self.spec = FeatureSpec.load(models_dir / "serving" / "feature_spec.json")
        self.tau = tau
        self.windows: dict[str, WindowModels] = {}
        for fdir in sorted((models_dir / "fusion").glob("*/fusion.json")):
            meta = json.loads(fdir.read_text(encoding="utf-8"))
            tag = fdir.parent.name
            art = models_dir / "artefacts" / tag
            self.windows[tag] = WindowModels(
                meta.get("observe_seconds"),
                _load(meta["supervised_model"], art / meta["supervised_model"]),
                _load(meta["anomaly_model"], art / meta["anomaly_model"]),
                RiskFusion.load(fdir.parent / "fusion.joblib"),
                meta,
            )
        if not self.windows:
            raise FileNotFoundError(f"no fused models under {models_dir / 'fusion'}")

    def risk_level(self, risk: float) -> str:
        """FARS action band of a risk score."""
        if risk >= self.tau[1]:
            return "high"
        return "elevated" if risk >= self.tau[0] else "low"

    def featurise(self, lines: list[str], observe_seconds: float | None) -> FeatureSet:
        """Featurise one block exactly as in training."""
        session = lines_to_session(lines, self.spec)
        return build_features(
            session, self.spec.vocab, self.spec.warn_level_ids, self.spec.max_len, observe_seconds
        )

    def predict(self, lines: list[str], window: str | None = None) -> dict[str, Any]:
        """Predict fault risk for one block's log lines.

        Args:
            lines: Raw HDFS log lines mentioning the block.
            window: ``"w60"`` (early warning, default when available), ``"w10"`` or ``"full"``.

        Returns:
            Risk, label, probabilities, risk band, explanations and latency.
        """
        tag = window or ("w60" if "w60" in self.windows else next(iter(self.windows)))
        if tag not in self.windows:
            raise KeyError(f"unknown window {tag!r}; available {sorted(self.windows)}")
        wm = self.windows[tag]
        started = time.perf_counter()
        feats = self.featurise(lines, wm.observe_seconds)
        p_raw = float(wm.supervised.predict_proba(feats)[0])
        anomaly = float(wm.anomaly.predict_proba(feats)[0])
        risk = float(wm.fusion(np.array([p_raw]), np.array([anomaly]))[0])
        is_fault = risk >= wm.fusion.threshold
        tokens = feats.sequences[0][: feats.lengths[0]]
        return {
            "prediction": int(is_fault),
            "label": "Anomaly" if is_fault else "Normal",
            "risk": risk,
            "risk_level": self.risk_level(risk),
            "confidence": risk if is_fault else 1.0 - risk,
            "threshold": wm.fusion.threshold,
            "probability_supervised": p_raw,
            "probability_calibrated": float(wm.fusion.calibrator(np.array([p_raw]))[0]),
            "anomaly_score": anomaly,
            "alpha": wm.fusion.alpha,
            "window": tag,
            "observe_seconds": wm.observe_seconds,
            "models": {
                "supervised": wm.meta["supervised_model"],
                "anomaly": wm.meta["anomaly_model"],
            },
            "events_observed": int(feats.stats[0, 0]),
            "unknown_templates": int(np.sum(tokens == 1)),
            "warnings_observed": int(feats.stats[0, 4]),
            "latency_ms": round(1000 * (time.perf_counter() - started), 2),
        }

    def info(self) -> dict[str, Any]:
        """Model/window metadata."""
        return {
            "windows": {t: w.meta for t, w in self.windows.items()},
            "vocab_size": self.spec.vocab.size,
            "tau": list(self.tau),
        }


def _load(name: str, path: Path) -> FaultPredictor:
    base = name
    meta_file = path / "meta.json"
    if meta_file.is_file():
        base = json.loads(meta_file.read_text(encoding="utf-8")).get("params", {}).get("base", name)
    cls: Any = model_class(base)
    try:
        return cls.load(path, device="cpu")
    except TypeError:
        return cls.load(path)


def tag_of(observe_seconds: float | None) -> str:
    """Re-export of :func:`faultaware.features.store.window_tag`."""
    return window_tag(observe_seconds)
