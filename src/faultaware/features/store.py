"""Load processed sessions and build/cache per-split feature sets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from faultaware.features.build import WARN_LEVELS, FeatureSet, Vocabulary, build_features
from faultaware.utils.config import project_path
from faultaware.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class DataBundle:
    """Everything needed to featurise sessions consistently (train-fitted vocabulary)."""

    sessions: pd.DataFrame
    vocab: Vocabulary
    warn_level_ids: set[int]
    template_names: dict[int, str]
    max_len: int


def window_tag(observe_seconds: float | None) -> str:
    """Directory-safe name of an observation window (``full`` or ``w60``)."""
    return "full" if observe_seconds is None else f"w{int(observe_seconds)}"


def load_bundle(cfg: DictConfig) -> DataBundle:
    """Load sessions and fit the vocabulary on the training split only."""
    dcfg = cfg.data
    sessions = pd.read_parquet(project_path(dcfg.sessions_file))
    templates = pd.read_csv(project_path(dcfg.templates_file))
    names = dict(zip(templates["template_id"].astype(int), templates["template"], strict=True))
    levels = json.loads(
        project_path(dcfg.templates_file).with_name("hdfs_levels.json").read_text("utf-8")
    )
    warn_ids = {i for i, name in enumerate(levels) if name in WARN_LEVELS}
    train = sessions[sessions["split"] == "train"]
    vocab = Vocabulary.fit(train["templates"], names, int(dcfg.features.min_template_count))
    return DataBundle(sessions, vocab, warn_ids, names, int(dcfg.features.max_seq_len))


def _cache_path(cfg: DictConfig, tag: str, split: str) -> Path:
    return project_path(cfg.paths.processed_dir) / "features" / tag / f"{split}.npz"


def get_features(
    cfg: DictConfig,
    bundle: DataBundle,
    split: str,
    observe_seconds: float | None,
    use_cache: bool = True,
) -> FeatureSet:
    """Return the feature set of one split, building and caching it on first use."""
    path = _cache_path(cfg, window_tag(observe_seconds), split)
    if use_cache and path.is_file():
        with np.load(path) as z:
            return FeatureSet(**{k: z[k] for k in z.files})
    rows = bundle.sessions[bundle.sessions["split"] == split]
    log.info(
        "building %s features for %d %s sessions", window_tag(observe_seconds), len(rows), split
    )
    feats = build_features(
        rows, bundle.vocab, bundle.warn_level_ids, bundle.max_len, observe_seconds
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **feats.__dict__)
    return feats


def save_feature_spec(bundle: DataBundle, levels: list[str], path: Path) -> None:
    """Persist what serving needs to featurise new logs exactly like training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = {
        "vocab": bundle.vocab.to_dict(),
        "warn_level_ids": sorted(bundle.warn_level_ids),
        "levels": levels,
        "max_len": bundle.max_len,
        "template_names": {str(k): v for k, v in bundle.template_names.items()},
    }
    path.write_text(json.dumps(spec), encoding="utf-8")
