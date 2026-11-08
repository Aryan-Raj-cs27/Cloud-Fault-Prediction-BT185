"""Leakage tests (CLAUDE.md §7.1, §10.1) — these must never be deleted or weakened."""

from __future__ import annotations

import numpy as np
import pandas as pd

from faultaware.features.build import Vocabulary, build_features
from faultaware.models.base import create_model


def test_train_ends_before_val_starts_with_gap(tiny_data: dict) -> None:
    s, b = tiny_data["split"], tiny_data["bounds"]
    gap = b.val_start - b.train_end - 1
    tr, va, te = (s[s.split == k] for k in ("train", "val", "test"))
    assert tr.first_ts.max() + gap < va.first_ts.min()
    assert va.first_ts.max() + gap < te.first_ts.min()


def test_splits_are_disjoint(tiny_data: dict) -> None:
    s = tiny_data["split"]
    groups = [set(s[s.split == k].block_id) for k in ("train", "val", "test")]
    assert not (groups[0] & groups[1]) and not (groups[0] & groups[2])
    assert not (groups[1] & groups[2])


def test_vocabulary_ignores_templates_only_in_test(tiny_data: dict) -> None:
    s = tiny_data["split"].copy()
    test_idx = s.index[s.split == "test"][0]
    s.at[test_idx, "templates"] = np.array([987654], dtype=np.int32)
    s.at[test_idx, "offsets"] = np.array([0], dtype=np.int32)
    s.at[test_idx, "ips"] = np.array([0], dtype=np.int32)
    s.at[test_idx, "levels"] = np.array([0], dtype=np.int8)
    vocab = Vocabulary.fit(s[s.split == "train"]["templates"], {})
    assert 987654 not in vocab.token_of


def test_features_do_not_use_future_events(tiny_data: dict) -> None:
    """Changing events AFTER the observation window must not change the features."""
    s = tiny_data["split"]
    rows = s[s.split == "train"].head(20).copy()
    vocab, warn = tiny_data["vocab"], tiny_data["warn"]
    before = build_features(rows, vocab, warn, 16, observe_seconds=10)
    mutated = rows.copy()
    mutated["templates"] = [
        np.where(o > 10, 1, t) for t, o in zip(rows.templates, rows.offsets, strict=True)
    ]
    after = build_features(mutated, vocab, warn, 16, observe_seconds=10)
    np.testing.assert_array_equal(before.counts, after.counts)
    np.testing.assert_array_equal(before.sequences, after.sequences)


def test_scaler_fitted_on_train_only(tiny_data: dict) -> None:
    f = tiny_data["features"]
    model = create_model("logreg", {"C": 1.0, "max_iter": 500}, seed=0)
    model.fit(f["train"], f["train"].y, f["val"], f["val"].y)
    scaler = model.estimator.named_steps["scale"]  # type: ignore[attr-defined]
    np.testing.assert_allclose(scaler.mean_, np.log1p(f["train"].tabular).mean(0), rtol=1e-5)


def test_smote_only_resamples_training_data(tiny_data: dict) -> None:
    f = tiny_data["features"]
    model = create_model(
        "random_forest", {"n_estimators": 10, "imbalance": "smote", "n_jobs": 1}, seed=0
    )
    model.fit(f["train"], f["train"].y, f["val"], f["val"].y)
    assert len(model.predict_proba(f["val"])) == len(f["val"].y)
    assert len(model.predict_proba(f["test"])) == len(f["test"].y)


def test_real_split_summary_if_present() -> None:
    """When processed data exists, the real split must satisfy the gap rule too."""
    from faultaware.utils.config import load_config, project_path

    cfg = load_config()
    path = project_path(cfg.data.sessions_file)
    if not path.is_file():
        return
    s = pd.read_parquet(path, columns=["first_ts", "split"])
    gap = int(cfg.data.split.gap_minutes) * 60
    assert s[s.split == "train"].first_ts.max() + gap < s[s.split == "val"].first_ts.min()
    assert s[s.split == "val"].first_ts.max() + gap < s[s.split == "test"].first_ts.min()
