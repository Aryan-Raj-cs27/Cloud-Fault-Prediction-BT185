"""Sessions, chronological split, schemas and feature construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from faultaware.data.schemas import EVENTS_SCHEMA, validate_sessions
from faultaware.data.sessions import truncate_session
from faultaware.data.split import chronological_split
from faultaware.features.build import PAD, STAT_NAMES, UNK, Vocabulary, tabular_feature_names


def test_events_schema_valid(tiny_data: dict) -> None:
    EVENTS_SCHEMA.validate(tiny_data["events"])


def test_sessions_valid_and_sorted(tiny_data: dict) -> None:
    validate_sessions(tiny_data["split"])
    s = tiny_data["sessions"]
    assert s.block_id.is_unique
    assert s.first_ts.is_monotonic_increasing
    assert (s.n_events == s.templates.map(len)).all()


def test_sessions_labels_match_source(tiny_data: dict) -> None:
    labels = pd.read_csv(tiny_data["label_path"])
    assert tiny_data["sessions"].label.sum() == (labels.Label == "Anomaly").sum()


def test_validate_sessions_rejects_bad_lengths(tiny_data: dict) -> None:
    bad = tiny_data["split"].copy()
    bad.at[0, "n_events"] = 999
    with pytest.raises(ValueError):
        validate_sessions(bad)


def test_truncate_session() -> None:
    offs = np.array([0, 0, 5, 10, 3600])
    tmpl = np.arange(5)
    assert truncate_session(tmpl, offs, None) == 5
    assert truncate_session(tmpl, offs, 0) == 2
    assert truncate_session(tmpl, offs, 10) == 4


@pytest.mark.parametrize("fractions", [(0.7, 0.15, 0.1), (0.0, 0.5, 0.5)])
def test_split_rejects_bad_fractions(fractions: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError):
        chronological_split(np.arange(100), fractions, 0)


def test_split_rejects_gap_larger_than_val() -> None:
    with pytest.raises(ValueError):
        chronological_split(np.arange(100), (0.7, 0.15, 0.15), gap_seconds=1000)


def test_split_assigns_by_time_and_respects_gap() -> None:
    ts = np.arange(0, 1000, 1)
    out, b = chronological_split(ts, (0.7, 0.15, 0.15), gap_seconds=10)
    assert set(out) == {"train", "val", "test", "gap"}
    assert ts[out == "train"].max() + 10 < ts[out == "val"].min()
    assert ts[out == "val"].max() + 10 < ts[out == "test"].min()
    assert b.val_start == b.train_end + 11


def test_vocabulary_train_only_and_unk(tiny_data: dict) -> None:
    vocab: Vocabulary = tiny_data["vocab"]
    assert vocab.names[PAD] == "<PAD>" and vocab.names[UNK] == "<UNK>"
    unseen = max(vocab.token_of) + 100
    assert vocab.encode(np.array([unseen]))[0] == UNK
    restored = Vocabulary.from_dict(vocab.to_dict())
    assert restored.token_of == vocab.token_of and restored.names == vocab.names


def test_feature_shapes(tiny_data: dict) -> None:
    vocab = tiny_data["vocab"]
    f = tiny_data["features"]["train"]
    n = len(f.y)
    assert f.counts.shape == (n, vocab.size)
    assert f.stats.shape == (n, len(STAT_NAMES))
    assert f.sequences.shape == (n, 16)
    assert f.tabular.shape[1] == len(tabular_feature_names(vocab))
    assert (f.counts[:, PAD] == 0).all()
    assert (f.lengths >= 1).all()
    np.testing.assert_allclose(f.counts.sum(1), f.stats[:, 0])


def test_early_window_uses_fewer_events(tiny_data: dict) -> None:
    full, early = tiny_data["features"]["train"], tiny_data["early"]["train"]
    assert (early.stats[:, 0] <= full.stats[:, 0]).all()
    assert (early.stats[:, 0] < full.stats[:, 0]).any()
    assert (early.lead_seconds >= full.lead_seconds).all()
