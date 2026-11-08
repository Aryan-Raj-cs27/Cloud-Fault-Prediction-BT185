"""Leakage-safe feature construction from session tables.

* The template vocabulary is fitted on the TRAIN split only; templates unseen in train map to
  ``UNK``. Index 0 is reserved for padding, 1 for ``UNK``.
* Every feature for a session uses only the events inside its observation window
  (``observe_seconds`` after the block's first event) — never later events.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from faultaware.data.sessions import truncate_session

PAD, UNK = 0, 1
WARN_LEVELS = frozenset({"WARN", "ERROR", "FATAL"})


@dataclass
class Vocabulary:
    """Mapping from global template ids to model token ids (fitted on train)."""

    token_of: dict[int, int] = field(default_factory=dict)
    names: list[str] = field(default_factory=lambda: ["<PAD>", "<UNK>"])

    @property
    def size(self) -> int:
        """Number of tokens including PAD and UNK."""
        return len(self.names)

    @classmethod
    def fit(
        cls, train_templates: pd.Series, template_names: dict[int, str], min_count: int = 1
    ) -> Vocabulary:
        """Fit the vocabulary on training sessions.

        Args:
            train_templates: Series of template-id arrays (train split only).
            template_names: Global template id -> template text.
            min_count: Minimum number of train occurrences to get an own token.

        Returns:
            A fitted vocabulary.
        """
        counts = np.bincount(np.concatenate(train_templates.to_list()))
        vocab = cls()
        for tid in np.flatnonzero(counts >= min_count):
            vocab.token_of[int(tid)] = len(vocab.names)
            vocab.names.append(template_names.get(int(tid), f"T{tid}"))
        return vocab

    def encode(self, template_ids: np.ndarray) -> np.ndarray:
        """Map global template ids to token ids (``UNK`` for unseen)."""
        get = self.token_of.get
        return np.fromiter(
            (get(int(t), UNK) for t in template_ids), dtype=np.int64, count=len(template_ids)
        )

    def to_dict(self) -> dict[str, object]:
        """Serialise to JSON-compatible dict."""
        return {"token_of": {str(k): v for k, v in self.token_of.items()}, "names": self.names}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Vocabulary:
        """Inverse of :meth:`to_dict`."""
        token_of = {int(k): int(v) for k, v in data["token_of"].items()}  # type: ignore[attr-defined]
        return cls(token_of=token_of, names=list(data["names"]))  # type: ignore[call-overload]


@dataclass
class FeatureSet:
    """Model inputs for one split.

    Attributes:
        counts: Template count matrix, shape ``(n, vocab_size)`` (PAD column always 0).
        stats: Session statistics, shape ``(n, len(STAT_NAMES))``.
        sequences: Padded token sequences, shape ``(n, max_len)``, int64.
        lengths: True (observed, truncated) sequence lengths, shape ``(n,)``.
        y: Labels, shape ``(n,)``.
        lead_seconds: ``last_ts - (first_ts + observe)`` clipped at 0, shape ``(n,)``.
    """

    counts: np.ndarray
    stats: np.ndarray
    sequences: np.ndarray
    lengths: np.ndarray
    y: np.ndarray
    lead_seconds: np.ndarray

    @property
    def tabular(self) -> np.ndarray:
        """Tabular design matrix ``[counts | stats]``, shape ``(n, vocab_size + n_stats)``."""
        return np.hstack([self.counts, self.stats])


STAT_NAMES = ("n_events", "duration_s", "n_unique_templates", "n_unique_ips", "n_warn", "frac_unk")


def tabular_feature_names(vocab: Vocabulary) -> list[str]:
    """Column names of :attr:`FeatureSet.tabular`."""
    return [f"count[{n}]" for n in vocab.names] + list(STAT_NAMES)


def build_features(
    sessions: pd.DataFrame,
    vocab: Vocabulary,
    warn_level_ids: set[int],
    max_len: int,
    observe_seconds: float | None = None,
) -> FeatureSet:
    """Compute all model inputs for a set of sessions.

    Args:
        sessions: Session rows (one split).
        vocab: Train-fitted vocabulary.
        warn_level_ids: Level ids counted as warnings (WARN/ERROR/FATAL).
        max_len: Sequence length; sequences keep their FIRST ``max_len`` events.
        observe_seconds: Observation window; ``None`` = full session.

    Returns:
        The feature set (see :class:`FeatureSet` for shapes).
    """
    n = len(sessions)
    counts = np.zeros((n, vocab.size), dtype=np.float32)
    stats = np.zeros((n, len(STAT_NAMES)), dtype=np.float32)
    seqs = np.zeros((n, max_len), dtype=np.int64)
    lengths = np.zeros(n, dtype=np.int64)
    lead = np.zeros(n, dtype=np.float32)
    warn_ids = np.fromiter(warn_level_ids, dtype=np.int64) if warn_level_ids else np.array([-1])
    for i, (tmpl, offs, ips, lvls, first, last) in enumerate(
        zip(
            sessions["templates"],
            sessions["offsets"],
            sessions["ips"],
            sessions["levels"],
            sessions["first_ts"],
            sessions["last_ts"],
            strict=True,
        )
    ):
        k = truncate_session(tmpl, offs, observe_seconds)
        tokens = vocab.encode(tmpl[:k])
        np.add.at(counts[i], tokens, 1.0)
        seq = tokens[:max_len]
        seqs[i, : len(seq)] = seq
        lengths[i] = max(len(seq), 1)
        stats[i] = (
            k,
            float(offs[k - 1]) if k else 0.0,
            len(np.unique(tokens)),
            len(np.unique(ips[:k])),
            float(np.isin(lvls[:k], warn_ids).sum()),
            float((tokens == UNK).mean()) if k else 0.0,
        )
        horizon_end = first + (observe_seconds if observe_seconds is not None else last - first)
        lead[i] = max(0.0, float(last - horizon_end))
    return FeatureSet(
        counts, stats, seqs, lengths, sessions["label"].to_numpy().astype(np.int64), lead
    )
