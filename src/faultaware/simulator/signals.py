"""Risk signals observed by the scheduler (CLAUDE.md §9: no peeking at future failures).

A node's *true* state is either healthy or degraded (inside the precursor window of an
upcoming predictable failure). The scheduler never sees that state — only a risk score drawn
from a predictor model:

* ``EmpiricalSignal`` reproduces the real HDFS predictor: while degraded, the score is drawn
  from the fused risk scores the model gave to *anomalous* test sessions; otherwise from those
  given to *normal* sessions. Its precision/recall therefore match the measured model.
* ``OracleSignal`` (perfect predictor) and ``RandomSignal`` (uninformative) are the sanity
  bounds required by CLAUDE.md §9.
* ``noise`` replaces a score by ``U(0, 1)`` with that probability (noise ablation, §8.3e).
* ``persistent=True`` (default, conservative) draws ONE positive score per degradation
  episode and keeps it until the episode ends, so a precursor the model misses stays missed;
  healthy nodes draw a fresh negative score every tick (independent false alarms).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class RiskSignal(Protocol):
    """Callable producing a risk score in ``[0, 1]`` for a node given its hidden state."""

    def __call__(self, node_id: int, degraded: bool, rng: np.random.Generator) -> float: ...


class OracleSignal:
    """Perfect predictor: 1 while degraded, 0 otherwise."""

    def __call__(self, node_id: int, degraded: bool, rng: np.random.Generator) -> float:
        return 1.0 if degraded else 0.0


class RandomSignal:
    """Uninformative predictor: ``U(0, 1)`` regardless of state."""

    def __call__(self, node_id: int, degraded: bool, rng: np.random.Generator) -> float:
        return float(rng.random())


class EmpiricalSignal:
    """Samples risk scores from the real model's class-conditional score distributions.

    Args:
        pos_scores: Risk scores of truly anomalous sessions, shape ``(n_pos,)``.
        neg_scores: Risk scores of normal sessions, shape ``(n_neg,)``.
        noise: Probability of replacing a draw by ``U(0, 1)``.
        persistent: Keep one positive draw per degradation episode.
    """

    def __init__(
        self,
        pos_scores: np.ndarray,
        neg_scores: np.ndarray,
        noise: float = 0.0,
        persistent: bool = True,
    ):
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            raise ValueError("need scores for both classes")
        if not 0.0 <= noise <= 1.0:
            raise ValueError("noise must be in [0, 1]")
        self.pos = np.asarray(pos_scores, dtype=np.float64)
        self.neg = np.asarray(neg_scores, dtype=np.float64)
        self.noise = noise
        self.persistent = persistent
        self._episode: dict[int, float] = {}

    def __call__(self, node_id: int, degraded: bool, rng: np.random.Generator) -> float:
        if not degraded:
            self._episode.pop(node_id, None)
        if self.noise > 0 and rng.random() < self.noise:
            return float(rng.random())
        if degraded and self.persistent:
            if node_id not in self._episode:
                self._episode[node_id] = float(self.pos[rng.integers(len(self.pos))])
            return self._episode[node_id]
        pool = self.pos if degraded else self.neg
        return float(pool[rng.integers(len(pool))])

    @classmethod
    def from_file(
        cls,
        path: Path,
        split: str,
        variant: str = "fused",
        noise: float = 0.0,
        persistent: bool = True,
    ) -> EmpiricalSignal:
        """Load ``<split>_<variant>_pos`` / ``_neg`` arrays from an ``.npz`` file."""
        with np.load(path) as z:
            return cls(z[f"{split}_{variant}_pos"], z[f"{split}_{variant}_neg"], noise, persistent)


def make_signal(
    source: str,
    noise: float = 0.0,
    scores_file: Path | None = None,
    split: str = "test",
    variant: str = "fused",
    persistent: bool = True,
) -> RiskSignal:
    """Factory for the configured signal source.

    Args:
        source: ``"empirical" | "oracle" | "random"``.
        noise: Noise probability (empirical only).
        scores_file: ``.npz`` with score distributions (empirical only).
        split: ``"val"`` for tuning runs, ``"test"`` for final runs.
        variant: Which score set (``fused``, ``supervised_raw``, ``supervised_cal``).
    """
    if source == "oracle":
        return OracleSignal()
    if source == "random":
        return RandomSignal()
    if source == "empirical":
        if scores_file is None:
            raise ValueError("empirical signal needs scores_file")
        return EmpiricalSignal.from_file(scores_file, split, variant, noise, persistent)
    raise ValueError(f"unknown signal source {source!r}")
