"""Chronological train/val/test split with a gap (CLAUDE.md §7.1).

Sessions are ordered by start time. Boundaries are placed at the time quantiles given by the
split fractions; sessions that start within ``gap`` seconds after a boundary are discarded so
that ``max(train.first_ts) + gap <= min(val.first_ts)`` (and likewise for val/test).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SplitBoundaries:
    """Start-time boundaries (epoch seconds) of the chronological split."""

    train_end: int
    val_start: int
    val_end: int
    test_start: int


def chronological_split(
    first_ts: np.ndarray, fractions: tuple[float, float, float], gap_seconds: int
) -> tuple[np.ndarray, SplitBoundaries]:
    """Assign each session to train/val/test by its start time.

    Args:
        first_ts: Session start times (epoch seconds), shape ``(n,)``. Need not be sorted.
        fractions: ``(train, val, test)`` fractions of sessions; must sum to 1.
        gap_seconds: Minimum separation between consecutive splits.

    Returns:
        ``(assignment, boundaries)``: ``assignment`` is an object array of shape ``(n,)`` with
        values ``"train" | "val" | "test" | "gap"``.

    Raises:
        ValueError: If fractions are invalid or the gap swallows a whole split.
    """
    if not np.isclose(sum(fractions), 1.0) or min(fractions) <= 0:
        raise ValueError(f"fractions must be positive and sum to 1, got {fractions}")
    if gap_seconds < 0:
        raise ValueError("gap_seconds must be >= 0")
    ts = np.sort(np.asarray(first_ts))
    n = len(ts)
    train_end = int(ts[int(n * fractions[0]) - 1])
    val_end = int(ts[int(n * (fractions[0] + fractions[1])) - 1])
    bounds = SplitBoundaries(
        train_end=train_end,
        val_start=train_end + gap_seconds + 1,
        val_end=val_end,
        test_start=val_end + gap_seconds + 1,
    )
    if bounds.val_start > bounds.val_end:
        raise ValueError("gap is larger than the validation period")
    first = np.asarray(first_ts)
    out = np.full(len(first), "gap", dtype=object)
    out[first <= bounds.train_end] = "train"
    out[(first >= bounds.val_start) & (first <= bounds.val_end)] = "val"
    out[first >= bounds.test_start] = "test"
    return out, bounds


def split_sessions(
    sessions: pd.DataFrame, fractions: tuple[float, float, float], gap_seconds: int
) -> tuple[pd.DataFrame, SplitBoundaries]:
    """Return ``sessions`` with an added ``split`` column (gap rows removed).

    Args:
        sessions: Session table with a ``first_ts`` column.
        fractions: ``(train, val, test)`` fractions.
        gap_seconds: Gap between splits in seconds.

    Returns:
        ``(sessions_with_split, boundaries)``.
    """
    assignment, bounds = chronological_split(
        sessions["first_ts"].to_numpy(), fractions, gap_seconds
    )
    out = sessions.assign(split=assignment)
    return out[out["split"] != "gap"].reset_index(drop=True), bounds
