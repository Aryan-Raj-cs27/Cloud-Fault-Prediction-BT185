"""Group parsed events into per-block sessions and attach labels.

A *session* is the chronologically ordered sequence of events that mention one HDFS block.
Loghub provides one expert label per block (Normal/Anomaly). Sessions are the samples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SESSION_COLUMNS = [
    "block_id",
    "label",
    "first_ts",
    "last_ts",
    "first_line",
    "n_events",
    "templates",
    "offsets",
    "ips",
    "levels",
]


def build_sessions(events: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Build one row per labelled block.

    Args:
        events: Output of :func:`faultaware.data.parse.parse_lines` (one row per line × block).
        labels: ``block_id`` / ``label`` table from :func:`faultaware.data.parse.load_labels`.

    Returns:
        DataFrame with columns :data:`SESSION_COLUMNS`. ``templates``, ``ips`` and ``levels`` are
        int arrays in log order; ``offsets`` are seconds since the block's first event.
        Rows are sorted by ``(first_ts, first_line)``. Blocks without a label are dropped.
    """
    ev = events.sort_values(["block_id", "line_no"], kind="stable")
    block = ev["block_id"].to_numpy()
    if len(block) == 0:
        return pd.DataFrame(columns=SESSION_COLUMNS)
    starts = np.flatnonzero(np.r_[True, block[1:] != block[:-1]])
    bounds = np.r_[starts, len(block)]
    ts = ev["timestamp"].to_numpy()
    tmpl = ev["template_id"].to_numpy().astype(np.int32)
    ips = ev["ip_id"].to_numpy().astype(np.int32)
    lvl = ev["level_id"].to_numpy().astype(np.int8)
    line = ev["line_no"].to_numpy()

    def split(arr: np.ndarray) -> list[np.ndarray]:
        return np.split(arr, bounds[1:-1])

    first_ts = ts[starts]
    sessions = pd.DataFrame(
        {
            "block_id": block[starts],
            "first_ts": first_ts,
            "last_ts": np.maximum.reduceat(ts, starts),
            "first_line": line[starts],
            "n_events": np.diff(bounds),
            "templates": split(tmpl),
            # Log lines are written by many threads, so second-resolution timestamps can be
            # slightly out of order; a running max keeps offsets non-decreasing (and >= 0).
            "offsets": [
                np.maximum.accumulate(np.maximum(o, 0)).astype(np.int32)
                for o in split(ts - np.repeat(first_ts, np.diff(bounds)))
            ],
            "ips": split(ips),
            "levels": split(lvl),
        }
    )
    sessions = sessions.merge(labels, on="block_id", how="inner", validate="one_to_one")
    sessions = sessions.sort_values(["first_ts", "first_line"], kind="stable").reset_index(
        drop=True
    )
    return sessions[SESSION_COLUMNS]


def truncate_session(
    templates: np.ndarray, offsets: np.ndarray, observe_seconds: float | None
) -> int:
    """Return how many leading events fall inside the observation window.

    Args:
        templates: Template ids of one session, shape ``(n_events,)``.
        offsets: Seconds since the session's first event, shape ``(n_events,)``, non-decreasing.
        observe_seconds: Window length; ``None`` = whole session.

    Returns:
        Number of events ``k`` such that events ``[0, k)`` have ``offset <= observe_seconds``.
        Always ``>= 1`` for a non-empty session (the first event has offset 0).
    """
    if observe_seconds is None:
        return len(templates)
    return int(np.searchsorted(offsets, observe_seconds, side="right"))
