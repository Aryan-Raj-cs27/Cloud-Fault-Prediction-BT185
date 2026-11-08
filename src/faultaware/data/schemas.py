"""Pandera schemas validating the data layer outputs (CLAUDE.md §10.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa

EVENTS_SCHEMA = pa.DataFrameSchema(
    {
        "line_no": pa.Column(np.int64, pa.Check.ge(0)),
        "timestamp": pa.Column(np.int64, pa.Check.gt(0)),
        "block_id": pa.Column(np.int64),
        "template_id": pa.Column(np.int32, pa.Check.ge(0)),
        "level_id": pa.Column(np.int8, pa.Check.ge(0)),
        "ip_id": pa.Column(np.int32, pa.Check.ge(0)),
    },
    strict=True,
)

SESSIONS_SCHEMA = pa.DataFrameSchema(
    {
        "block_id": pa.Column(np.int64, unique=True),
        "label": pa.Column(np.int8, pa.Check.isin([0, 1])),
        "first_ts": pa.Column(np.int64),
        "last_ts": pa.Column(np.int64),
        "first_line": pa.Column(np.int64),
        "n_events": pa.Column(np.int64, pa.Check.ge(1)),
        "templates": pa.Column(object),
        "offsets": pa.Column(object),
        "ips": pa.Column(object),
        "levels": pa.Column(object),
        "split": pa.Column(str, pa.Check.isin(["train", "val", "test"]), required=False),
    },
    checks=[
        pa.Check(lambda df: (df["last_ts"] >= df["first_ts"]).all(), name="last_after_first"),
        pa.Check(lambda df: df["first_ts"].is_monotonic_increasing, name="sorted_by_start_time"),
    ],
    strict=True,
)


def validate_sessions(sessions: pd.DataFrame) -> pd.DataFrame:
    """Validate a session table, including per-row array consistency.

    Args:
        sessions: Session table.

    Returns:
        The validated table.

    Raises:
        pandera.errors.SchemaError: On schema violations.
        ValueError: If array columns disagree with ``n_events`` or offsets decrease.
    """
    SESSIONS_SCHEMA.validate(sessions)
    lengths = sessions["templates"].map(len).to_numpy()
    if not (lengths == sessions["n_events"].to_numpy()).all():
        raise ValueError("templates length != n_events")
    if not all(bool(np.all(np.diff(o) >= 0)) for o in sessions["offsets"]):
        raise ValueError("offsets must be non-decreasing within a session")
    return sessions
