"""Build processed data from raw HDFS logs: parse -> sessions -> chronological split.

Usage:
    python scripts/make_dataset.py [overrides...]   e.g. data.parse.n_workers=4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from faultaware.data.parse import load_labels, parse_log_file
from faultaware.data.schemas import EVENTS_SCHEMA, validate_sessions
from faultaware.data.sessions import build_sessions
from faultaware.data.split import split_sessions
from faultaware.utils.config import load_config, project_path
from faultaware.utils.logging import get_logger

log = get_logger("make_dataset")


def levels_path(templates_path: Path) -> Path:
    """Location of the level-name list written next to the template table."""
    return templates_path.with_name("hdfs_levels.json")


def main(overrides: list[str]) -> None:
    """Run the data pipeline and write interim/processed artefacts."""
    cfg = load_config(overrides)
    dcfg = cfg.data
    events_path = project_path(dcfg.events_file)
    templates_path = project_path(dcfg.templates_file)
    if events_path.is_file() and templates_path.is_file():
        log.info("reusing parsed events %s", events_path)
        events = pd.read_parquet(events_path)
    else:
        events, templates = parse_log_file(
            project_path(dcfg.raw_log), dcfg.parse.n_workers, dcfg.parse.chunk_lines
        )
        EVENTS_SCHEMA.validate(events.head(100_000))
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events.to_parquet(events_path, index=False)
        templates.to_csv(templates_path, index=False)
        levels_path(templates_path).write_text(json.dumps(events.attrs["levels"]), encoding="utf-8")
        log.info("wrote %d events, %d templates", len(events), len(templates))

    labels = load_labels(project_path(dcfg.raw_labels))
    sessions = build_sessions(events, labels)
    log.info(
        "built %d labelled sessions (anomaly rate %.4f)", len(sessions), sessions["label"].mean()
    )
    split, bounds = split_sessions(
        sessions,
        (dcfg.split.train, dcfg.split.val, dcfg.split.test),
        int(dcfg.split.gap_minutes) * 60,
    )
    validate_sessions(split)
    out = project_path(dcfg.sessions_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    split.to_parquet(out, index=False)
    summary = {
        "n_sessions_total": len(sessions),
        "n_dropped_gap": len(sessions) - len(split),
        "boundaries": bounds.__dict__,
        "splits": {
            name: {
                "n": int((split["split"] == name).sum()),
                "anomaly_rate": float(split.loc[split["split"] == name, "label"].mean()),
                "start": int(split.loc[split["split"] == name, "first_ts"].min()),
                "end": int(split.loc[split["split"] == name, "first_ts"].max()),
            }
            for name in ("train", "val", "test")
        },
        "n_events_median": float(np.median(split["n_events"])),
    }
    summary_path = out.parent / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("split summary: %s", json.dumps(summary["splits"]))


if __name__ == "__main__":
    main(sys.argv[1:])
