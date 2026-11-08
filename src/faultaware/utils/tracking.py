"""Lightweight, file-based experiment tracking (see docs/DECISIONS.md D-007).

Each run writes ``runs/<run_id>/run.json`` containing the git commit, full resolved config,
seed, dataset version, metrics and artefact paths — everything CLAUDE.md §6.3 requires.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from faultaware.utils.config import PROJECT_ROOT


def git_commit() -> str:
    """Return the current git commit hash (with ``-dirty`` suffix if uncommitted changes)."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@dataclass
class Run:
    """A single tracked experiment run."""

    name: str
    config: dict[str, Any]
    seed: int
    runs_dir: Path
    dataset_version: str = "hdfs_v1"
    metrics: dict[str, Any] = field(default_factory=dict)
    artefacts: dict[str, str] = field(default_factory=dict)
    run_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{self.name}_s{self.seed}"

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Merge metrics into the run record."""
        self.metrics.update(metrics)

    def log_artefact(self, key: str, path: Path) -> None:
        """Record the path of a produced artefact."""
        self.artefacts[key] = str(path)

    def save(self) -> Path:
        """Write ``run.json`` and return its path."""
        out_dir = self.runs_dir / self.run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "run_id": self.run_id,
            "name": self.name,
            "seed": self.seed,
            "git_commit": git_commit(),
            "dataset_version": self.dataset_version,
            "config": self.config,
            "metrics": self.metrics,
            "artefacts": self.artefacts,
        }
        path = out_dir / "run.json"
        path.write_text(json.dumps(record, indent=2, default=float), encoding="utf-8")
        return path
