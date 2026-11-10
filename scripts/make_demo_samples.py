"""Extract raw log lines of held-out TEST blocks for the dashboard's live replay.

Picks ``--n-normal`` normal and ``--n-anomaly`` anomalous test blocks (seeded) and scans the raw
log once to collect their lines. Output: ``app/demo_samples.json``.

Usage:
    python scripts/make_demo_samples.py [--n-normal 300] [--n-anomaly 100]
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np
import pandas as pd

from faultaware.utils.config import PROJECT_ROOT, load_config, project_path
from faultaware.utils.logging import get_logger

log = get_logger("demo_samples")
BLOCK_RE = re.compile(r"blk_-?\d+")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-normal", type=int, default=300)
    parser.add_argument("--n-anomaly", type=int, default=100)
    args = parser.parse_args()
    cfg = load_config()
    sessions = pd.read_parquet(
        project_path(cfg.data.sessions_file), columns=["block_id", "label", "split"]
    )
    test = sessions[sessions.split == "test"]
    rng = np.random.default_rng(cfg.seed)
    chosen = np.r_[
        rng.choice(test[test.label == 0].block_id, args.n_normal, replace=False),
        rng.choice(test[test.label == 1].block_id, args.n_anomaly, replace=False),
    ]
    wanted = {f"blk_{b}": int(b) for b in chosen}
    label_of = dict(zip(test.block_id, test.label, strict=True))
    lines: dict[str, list[str]] = {k: [] for k in wanted}
    with project_path(cfg.data.raw_log).open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            for blk in set(BLOCK_RE.findall(line)):
                if blk in lines:
                    lines[blk].append(line.rstrip("\n"))
    samples = [
        {"block_id": k, "label": "Anomaly" if label_of[v] else "Normal", "log_lines": lines[k]}
        for k, v in wanted.items()
        if lines[k]
    ]
    rng.shuffle(samples)  # type: ignore[arg-type]
    out = PROJECT_ROOT / "app" / "demo_samples.json"
    out.write_text(json.dumps(samples), encoding="utf-8")
    log.info("wrote %d samples to %s", len(samples), out)


if __name__ == "__main__":
    main()
