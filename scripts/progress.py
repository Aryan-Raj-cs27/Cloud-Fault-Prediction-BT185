"""Show training progress: finished runs per window/model and an ETA.

Usage:
    python scripts/progress.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = 14
WINDOWS = ["full", "w60", "w10"]
SEEDS = 5


def main() -> None:
    """Print a progress table from models/preds and runs/."""
    preds = ROOT / "models" / "preds"
    total = MODELS * len(WINDOWS) * SEEDS
    done = 0
    for window in WINDOWS:
        wdir = preds / window
        models = sorted(p.name for p in wdir.iterdir()) if wdir.is_dir() else []
        counts = {m: len(list((wdir / m).glob("seed*.npz"))) for m in models}
        done += sum(counts.values())
        print(f"[{window}] " + ", ".join(f"{m}:{c}/{SEEDS}" for m, c in counts.items()))
    runs = sorted((ROOT / "runs").glob("*/run.json"))
    print(f"\nFinished {done}/{total} training runs ({100 * done / total:.0f}%).")
    if runs:
        last = json.loads(runs[-1].read_text(encoding="utf-8"))
        print(
            f"Latest: {last['name']} seed={last['seed']} "
            f"val PR-AUC={last['metrics'].get('val_pr_auc', float('nan')):.4f}"
        )


if __name__ == "__main__":
    main()
