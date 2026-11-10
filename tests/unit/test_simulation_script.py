"""The score-source table compares FARS with the best baseline on the same paired seeds."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from run_simulation import score_source_table


def _runs(name: str, interruptions: list[float], lost: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scheduler": name,
            "seed": range(len(lost)),
            "task_interruptions": interruptions,
            "wasted_compute_min": lost,
            "lost_node_min": lost,
            "migrations": 0.0,
        }
    )


def test_score_source_table_relative_to_best_baseline() -> None:
    main = pd.concat(
        [
            _runs("least_loaded", [100, 110, 90, 100, 105, 95], [1000] * 6),
            _runs("random", [120] * 6, [1200] * 6),
        ],
        ignore_index=True,
    )
    fars = {
        "test": _runs("fars", [50] * 6, [800] * 6),
        "val": _runs("fars", [100, 112, 91, 99, 104, 94], [1100] * 6),
    }
    table = score_source_table(fars, main, ["least_loaded", "random"]).set_index("score_source")
    strict, pessimistic = table.index
    # best baseline = least_loaded (mean interruptions 100, lost 1000)
    assert table.loc[strict, "task_interruptions_vs_best_baseline_%"] == -50.0
    assert table.loc[strict, "lost_node_min_vs_best_baseline_%"] == -20.0
    assert table.loc[pessimistic, "lost_node_min_vs_best_baseline_%"] == 10.0
    assert table.loc[strict, "task_interruptions_p_vs_least_loaded"] < 0.05
    assert table.loc[pessimistic, "task_interruptions_p_vs_least_loaded"] > 0.05
