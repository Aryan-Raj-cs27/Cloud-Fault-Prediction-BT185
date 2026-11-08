"""Running schedulers over many seeds and tuning FARS thresholds on validation runs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from faultaware.scheduler.base import Scheduler
from faultaware.scheduler.baselines import (
    BestFitScheduler,
    FirstFitScheduler,
    LeastLoadedScheduler,
    RandomScheduler,
    ReactiveScheduler,
    RoundRobinScheduler,
)
from faultaware.scheduler.fars import FARSConfig, FARSScheduler
from faultaware.simulator.engine import SimConfig, Simulation
from faultaware.simulator.signals import make_signal


def lost_node_minutes(m: dict[str, float], sim: SimConfig) -> float:
    """Tuning objective: node time lost to failures and to mitigation overheads.

    ``wasted compute + checkpoint overhead + migration downtime`` (minutes). Lower is better.
    """
    return (
        m["wasted_compute_min"]
        + m["checkpoint_overhead_min"]
        + m["migrations"] * sim.migration_time_min
    )


def make_scheduler(
    name: str, seed: int, fars: FARSConfig | None = None, reactive_cooldown: float = 60.0
) -> Scheduler:
    """Instantiate a scheduler by name."""
    factories: dict[str, Any] = {
        "random": lambda: RandomScheduler(seed),
        "round_robin": RoundRobinScheduler,
        "first_fit": FirstFitScheduler,
        "best_fit": BestFitScheduler,
        "least_loaded": LeastLoadedScheduler,
        "reactive": lambda: ReactiveScheduler(reactive_cooldown),
        "fars": lambda: FARSScheduler(fars or FARSConfig()),
    }
    return factories[name]()


def run_one(
    sim: SimConfig,
    scheduler: str,
    seed: int,
    signal_spec: dict[str, Any],
    fars: FARSConfig | None = None,
    record_trace: bool = False,
    reactive_cooldown: float = 60.0,
) -> dict[str, Any]:
    """Run a single simulation and return its metrics (plus trace if requested)."""
    signal = make_signal(
        signal_spec["source"],
        signal_spec.get("noise", 0.0),
        Path(signal_spec["scores_file"]) if signal_spec.get("scores_file") else None,
        signal_spec.get("split", "test"),
        signal_spec.get("variant", "fused"),
        signal_spec.get("persistent", True),
    )
    result = Simulation(
        sim, make_scheduler(scheduler, seed, fars, reactive_cooldown), signal, seed, record_trace
    ).run()
    row: dict[str, Any] = {"scheduler": scheduler, "seed": seed, **result.metrics}
    row["lost_node_min"] = lost_node_minutes(result.metrics, sim)
    if record_trace:
        row["trace"] = result.node_risk_trace
    return row


def run_many(
    sim: SimConfig,
    schedulers: list[str],
    seeds: list[int],
    signal_spec: dict[str, Any],
    fars: FARSConfig | None = None,
    n_jobs: int = 8,
    reactive_cooldown: float = 60.0,
) -> pd.DataFrame:
    """Run every (scheduler, seed) pair in parallel; returns one row per run."""
    rows = Parallel(n_jobs=n_jobs)(
        delayed(run_one)(sim, s, seed, signal_spec, fars, False, reactive_cooldown)
        for s in schedulers
        for seed in seeds
    )
    return pd.DataFrame(rows)


def tune_fars(
    sim: SimConfig,
    base: FARSConfig,
    seeds: list[int],
    signal_spec: dict[str, Any],
    grid_low: list[float],
    grid_high: list[float],
    n_jobs: int = 8,
) -> tuple[FARSConfig, np.ndarray]:
    """Grid-search ``(tau_low, tau_high)`` on VALIDATION seeds/signal.

    Returns:
        ``(best_config, grid)`` where ``grid[i, j]`` is mean lost node-minutes for
        ``tau_high = grid_high[i]``, ``tau_low = grid_low[j]`` (NaN where invalid).
    """
    combos = [(i, j) for i, th in enumerate(grid_high) for j, tl in enumerate(grid_low) if tl < th]
    jobs = [(i, j, seed) for i, j in combos for seed in seeds]
    rows = Parallel(n_jobs=n_jobs)(
        delayed(run_one)(
            sim,
            "fars",
            seed,
            signal_spec,
            replace(base, tau_low=grid_low[j], tau_high=grid_high[i]),
        )
        for i, j, seed in jobs
    )
    grid = np.full((len(grid_high), len(grid_low)), np.nan)
    for (i, j, _), row in zip(jobs, rows, strict=True):
        grid[i, j] = (0.0 if np.isnan(grid[i, j]) else grid[i, j]) + row["lost_node_min"]
    grid /= len(seeds)
    i_best, j_best = np.unravel_index(np.nanargmin(grid), grid.shape)
    return replace(base, tau_low=grid_low[j_best], tau_high=grid_high[i_best]), grid
