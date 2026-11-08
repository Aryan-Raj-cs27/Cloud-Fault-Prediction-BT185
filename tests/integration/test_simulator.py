"""Simulator sanity tests (CLAUDE.md §9): oracle beats baselines, random doesn't, no lost tasks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from faultaware.scheduler.baselines import (
    BestFitScheduler,
    FirstFitScheduler,
    LeastLoadedScheduler,
    RandomScheduler,
    ReactiveScheduler,
    RoundRobinScheduler,
)
from faultaware.scheduler.fars import FARSConfig, FARSScheduler
from faultaware.simulator.cluster import ClusterState, Node, Task
from faultaware.simulator.engine import SimConfig, Simulation, exec_time, progress
from faultaware.simulator.experiment import run_many, tune_fars
from faultaware.simulator.signals import (
    EmpiricalSignal,
    OracleSignal,
    RandomSignal,
    make_signal,
)

SMALL = SimConfig(n_nodes=12, n_tasks=500, arrival_rate_per_min=0.8, weibull_scale_min=600.0)
SEEDS = list(range(6))


def _run(scheduler_factory, signal, seed: int) -> dict[str, float]:  # type: ignore[no-untyped-def]
    return Simulation(SMALL, scheduler_factory(), signal, seed).run().metrics


def test_exec_time_and_progress() -> None:
    assert exec_time(90, 30, 1) == 92  # checkpoints after 30 and 60, none at the end
    assert exec_time(0, 30, 1) == 0
    done, saved = progress(45, 30, 1)  # one checkpoint (31 min) + 14 min of work
    assert (done, saved) == (44, 30)
    done, saved = progress(30.5, 30, 1)  # writing the first checkpoint
    assert (done, saved) == (30, 0)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RandomScheduler(0),
        RoundRobinScheduler,
        FirstFitScheduler,
        BestFitScheduler,
        LeastLoadedScheduler,
        ReactiveScheduler,
        FARSScheduler,
    ],
)
def test_no_task_lost_or_duplicated(factory) -> None:  # type: ignore[no-untyped-def]
    sim = Simulation(SMALL, factory(), OracleSignal(), seed=1)
    m = sim.run().metrics
    assert m["completed"] == SMALL.n_tasks
    for task in sim.tasks.values():
        assert task.finished_at is not None and task.remaining_min == 0
    assert all(not n.tasks for n in sim.nodes.values())


def test_simulation_is_deterministic() -> None:
    a = _run(FARSScheduler, EmpiricalSignal(np.array([0.9]), np.array([0.1])), 3)
    b = _run(FARSScheduler, EmpiricalSignal(np.array([0.9]), np.array([0.1])), 3)
    assert a == b


def test_faults_identical_across_schedulers() -> None:
    a = Simulation(SMALL, LeastLoadedScheduler(), OracleSignal(), 2)
    b = Simulation(SMALL, FARSScheduler(), OracleSignal(), 2)
    a.run()
    b.run()
    horizon = min(a.env.now, b.env.now)
    fa = [f for f in a.failure_log if f[0] <= horizon]
    fb = [f for f in b.failure_log if f[0] <= horizon]
    assert fa == fb and len(fa) > 0


def test_oracle_fars_beats_baseline() -> None:
    ll = [_run(LeastLoadedScheduler, OracleSignal(), s)["task_interruptions"] for s in SEEDS]
    fo = [_run(FARSScheduler, OracleSignal(), s)["task_interruptions"] for s in SEEDS]
    assert np.mean(fo) < 0.6 * np.mean(ll)


def test_random_predictor_does_not_reduce_failures() -> None:
    ll = [_run(LeastLoadedScheduler, RandomSignal(), s)["task_interruptions"] for s in SEEDS]
    fr = [_run(FARSScheduler, RandomSignal(), s)["task_interruptions"] for s in SEEDS]
    assert np.mean(fr) > 0.8 * np.mean(ll)


def test_signals() -> None:
    rng = np.random.default_rng(0)
    assert OracleSignal()(0, True, rng) == 1.0 and OracleSignal()(0, False, rng) == 0.0
    assert 0 <= RandomSignal()(0, True, rng) <= 1
    sig = EmpiricalSignal(np.array([0.7, 0.8, 0.9]), np.array([0.1]), persistent=True)
    first = sig(5, True, rng)
    assert all(sig(5, True, rng) == first for _ in range(10))  # persistent within episode
    assert sig(5, False, rng) == 0.1
    noisy = EmpiricalSignal(np.array([1.0]), np.array([0.0]), noise=1.0)
    assert 0 < noisy(1, False, rng) < 1
    with pytest.raises(ValueError):
        EmpiricalSignal(np.array([]), np.array([0.1]))
    with pytest.raises(ValueError):
        EmpiricalSignal(np.array([0.5]), np.array([0.1]), noise=2.0)


def test_make_signal(tmp_path: Path) -> None:
    f = tmp_path / "d.npz"
    np.savez(f, test_fused_pos=np.array([0.9]), test_fused_neg=np.array([0.1]))
    assert isinstance(make_signal("empirical", scores_file=f), EmpiricalSignal)
    assert isinstance(make_signal("oracle"), OracleSignal)
    assert isinstance(make_signal("random"), RandomSignal)
    with pytest.raises(ValueError):
        make_signal("empirical")
    with pytest.raises(ValueError):
        make_signal("bogus")


def test_fars_config_validation() -> None:
    with pytest.raises(ValueError):
        FARSConfig(tau_low=0.8, tau_high=0.2)
    with pytest.raises(ValueError):
        FARSConfig(smoothing=0.0)


def test_fars_placement_avoids_risky_nodes() -> None:
    nodes = {i: Node(i) for i in range(3)}
    fars = FARSScheduler(FARSConfig(tau_low=0.3, tau_high=0.6, smoothing=1.0))
    fars.on_tick(ClusterState(0, nodes), {0: 0.9, 1: 0.4, 2: 0.1})
    task = Task(0, 0.0, 10.0, 0.1, 0.1, 100.0)
    assert fars.place(task, ClusterState(0, nodes)) == 2
    fars.on_tick(ClusterState(0, nodes), {0: 0.9, 1: 0.95, 2: 0.99})
    assert fars.place(task, ClusterState(0, nodes)) is None


def test_fars_migrates_off_high_risk_node() -> None:
    nodes = {i: Node(i) for i in range(2)}
    task = Task(0, 0.0, 10.0, 0.2, 0.2, 100.0)
    nodes[0].tasks[0] = task
    fars = FARSScheduler(FARSConfig(smoothing=1.0))
    actions = fars.on_tick(ClusterState(0, nodes), {0: 0.95, 1: 0.0})
    assert [type(a).__name__ for a in actions] == ["Migrate"]


def test_reactive_avoids_recently_failed() -> None:
    nodes = {i: Node(i) for i in range(2)}
    nodes[0].last_failure = 95.0
    task = Task(0, 0.0, 10.0, 0.1, 0.1, 100.0)
    assert ReactiveScheduler(60).place(task, ClusterState(100.0, nodes)) == 1


def test_run_many_and_tuning(tmp_path: Path) -> None:
    f = tmp_path / "d.npz"
    np.savez(f, val_fused_pos=np.array([0.9, 0.8]), val_fused_neg=np.array([0.05, 0.1]))
    spec = {"source": "empirical", "scores_file": str(f), "split": "val"}
    df = run_many(SMALL, ["least_loaded", "fars"], [0, 1], spec, n_jobs=1)
    assert len(df) == 4 and "lost_node_min" in df
    best, grid = tune_fars(SMALL, FARSConfig(), [0], spec, [0.2, 0.5], [0.4, 0.7], n_jobs=1)
    assert best.tau_low < best.tau_high
    assert np.isnan(grid[0, 1])  # tau_low 0.5 >= tau_high 0.4 is invalid
