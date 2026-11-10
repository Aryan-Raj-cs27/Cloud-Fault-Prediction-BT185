"""Discrete-event cloud simulator (SimPy) with fault injection (CLAUDE.md §9).

Time unit: minutes. Model summary (assumptions are listed in the final report):

* **Workload** — ``n_tasks`` tasks with Poisson arrivals, lognormal work, uniform CPU/memory
  demands and a deadline of ``arrival + slack * work``. Generated up-front from its own RNG.
* **Faults** — each node fails as a renewal process with Weibull time-to-failure. A fraction
  ``1 - unpredictable_fraction`` of failures are preceded by a degradation window of
  ``U(precursor_min)`` minutes (hidden state). A failure kills every task on the node; the
  node is repaired after ``repair_min``. Fault times come from per-node RNGs, so every
  scheduler faces exactly the same faults for a given seed (common random numbers).
* **Checkpointing** — tasks checkpoint every ``interval`` minutes of work at a cost of
  ``checkpoint_cost_min``. On a failure, work since the last completed checkpoint is lost
  and the task is re-queued with a restart penalty.
* **Migration** — moves a running task (state included) to another node in
  ``migration_time_min``.
* **Signals** — every Δt the scheduler receives one risk score per UP node from a
  :class:`~faultaware.simulator.signals.RiskSignal`; it never sees the hidden state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import simpy

from faultaware.scheduler.base import Scheduler
from faultaware.simulator.cluster import (
    Checkpoint,
    ClusterState,
    Migrate,
    Node,
    NodeStatus,
    Task,
)
from faultaware.simulator.signals import RiskSignal

MAX_SIM_MIN = 200_000.0


def exec_time(work: float, interval: float, cost: float) -> float:
    """Wall time to finish ``work`` minutes with a checkpoint after each full ``interval``.

    No checkpoint is taken at the very end (the task simply completes).
    """
    n_ckpt = max(0, math.ceil(work / interval) - 1) if work > 0 else 0
    return work + n_ckpt * cost


def progress(elapsed: float, interval: float, cost: float) -> tuple[float, float]:
    """Work done and work durably saved after ``elapsed`` minutes of execution.

    Returns:
        ``(done, saved)`` with ``saved <= done <= elapsed``.
    """
    cycle = interval + cost
    full = math.floor(elapsed / cycle) if cycle > 0 else 0
    partial = elapsed - full * cycle
    done = full * interval + min(partial, interval)
    return done, full * interval


@dataclass
class SimConfig:
    """Typed view of ``configs/simulator/default.yaml``."""

    n_nodes: int = 40
    n_tasks: int = 5000
    arrival_rate_per_min: float = 2.6
    duration_mean_log: float = 3.6
    duration_sigma_log: float = 0.8
    duration_min: float = 5.0
    duration_max: float = 600.0
    task_cpu: tuple[float, float] = (0.05, 0.35)
    task_mem: tuple[float, float] = (0.05, 0.35)
    deadline_slack: float = 1.3
    tick_min: float = 5.0
    weibull_shape: float = 1.5
    weibull_scale_min: float = 1800.0
    precursor_min: tuple[float, float] = (15.0, 60.0)
    unpredictable_fraction: float = 0.2
    repair_min: float = 30.0
    checkpoint_interval_min: float = 30.0
    checkpoint_cost_min: float = 1.0
    migration_time_min: float = 0.5
    restart_penalty_min: float = 2.0
    idle_w: float = 100.0
    peak_w: float = 250.0

    @classmethod
    def from_cfg(cls, c: Any) -> SimConfig:
        """Build from the OmegaConf ``simulator`` group."""
        return cls(
            n_nodes=c.n_nodes,
            n_tasks=c.n_tasks,
            arrival_rate_per_min=c.arrival_rate_per_min,
            duration_mean_log=c.task_duration.mean_log,
            duration_sigma_log=c.task_duration.sigma_log,
            duration_min=c.task_duration.min,
            duration_max=c.task_duration.max,
            task_cpu=tuple(c.task_cpu),
            task_mem=tuple(c.task_mem),
            deadline_slack=c.deadline_slack,
            tick_min=c.tick_min,
            weibull_shape=c.faults.weibull_shape,
            weibull_scale_min=c.faults.weibull_scale_min,
            precursor_min=tuple(c.faults.precursor_min),
            unpredictable_fraction=c.faults.unpredictable_fraction,
            repair_min=c.faults.repair_min,
            checkpoint_interval_min=c.mitigation.checkpoint_interval_min,
            checkpoint_cost_min=c.mitigation.checkpoint_cost_min,
            migration_time_min=c.mitigation.migration_time_min,
            restart_penalty_min=c.mitigation.restart_penalty_min,
            idle_w=c.energy.idle_w,
            peak_w=c.energy.peak_w,
        )


@dataclass
class SimResult:
    """System-level metrics of one simulation run (CLAUDE.md §8.2)."""

    metrics: dict[str, float]
    node_risk_trace: list[dict[str, Any]] = field(default_factory=list)


class Simulation:
    """One seeded simulation of a scheduler on a synthetic faulty cluster.

    Args:
        cfg: Simulator configuration.
        scheduler: Scheduler under test.
        signal: Risk-signal model.
        seed: Seed; workload and fault RNGs depend only on it (not on the scheduler).
        record_trace: Keep per-tick node risk/state (for the dashboard replay).
    """

    def __init__(
        self,
        cfg: SimConfig,
        scheduler: Scheduler,
        signal: RiskSignal,
        seed: int,
        record_trace: bool = False,
    ) -> None:
        self.cfg = cfg
        self.scheduler = scheduler
        self.signal = signal
        self.seed = seed
        self.record_trace = record_trace
        self.env = simpy.Environment()
        self.nodes = {i: Node(i) for i in range(cfg.n_nodes)}
        self.tasks = self._make_tasks(np.random.default_rng([seed, 1]))
        self.pending: list[tuple[Task, float]] = []  # (task, startup delay)
        self.signal_rng = np.random.default_rng([seed, 3])
        self.done = 0
        self.all_done = self.env.event()
        self.stats = {
            "migrations": 0,
            "checkpoints_forced": 0,
            "node_failures": 0,
            "predictable_failures": 0,
            "energy_wh": 0.0,
            "util_sum": 0.0,
            "util_ticks": 0,
            "checkpoint_overhead_min": 0.0,
        }
        self.trace: list[dict[str, Any]] = []
        self.failure_log: list[tuple[float, int]] = []

    # ------------------------------------------------------------------ setup
    def _make_tasks(self, rng: np.random.Generator) -> dict[int, Task]:
        c = self.cfg
        arrivals = np.cumsum(rng.exponential(1.0 / c.arrival_rate_per_min, c.n_tasks))
        work = np.clip(
            rng.lognormal(c.duration_mean_log, c.duration_sigma_log, c.n_tasks),
            c.duration_min,
            c.duration_max,
        )
        cpu = rng.uniform(*c.task_cpu, c.n_tasks)
        mem = rng.uniform(*c.task_mem, c.n_tasks)
        return {
            i: Task(
                i,
                float(arrivals[i]),
                float(work[i]),
                float(cpu[i]),
                float(mem[i]),
                float(arrivals[i] + c.deadline_slack * work[i]),
                checkpoint_interval=c.checkpoint_interval_min,
            )
            for i in range(c.n_tasks)
        }

    def state(self) -> ClusterState:
        """Current cluster view for the scheduler."""
        return ClusterState(self.env.now, self.nodes)

    # --------------------------------------------------------------- processes
    def _arrivals(self) -> Any:
        for task in sorted(self.tasks.values(), key=lambda t: t.arrival):
            yield self.env.timeout(max(0.0, task.arrival - self.env.now))
            self.pending.append((task, 0.0))
            self._dispatch()

    def _node_faults(self, node: Node) -> Any:
        c = self.cfg
        rng = np.random.default_rng([self.seed, 2, node.node_id])
        while True:
            ttf = c.weibull_scale_min * rng.weibull(c.weibull_shape)
            predictable = rng.random() >= c.unpredictable_fraction
            precursor = rng.uniform(*c.precursor_min)
            if predictable and ttf > precursor:
                yield self.env.timeout(ttf - precursor)
                node.degraded = True
                yield self.env.timeout(precursor)
            else:
                yield self.env.timeout(ttf)
            self._fail(node, predictable)
            yield self.env.timeout(c.repair_min)
            node.status = NodeStatus.UP
            self._dispatch()

    def _fail(self, node: Node, predictable: bool) -> None:
        node.degraded = False
        node.status = NodeStatus.DOWN
        node.last_failure = self.env.now
        node.failures += 1
        self.failure_log.append((self.env.now, node.node_id))
        self.stats["node_failures"] += 1
        self.stats["predictable_failures"] += int(predictable)
        for task in list(node.tasks.values()):
            if task.process is not None and task.process.is_alive:
                task.process.interrupt(("fail",))

    def _ticks(self) -> Any:
        c = self.cfg
        while True:
            risk = {
                nid: self.signal(nid, n.degraded, self.signal_rng)
                for nid, n in self.nodes.items()
                if n.status is NodeStatus.UP
            }
            for action in self.scheduler.on_tick(self.state(), risk):
                self._apply(action)
            self._dispatch()
            self._account(risk)
            yield self.env.timeout(c.tick_min)

    def _account(self, risk: dict[int, float]) -> None:
        c = self.cfg
        up = [n for n in self.nodes.values() if n.status is NodeStatus.UP]
        if up:
            util = sum(n.utilisation() for n in up) / len(up)
            self.stats["util_sum"] += util
            self.stats["util_ticks"] += 1
        power = sum(c.idle_w + (c.peak_w - c.idle_w) * n.utilisation() for n in up)
        self.stats["energy_wh"] += power * c.tick_min / 60.0
        if self.record_trace:
            self.trace.append(
                {
                    "t": round(self.env.now, 1),
                    "risk": {int(k): round(v, 3) for k, v in risk.items()},
                    "down": [n.node_id for n in self.nodes.values() if n.status is NodeStatus.DOWN],
                    "degraded": [n.node_id for n in self.nodes.values() if n.degraded],
                    "running": sum(len(n.tasks) for n in self.nodes.values()),
                    "queued": len(self.pending),
                }
            )

    def _run_task(self, task: Task, node: Node, delay: float) -> Any:
        c = self.cfg
        while True:
            seg_start: float | None = None
            try:
                if delay > 0:
                    yield self.env.timeout(delay)
                    delay = 0.0
                seg_start = self.env.now
                yield self.env.timeout(
                    exec_time(task.remaining_min, task.checkpoint_interval, c.checkpoint_cost_min)
                )
                self.stats["checkpoint_overhead_min"] += (
                    exec_time(task.remaining_min, task.checkpoint_interval, c.checkpoint_cost_min)
                    - task.remaining_min
                )
                task.remaining_min = 0.0
                task.finished_at = self.env.now
                del node.tasks[task.task_id]
                task.process = None
                self.done += 1
                if self.done == len(self.tasks):
                    self.all_done.succeed()
                self._dispatch()
                return
            except simpy.Interrupt as intr:
                cause: tuple[Any, ...] = intr.cause or ("fail",)
                elapsed = self.env.now - seg_start if seg_start is not None else 0.0
                done, saved = progress(elapsed, task.checkpoint_interval, c.checkpoint_cost_min)
                self.stats["checkpoint_overhead_min"] += max(0.0, elapsed - done)
                kind = cause[0]
                if kind == "fail":
                    task.remaining_min = max(0.0, task.remaining_min - saved)
                    task.wasted_min += done - saved
                    task.interruptions += 1
                    del node.tasks[task.task_id]
                    task.node, task.process = None, None
                    self.pending.insert(0, (task, c.restart_penalty_min))
                    return
                task.remaining_min = max(0.0, task.remaining_min - done)
                if kind == "checkpoint":
                    task.checkpoint_interval = cause[1]
                    self.stats["checkpoints_forced"] += 1
                    self.stats["checkpoint_overhead_min"] += c.checkpoint_cost_min
                    delay = c.checkpoint_cost_min
                elif kind == "migrate":
                    node = self.nodes[cause[1]]  # already moved by _apply
                    task.migrations += 1
                    self.stats["migrations"] += 1
                    delay = c.migration_time_min

    # ---------------------------------------------------------------- control
    def _start(self, task: Task, node: Node, delay: float) -> None:
        node.tasks[task.task_id] = task
        task.node = node.node_id
        if task.started_at is None:
            task.started_at = self.env.now + delay
        task.process = self.env.process(self._run_task(task, node, delay))

    def _dispatch(self) -> None:
        if not self.pending:
            return
        state = self.state()
        still: list[tuple[Task, float]] = []
        for task, delay in self.pending:
            nid = self.scheduler.place(task, state)
            if nid is not None and self.nodes[nid].fits(task):
                self._start(task, self.nodes[nid], delay)
            else:
                still.append((task, delay))
        self.pending = still

    def _apply(self, action: Migrate | Checkpoint) -> None:
        task = self.tasks[action.task_id]
        if task.process is None or not task.process.is_alive or task.node is None:
            return
        if isinstance(action, Migrate):
            target = self.nodes[action.target]
            if action.target == task.node or not target.fits(task):
                return
            # Move the task now so that the target's capacity is reserved before any
            # placement in this same instant; the process handler only accounts progress.
            source = self.nodes[task.node]
            del source.tasks[task.task_id]
            target.tasks[task.task_id] = task
            task.node = target.node_id
            task.process.interrupt(("migrate", action.target))
        else:
            task.process.interrupt(("checkpoint", action.interval_min))

    # -------------------------------------------------------------------- run
    def run(self) -> SimResult:
        """Simulate until all tasks complete (or the safety horizon) and return metrics."""
        self.env.process(self._arrivals())
        for node in self.nodes.values():
            self.env.process(self._node_faults(node))
        self.env.process(self._ticks())
        self.env.run(until=simpy.AnyOf(self.env, [self.all_done, self.env.timeout(MAX_SIM_MIN)]))
        return SimResult(self._metrics(), self.trace)

    def _metrics(self) -> dict[str, float]:
        tasks = list(self.tasks.values())
        finished = [t for t in tasks if t.finished_at is not None]
        resp = [t.finished_at - t.arrival for t in finished]  # type: ignore[operator]
        total_work = sum(t.work_min for t in tasks)
        interruptions = sum(t.interruptions for t in tasks)
        return {
            "completed": float(len(finished)),
            "completion_rate": len(finished) / len(tasks),
            "task_interruptions": float(interruptions),
            "interrupted_task_frac": float(np.mean([t.interruptions > 0 for t in tasks])),
            "wasted_compute_min": float(sum(t.wasted_min for t in tasks)),
            "wasted_compute_frac": float(sum(t.wasted_min for t in tasks) / total_work),
            "sla_violation_rate": float(
                np.mean([(t.finished_at is None) or (t.finished_at > t.deadline) for t in tasks])
            ),
            "mean_response_min": float(np.mean(resp)) if resp else float("nan"),
            "makespan_min": float(max((t.finished_at or 0.0) for t in tasks)),
            "utilisation": self.stats["util_sum"] / max(1, self.stats["util_ticks"]),
            "migrations": float(self.stats["migrations"]),
            "forced_checkpoints": float(self.stats["checkpoints_forced"]),
            "checkpoint_overhead_min": float(self.stats["checkpoint_overhead_min"]),
            "energy_kwh": self.stats["energy_wh"] / 1000.0,
            "node_failures": float(self.stats["node_failures"]),
        }
