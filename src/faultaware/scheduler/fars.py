"""FARS — Fault-Aware Risk-based Scheduler (the project's core algorithm, CLAUDE.md §2.1).

Every Δt the scheduler receives a risk score ``r_i`` per node (fused, calibrated predictor
output) and smooths it with an EWMA. Then:

* ``r_i >= tau_high``  -> proactively migrate tasks off node ``i`` to the best low-risk node;
  if no target fits, checkpoint them immediately at the elevated frequency.
* ``tau_low <= r_i < tau_high`` -> no new placements on ``i`` (if any alternative exists);
  running tasks switch to the elevated checkpoint frequency.
* ``r_i < tau_low`` -> normal placement; tasks return to the default checkpoint interval.

Placement score of node ``j`` for a task: ``w1 * free_capacity_j - w2 * r_j - w3 * cost_j``
where ``cost_j`` is the migration cost (task memory footprint) when moving a running task and
0 for new tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from faultaware.simulator.cluster import (
    Action,
    Checkpoint,
    ClusterState,
    Migrate,
    Node,
    NodeId,
    Task,
)


@dataclass
class FARSConfig:
    """FARS hyperparameters (tuned on validation simulations)."""

    tau_low: float = 0.3
    tau_high: float = 0.6
    w_free: float = 1.0
    w_risk: float = 2.0
    w_migration: float = 0.5
    smoothing: float = 0.5  # EWMA weight of the newest observation
    elevated_interval: float = 10.0  # checkpoint interval on risky nodes (min)
    default_interval: float = 30.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.tau_low <= self.tau_high <= 1.0:
            raise ValueError("need 0 <= tau_low <= tau_high <= 1")
        if not 0.0 < self.smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")


@dataclass
class FARSScheduler:
    """Fault-aware risk-based scheduler."""

    cfg: FARSConfig = field(default_factory=FARSConfig)
    name: str = "fars"
    risk: dict[NodeId, float] = field(default_factory=dict)

    def _r(self, node_id: NodeId) -> float:
        return self.risk.get(node_id, 0.0)

    def _score(self, node: Node, task: Task, migration: bool) -> float:
        free_after = node.free_capacity() - max(task.cpu, task.mem)
        cost = task.mem if migration else 0.0
        return (
            self.cfg.w_free * free_after
            - self.cfg.w_risk * self._r(node.node_id)
            - self.cfg.w_migration * cost
        )

    def place(self, task: Task, cluster: ClusterState) -> NodeId | None:
        cands = cluster.candidates(task)
        if not cands:
            return None
        low = [n for n in cands if self._r(n.node_id) < self.cfg.tau_low]
        mid = [n for n in cands if self._r(n.node_id) < self.cfg.tau_high]
        pool = low or mid
        if not pool:
            return None  # only high-risk nodes have room: wait for a safer slot
        return max(pool, key=lambda n: (self._score(n, task, False), -n.node_id)).node_id

    def on_tick(self, cluster: ClusterState, risk: dict[NodeId, float]) -> list[Action]:
        a = self.cfg.smoothing
        for nid, r in risk.items():
            self.risk[nid] = a * r + (1 - a) * self.risk.get(nid, r)
        actions: list[Action] = []
        reserved: dict[NodeId, tuple[float, float]] = {}
        for node in cluster.up_nodes():
            r = self._r(node.node_id)
            for task in list(node.tasks.values()):
                if r >= self.cfg.tau_high:
                    target = self._migration_target(task, node, cluster, reserved)
                    if target is not None:
                        actions.append(Migrate(task.task_id, target))
                        continue
                    if task.checkpoint_interval > self.cfg.elevated_interval:
                        actions.append(Checkpoint(task.task_id, self.cfg.elevated_interval))
                elif r >= self.cfg.tau_low:
                    if task.checkpoint_interval > self.cfg.elevated_interval:
                        actions.append(Checkpoint(task.task_id, self.cfg.elevated_interval))
                elif task.checkpoint_interval < self.cfg.default_interval:
                    actions.append(Checkpoint(task.task_id, self.cfg.default_interval))
        return actions

    def _migration_target(
        self,
        task: Task,
        source: Node,
        cluster: ClusterState,
        reserved: dict[NodeId, tuple[float, float]],
    ) -> NodeId | None:
        best: tuple[float, int] | None = None
        for node in cluster.up_nodes():
            if node.node_id == source.node_id or self._r(node.node_id) >= self.cfg.tau_low:
                continue
            rc, rm = reserved.get(node.node_id, (0.0, 0.0))
            if (
                node.cpu_used + rc + task.cpu > node.cpu_capacity + 1e-9
                or node.mem_used + rm + task.mem > node.mem_capacity + 1e-9
            ):
                continue
            key = (self._score(node, task, True), -node.node_id)
            if best is None or key > best:
                best = key
        if best is None:
            return None
        target = -best[1]
        rc, rm = reserved.get(target, (0.0, 0.0))
        reserved[target] = (rc + task.cpu, rm + task.mem)
        return target
