"""Fault-unaware and reactive baseline schedulers (CLAUDE.md §2.1)."""

from __future__ import annotations

import numpy as np

from faultaware.simulator.cluster import Action, ClusterState, Node, NodeId, Task


class _NoMitigation:
    """Baselines ignore risk signals: they only restart tasks after a failure."""

    name = "baseline"

    def on_tick(self, cluster: ClusterState, risk: dict[NodeId, float]) -> list[Action]:
        return []


class RandomScheduler(_NoMitigation):
    """Uniformly random feasible node."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def place(self, task: Task, cluster: ClusterState) -> NodeId | None:
        cands = cluster.candidates(task)
        return cands[int(self.rng.integers(len(cands)))].node_id if cands else None


class RoundRobinScheduler(_NoMitigation):
    """Cycle through nodes; take the first feasible one after the last choice."""

    name = "round_robin"

    def __init__(self) -> None:
        self.pointer = -1

    def place(self, task: Task, cluster: ClusterState) -> NodeId | None:
        ids = sorted(cluster.nodes)
        n = len(ids)
        for step in range(1, n + 1):
            nid = ids[(self.pointer + step) % n]
            if cluster.nodes[nid].fits(task):
                self.pointer = (self.pointer + step) % n
                return nid
        return None


class FirstFitScheduler(_NoMitigation):
    """Lowest-id feasible node."""

    name = "first_fit"

    def place(self, task: Task, cluster: ClusterState) -> NodeId | None:
        cands = cluster.candidates(task)
        return cands[0].node_id if cands else None


class BestFitScheduler(_NoMitigation):
    """Feasible node with the least free capacity left after placement (tight packing)."""

    name = "best_fit"

    def place(self, task: Task, cluster: ClusterState) -> NodeId | None:
        cands = cluster.candidates(task)
        if not cands:
            return None
        return min(cands, key=lambda n: (n.free_capacity(), n.node_id)).node_id


class LeastLoadedScheduler(_NoMitigation):
    """Feasible node with the most free capacity (load spreading)."""

    name = "least_loaded"

    def place(self, task: Task, cluster: ClusterState) -> NodeId | None:
        cands = cluster.candidates(task)
        if not cands:
            return None
        return max(cands, key=lambda n: (n.free_capacity(), -n.node_id)).node_id


class ReactiveScheduler(LeastLoadedScheduler):
    """Least-loaded, but avoids nodes that failed within the last ``cooldown_min`` minutes.

    This is the classic *reactive* fault-tolerance baseline: it learns from failures only
    after they happen.
    """

    name = "reactive"

    def __init__(self, cooldown_min: float = 60.0) -> None:
        self.cooldown = cooldown_min

    def place(self, task: Task, cluster: ClusterState) -> NodeId | None:
        cands = cluster.candidates(task)
        if not cands:
            return None
        fresh = [n for n in cands if cluster.now - n.last_failure > self.cooldown]
        pool: list[Node] = fresh or cands
        return max(pool, key=lambda n: (n.free_capacity(), -n.node_id)).node_id
