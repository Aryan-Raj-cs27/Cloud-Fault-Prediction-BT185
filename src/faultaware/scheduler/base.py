"""Scheduler interface (CLAUDE.md §6.2)."""

from __future__ import annotations

from typing import Protocol

from faultaware.simulator.cluster import Action, ClusterState, NodeId, Task


class Scheduler(Protocol):
    """Placement + periodic mitigation policy."""

    name: str

    def place(self, task: Task, cluster: ClusterState) -> NodeId | None:
        """Choose a node for ``task`` (must fit), or ``None`` to keep it queued."""
        ...

    def on_tick(self, cluster: ClusterState, risk: dict[NodeId, float]) -> list[Action]:
        """Observe per-node risk scores every Δt and return mitigation actions."""
        ...
