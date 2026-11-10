"""Cluster state: nodes, tasks and scheduler actions (shared by simulator and schedulers)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

NodeId = int


class NodeStatus(StrEnum):
    """Lifecycle of a node."""

    UP = "up"
    DOWN = "down"


@dataclass
class Task:
    """A unit of work with fixed resource demands.

    Attributes:
        work_min: Total compute minutes required (excluding checkpoint overheads).
        remaining_min: Work still to do from the last durable state.
    """

    task_id: int
    arrival: float
    work_min: float
    cpu: float
    mem: float
    deadline: float
    remaining_min: float = field(init=False)
    node: NodeId | None = None
    started_at: float | None = None
    finished_at: float | None = None
    interruptions: int = 0
    migrations: int = 0
    wasted_min: float = 0.0
    checkpoint_interval: float = 30.0
    process: Any = field(default=None, repr=False)  # simpy.Process while running

    def __post_init__(self) -> None:
        self.remaining_min = self.work_min


@dataclass
class Node:
    """A compute node with CPU/memory capacity (normalised to 1.0)."""

    node_id: NodeId
    cpu_capacity: float = 1.0
    mem_capacity: float = 1.0
    status: NodeStatus = NodeStatus.UP
    tasks: dict[int, Task] = field(default_factory=dict)
    last_failure: float = -1e18
    degraded: bool = False  # ground truth: in a pre-failure degradation window
    failures: int = 0

    @property
    def cpu_used(self) -> float:
        return sum(t.cpu for t in self.tasks.values())

    @property
    def mem_used(self) -> float:
        return sum(t.mem for t in self.tasks.values())

    def fits(self, task: Task) -> bool:
        """Whether ``task`` fits in the remaining capacity of an UP node."""
        return (
            self.status is NodeStatus.UP
            and self.cpu_used + task.cpu <= self.cpu_capacity + 1e-9
            and self.mem_used + task.mem <= self.mem_capacity + 1e-9
        )

    def free_capacity(self) -> float:
        """Free fraction of the scarcer resource, in ``[0, 1]``."""
        return max(0.0, min(self.cpu_capacity - self.cpu_used, self.mem_capacity - self.mem_used))

    def utilisation(self) -> float:
        """CPU utilisation in ``[0, 1]``."""
        return min(1.0, self.cpu_used / self.cpu_capacity)


@dataclass
class ClusterState:
    """Read-mostly view of the cluster passed to schedulers."""

    now: float
    nodes: dict[NodeId, Node]

    def up_nodes(self) -> list[Node]:
        """Nodes currently UP, in id order."""
        return [n for n in self.nodes.values() if n.status is NodeStatus.UP]

    def candidates(self, task: Task) -> list[Node]:
        """UP nodes with room for ``task``, in id order."""
        return [n for n in self.nodes.values() if n.fits(task)]


@dataclass(frozen=True)
class Migrate:
    """Move a running task to another node (keeps in-memory progress)."""

    task_id: int
    target: NodeId


@dataclass(frozen=True)
class Checkpoint:
    """Checkpoint a running task now and switch to a new checkpoint interval."""

    task_id: int
    interval_min: float


Action = Migrate | Checkpoint
