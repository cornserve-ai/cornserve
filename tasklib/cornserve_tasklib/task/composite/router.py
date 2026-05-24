"""Composite task that routes requests across multiple tasks by weights."""

from __future__ import annotations

import hashlib
from threading import Lock
from typing import Generic, TypeVar, cast

from pydantic import PrivateAttr, model_validator

from cornserve.logging import get_logger
from cornserve.task.base import Task, TaskInput, TaskOutput, task_context

InputT = TypeVar("InputT", bound=TaskInput)
OutputT = TypeVar("OutputT", bound=TaskOutput)

logger = get_logger(__name__)


class RouterApp(Task[InputT, OutputT], Generic[InputT, OutputT]):
    """Route each request to one task from a weighted task set.

    Attributes:
        routing_tasks: Candidate tasks to route to. Each can be a unit task or
            another composite task.
        routing_weights: Non-negative weights with the same length as
            ``routing_tasks``.

    Notes:
        Routing is deterministic per request so the same request is routed to the
        same subtask in both record and replay phases.
    """

    routing_tasks: list[Task]
    routing_weights: list[float]

    _routing_cdf: list[float] = PrivateAttr(default_factory=list)
    _routing_salt: str = PrivateAttr(default="")
    # NOTE: Routing stats are process-local (for example, per gateway pod process)
    # and are not aggregated across multiple gateway replicas.
    _route_counts: list[int] = PrivateAttr(default_factory=list)
    _route_total: int = PrivateAttr(default=0)
    _route_lock: Lock = PrivateAttr(default_factory=Lock)

    @model_validator(mode="after")
    def _validate_routing_config(self) -> RouterApp[InputT, OutputT]:
        """Validate routing config shape and values."""
        if not self.routing_tasks:
            raise ValueError("routing_tasks must contain at least one task")

        if len(self.routing_tasks) != len(self.routing_weights):
            raise ValueError(
                "routing_weights must have the same length as routing_tasks"
            )

        if any(w < 0 for w in self.routing_weights):
            raise ValueError("routing_weights must be non-negative")

        if sum(self.routing_weights) <= 0:
            raise ValueError("routing_weights must sum to a positive value")

        return self

    def post_init(self) -> None:
        """Initialize cached routing state and register subtasks for discovery."""
        if not self.routing_tasks or not self.routing_weights:
            self._routing_cdf = []
            self._route_counts = []
            self._route_total = 0
            return

        total_weight = sum(self.routing_weights)
        if total_weight <= 0:
            self._routing_cdf = []
            self._route_counts = []
            self._route_total = 0
            return

        running = 0.0
        self._routing_cdf = []
        for weight in self.routing_weights:
            running += weight / total_weight
            self._routing_cdf.append(running)
        if self._routing_cdf:
            self._routing_cdf[-1] = 1.0

        self._route_counts = [0] * len(self.routing_tasks)
        self._route_total = 0

        # Deterministic salt so nested RouterApps produce independent hash
        # values while preserving record/replay determinism.  The salt is
        # derived from the routing structure itself.
        salt_source = ",".join(f"{w:.6f}" for w in self.routing_weights)
        self._routing_salt = hashlib.sha256(salt_source.encode()).hexdigest()[:16]

        # `routing_tasks` are provided as pre-built objects, so register them as
        # subtasks explicitly for `discover_unit_tasks` traversal.
        for task in self.routing_tasks:
            attr_name = f"__subtask_{len(self.subtask_attr_names)}__"
            setattr(self, attr_name, task)
            self.subtask_attr_names.append(attr_name)

    def _routing_key(self, task_input: InputT) -> str:
        """Build a deterministic key for request-level routing."""
        request_id = getattr(task_input, "request_id", None)
        if isinstance(request_id, str) and request_id:
            return request_id
        return task_input.model_dump_json()

    def _choose_index(self, task_input: InputT) -> int:
        """Choose a task index from weighted CDF using deterministic hash."""
        key = self._routing_salt + self._routing_key(task_input)
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        sample = int.from_bytes(digest[:8], byteorder="big") / float(2**64)

        for i, cutoff in enumerate(self._routing_cdf):
            if sample < cutoff:
                return i
        return len(self._routing_cdf) - 1

    def _should_track_route_decision(self) -> bool:
        """Return whether route stats should be updated for this invoke."""
        try:
            ctx = task_context.get()
        except LookupError:
            return True

        return not ctx.is_replaying

    def _record_route_decision(self, index: int) -> None:
        """Update routing counters and log the latest cumulative stats."""
        with self._route_lock:
            self._route_counts[index] += 1
            self._route_total += 1
            counts = list(self._route_counts)
            total = self._route_total

        percentages = [round(count / total, 2) for count in counts]
        logger.info(
            "RouterApp current routing index [%d], accumulated routing %s percentages %s",
            index,
            counts,
            percentages,
        )

    def get_routing_stats(self) -> tuple[list[int], int, list[float]]:
        """Return routing counters as (counts, total, percentages)."""
        with self._route_lock:
            counts = list(self._route_counts)
            total = self._route_total

        if total == 0:
            return counts, total, [0.0] * len(counts)
        return counts, total, [100.0 * count / total for count in counts]

    def invoke(self, task_input: InputT) -> OutputT:
        """Route and invoke one subtask."""
        index = self._choose_index(task_input)
        if self._should_track_route_decision():
            self._record_route_decision(index)

        routed = self.routing_tasks[index]
        return cast(OutputT, routed.invoke(task_input))  # type: ignore[arg-type]
