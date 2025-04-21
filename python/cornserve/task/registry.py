"""Tasks registered and known to the system."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cornserve.task.base import UnitTask


class TaskRegistry:
    """Registry of unit tasks.

    Composite tasks are not registered here; they can simply be invoked
    and will be decomposed into a series of unit task invocations.
    """

    def __init__(self) -> None:
        """Initialize the task registry."""
        self._tasks: dict[str, type[UnitTask]] = {}

    def register(self, task: type[UnitTask], name: str | None = None) -> None:
        """Register a task with the given ID."""
        name = name or task.__class__.__name__
        if name in self._tasks:
            raise ValueError(f"Unit task with {name=} already exists. Unit task names must be unique.")
        self._tasks[name] = task

    def get(self, name: str) -> type[UnitTask]:
        """Get a task by its name."""
        # Lazy import builtin tasks to avoid circular import issues
        import cornserve.task.builtins  # noqa: F401

        if name not in self._tasks:
            raise KeyError(f"Unit task with {name=} not found")
        return self._tasks[name]

    def __contains__(self, name: str) -> bool:
        """Check if a task is registered."""
        return name in self._tasks


TASK_REGISTRY = TaskRegistry()
