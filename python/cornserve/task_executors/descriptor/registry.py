"""Task execution descriptor registry.

Task execution descriptor classes register themselves to the registry
specifying a task they can execute. Exactly one descriptor class per
task is marked as the default descriptor class. The registry is used
to look up the descriptor class for a task when executing it.

`DESCRIPTOR_REGISTRY` is a singleton instance of the registry.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cornserve.task.base import UnitTask
    from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor

DEFAULT = "__default_descriptor__"


class TaskExecutionDescriptorRegistry:
    """Registry for task execution descriptors."""

    def __init__(self) -> None:
        """Initialize the registry."""
        self.registry: dict[type[UnitTask], dict[str, type[TaskExecutionDescriptor]]] = defaultdict(dict)
        self.default_registry: dict[type[UnitTask], type[TaskExecutionDescriptor]] = {}

    def register(
        self,
        task: type[UnitTask],
        descriptor: type[TaskExecutionDescriptor],
        name: str | None = None,
        default: bool = False,
    ) -> None:
        """Register a task execution descriptor.

        Args:
            task: The task class to register the descriptor for.
            descriptor: The task execution descriptor class.
            name: The name of the descriptor. If None, use the class name.
            default: Whether this is the default descriptor for the task.
        """
        if name is None:
            name = descriptor.__name__

        if name in self.registry[task]:
            raise ValueError(f"Descriptor {name} already registered for task {task.__name__}")

        self.registry[task][name] = descriptor

        if default:
            if task in self.default_registry:
                raise ValueError(f"Default descriptor already registered for task {task.__name__}")
            self.default_registry[task] = descriptor

    def get(self, task: type[UnitTask], name: str | None = None) -> type[TaskExecutionDescriptor]:
        """Get the task execution descriptor for a task.

        Args:
            task: The task class to get the descriptor for.
            name: The name of the descriptor. If None, use the default descriptor.
        """
        # Lazily import built-in descriptors to avoid circular imports
        import cornserve.task_executors.descriptor.builtins  # noqa: F401

        if task not in self.registry:
            raise ValueError(f"No descriptors registered for task {task.__name__}")

        if name is None:
            if task not in self.default_registry:
                raise ValueError(f"No default descriptor registered for task {task.__name__}")
            return self.default_registry[task]

        if name not in self.registry[task]:
            raise ValueError(f"Descriptor {name} not registered for task {task.__name__}")

        return self.registry[task][name]


DESCRIPTOR_REGISTRY = TaskExecutionDescriptorRegistry()
