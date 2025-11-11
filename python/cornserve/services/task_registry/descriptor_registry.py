"""Registry for execution descriptor classes dynamically loaded from k8s CRs."""

from __future__ import annotations

import base64
from collections import defaultdict
from typing import TYPE_CHECKING

from cornserve.logging import get_logger
from cornserve.services.task_registry.task_class_registry import TASK_CLASS_REGISTRY
from cornserve.services.task_registry.util import write_to_file_and_import
from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor

if TYPE_CHECKING:
    from cornserve.task.base import UnitTask

logger = get_logger(__name__)


class TaskExecutionDescriptorRegistry:
    """Registry for dynamically loaded `TaskExecutionDescriptor` classes per `UnitTask`."""

    def __init__(self) -> None:
        """Initialize in-memory registries and pending queues."""
        self.registry: dict[str, dict[str, type[TaskExecutionDescriptor]]] = defaultdict(dict)
        self.default_registry: dict[str, type[TaskExecutionDescriptor]] = {}
        # This stores descriptors that arrive before their corresponding task class is loaded
        # each item is (decoded_source, module_name, descriptor_class_name, is_default)
        self._pending: dict[str, list[tuple[str, str, str, bool]]] = defaultdict(list)

    def _install_and_register(
        self,
        decoded_source: str,
        module_name: str,
        descriptor_class_name: str,
        task_class_name: str,
        is_default: bool,
    ) -> None:
        module = write_to_file_and_import(module_name=module_name, decoded_source=decoded_source)

        # Validate descriptor exists and type
        if not hasattr(module, descriptor_class_name):
            raise ValueError(f"Descriptor class {descriptor_class_name} not found in its source code")
        descriptor_cls = getattr(module, descriptor_class_name)
        if not issubclass(descriptor_cls, TaskExecutionDescriptor):
            raise ValueError(f"Class {descriptor_class_name} is not a TaskExecutionDescriptor subclass")

        # Get task class from registry and register
        task_cls, _, _ = TASK_CLASS_REGISTRY.get_unit_task(task_class_name)
        self._register(task_cls, descriptor_cls, descriptor_class_name, default=is_default)

    def _register(
        self,
        task: type[UnitTask],
        descriptor: type[TaskExecutionDescriptor],
        name: str | None = None,
        default: bool = False,
    ) -> None:
        """Register a descriptor class for a task, optionally as default."""
        if name is None:
            name = descriptor.__name__
        task_name = task.__name__
        if name in self.registry[task_name]:
            raise ValueError(f"Descriptor {name} already registered for task {task_name}")
        self.registry[task_name][name] = descriptor
        if default:
            if task_name in self.default_registry:
                raise ValueError(f"Default descriptor already registered for task {task_name}")
            self.default_registry[task_name] = descriptor
        logger.info(
            "Registered execution descriptor: %s for task: %s%s", name, task_name, " (default)" if default else ""
        )

    def load_from_source(
        self, source_code: str, descriptor_class_name: str, module_name: str, task_class_name: str
    ) -> None:
        """Load a descriptor class from base64-encoded source.

        Because descriptor code imports task classes, we need to ensure task classes registered before
        their corresponding descriptors are loaded. So, if required task class presents, register immediately;
        otherwise put to the pending queue.
        """
        decoded_source = base64.b64decode(source_code).decode("utf-8")

        # If the target task is not yet registered, defer execution entirely.
        # We queue the raw decoded source and module metadata to be executed later.
        from cornserve.services.task_registry.task_class_registry import TASK_CLASS_REGISTRY  # noqa: PLC0415

        if task_class_name not in TASK_CLASS_REGISTRY:
            self._pending[task_class_name].append((decoded_source, module_name, descriptor_class_name, True))
            logger.info(
                "Queued execution descriptor %s for task %s until task class is loaded",
                descriptor_class_name,
                task_class_name,
            )
            return

        # Task present: execute and register via shared helper
        self._install_and_register(
            decoded_source=decoded_source,
            module_name=module_name,
            descriptor_class_name=descriptor_class_name,
            task_class_name=task_class_name,
            is_default=True,
        )

    def bind_pending_descriptor_for_task_class(self, task: type[UnitTask]) -> None:
        """Bind any queued descriptors to the now-available task class.

        This is called by the task class registry, whenever there's new task classes comming in.
        """
        task_name = task.__name__
        if task_name not in self._pending:
            return
        pending_items = self._pending.pop(task_name)
        for decoded_source, module_name, descriptor_class_name, is_default in pending_items:
            self._install_and_register(
                decoded_source=decoded_source,
                module_name=module_name,
                descriptor_class_name=descriptor_class_name,
                task_class_name=task_name,
                is_default=is_default,
            )
            logger.info("Registered pending execution descriptor: %s for task: %s", descriptor_class_name, task_name)

    def get(self, task: type[UnitTask], name: str | None = None) -> type[TaskExecutionDescriptor]:
        """Return a registered descriptor class for the given task.

        If `name` is None, return the default descriptor for the task.
        """
        task_name = task.__name__
        if task_name not in self.registry:
            available_tasks = list(self.registry.keys())
            raise ValueError(
                f"No descriptors registered for task {task_name}. Available descriptor tasks: {available_tasks}."
            )
        if name is None:
            if task_name not in self.default_registry:
                available_descriptors = list(self.registry[task_name].keys())
                raise ValueError(
                    f"No default descriptor registered for task {task_name}. "
                    f"Available descriptors: {available_descriptors}."
                )
            return self.default_registry[task_name]
        if name not in self.registry[task_name]:
            available_descriptors = list(self.registry[task_name].keys())
            raise ValueError(
                f"Descriptor {name} not registered for task {task_name}. "
                f"Available descriptors: {available_descriptors}."
            )
        return self.registry[task_name][name]

    def list_registered_descriptors(self) -> dict[str, list[str]]:
        """List registered descriptors per task name."""
        return {task_name: list(descriptors.keys()) for task_name, descriptors in self.registry.items()}

    def clear(self) -> None:
        """Clear all registrations and defaults."""
        self.registry.clear()
        self.default_registry.clear()


DESCRIPTOR_REGISTRY = TaskExecutionDescriptorRegistry()
