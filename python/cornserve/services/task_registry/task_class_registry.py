"""Registry for task classes dynamically loaded from k8s CRs (unit and composite)."""

from __future__ import annotations

import base64
import inspect
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from cornserve.logging import get_logger
from cornserve.services.task_registry.utils import write_to_file_and_import
from cornserve.task.base import Task, UnitTask

if TYPE_CHECKING:
    from cornserve.task.base import TaskInput, TaskOutput

logger = get_logger(__name__)


class TaskClassRegistry:
    """Registry for dynamically loaded Task classes (both Unit and Composite).

    For unit tasks: maps name -> (task class, input model, output model).
    For composite tasks: maps name -> task class.
    """

    def __init__(self) -> None:
        """Initialize registries for unit and composite task classes."""
        self._unit_tasks: dict[str, tuple[type[UnitTask], type[TaskInput], type[TaskOutput]]] = {}
        self._composite_tasks: dict[str, type[Task]] = {}

        # Task modules that failed to load due to dependency/order issues
        # each item is (decoded_source, module_name, task_class_name, is_unit_task)
        self._pending: list[tuple[str, str, str, bool]] = []

    def _register(
        self,
        task: type[UnitTask],
        task_input: type[TaskInput],
        task_output: type[TaskOutput],
        name: str | None = None,
    ) -> None:
        """Register a unit task class and its IO models under an optional name."""
        name = name or task.__name__
        self._unit_tasks[name] = (task, task_input, task_output)
        logger.info("Registered unit task: %s", name)

    def _register_composite(
        self,
        task: type[Task],
        name: str | None = None,
    ) -> None:
        """Register a composite task class under an optional name."""
        name = name or task.__name__
        self._composite_tasks[name] = task
        logger.info("Registered composite task: %s", name)

    def load_from_source(
        self, source_code: str, task_class_name: str, module_name: str, is_unit_task: bool = True
    ) -> None:
        """Load a Task class from base64-encoded source and register it.

        Args:
            source_code: Base64-encoded source code file
            task_class_name: Name of the task class to register
            module_name: Full module path of where the task class is defined
            is_unit_task: True for UnitTask, False otherwise
        """
        decoded_source = base64.b64decode(source_code).decode("utf-8")
        try:
            self._install_and_register_task(
                decoded_source=decoded_source,
                module_name=module_name,
                task_class_name=task_class_name,
                is_unit_task=is_unit_task,
            )
            # After success, try to bind any pending tasks that might now resolve
            self._bind_pending_tasks()
        except ModuleNotFoundError as e:
            missing = getattr(e, "name", "") or str(e)
            # If dependency is another CR-backed module, queue for retry
            if isinstance(missing, str) and (
                missing.startswith("cornserve_tasklib.") or missing.startswith("cornserve.")
            ):
                self._pending.append((decoded_source, module_name, task_class_name, is_unit_task))
                logger.info(
                    "Queued task module %s for later due to missing dependency: %s",
                    module_name,
                    missing,
                )
                return
            raise

    def _install_and_register_task(
        self,
        decoded_source: str,
        module_name: str,
        task_class_name: str,
        is_unit_task: bool,
    ) -> None:
        module = write_to_file_and_import(module_name=module_name, decoded_source=decoded_source)

        # Validate class presence and type
        if not hasattr(module, task_class_name):
            raise ValueError(f"Task class {task_class_name} not found in its source code")
        task_cls = getattr(module, task_class_name)

        if is_unit_task:
            if not issubclass(task_cls, UnitTask):
                raise ValueError(f"Class {task_class_name} is not a UnitTask subclass")
        elif not issubclass(task_cls, Task):
            raise ValueError(f"Class {task_class_name} is not a Task subclass")

        if not is_unit_task:
            self._register_composite(task_cls, task_class_name)
            return

        # Extract generic types from MRO
        # Assert again for static type checkers
        assert issubclass(task_cls, UnitTask)
        task_input_cls = None
        task_output_cls = None
        for base in task_cls.__mro__:
            if hasattr(base, "__pydantic_generic_metadata__"):
                metadata = base.__pydantic_generic_metadata__
                if metadata and "args" in metadata:
                    args = metadata["args"]
                    # Generic classes have arguments of TypeVars.
                    # Concrete classes have arguments of BaseModel subclasses.
                    # We register both kinds.
                    if len(args) == 2 and (
                        (
                            inspect.isclass(args[0])
                            and inspect.isclass(args[1])
                            and issubclass(args[0], BaseModel)
                            and issubclass(args[1], BaseModel)
                        )
                        or (isinstance(args[0], TypeVar) and isinstance(args[1], TypeVar))
                    ):
                        task_input_cls, task_output_cls = args
                        break

        if task_input_cls is None or task_output_cls is None:
            # Some illegal classes
            logger.warning("Task class %s has no valid generic type arguments", task_class_name)
            return

        self._register(task_cls, task_input_cls, task_output_cls, task_class_name)

    def _bind_pending_tasks(self) -> None:
        """Attempt to load any pending task modules deferred due to missing deps."""
        if not self._pending:
            return
        remaining: list[tuple[str, str, str, bool]] = []
        for decoded_source, module_name, task_class_name, is_unit_task in self._pending:
            try:
                self._install_and_register_task(
                    decoded_source=decoded_source,
                    module_name=module_name,
                    task_class_name=task_class_name,
                    is_unit_task=is_unit_task,
                )
                logger.info("Registered pending task class: %s", task_class_name)
                # Also bind any pending descriptors for this newly registered task
                if is_unit_task:
                    from cornserve.services.task_registry.descriptor_registry import (  # noqa: PLC0415
                        DESCRIPTOR_REGISTRY,
                    )

                    task_cls, _, _ = self.get_unit_task(task_class_name)
                    DESCRIPTOR_REGISTRY.bind_pending_descriptor_for_task_class(task_cls)
            except ModuleNotFoundError as e:
                missing = getattr(e, "name", "") or str(e)
                if isinstance(missing, str) and (
                    missing.startswith("cornserve_tasklib.") or missing.startswith("cornserve.")
                ):
                    remaining.append((decoded_source, module_name, task_class_name, is_unit_task))
                else:
                    logger.error("Failed to load pending task %s: %r", task_class_name, e)
            except Exception as e:
                logger.error("Failed to load pending task %s: %r", task_class_name, e)
        self._pending = remaining

    def get_unit_task(self, name: str) -> tuple[type[UnitTask], type[TaskInput], type[TaskOutput]]:
        """Return (task class, input model, output model) by registered name for unit tasks."""
        if name not in self._unit_tasks:
            raise KeyError(
                f"Unit task with name={name} not found. Available unit tasks: {self.list_registered_unit_tasks()}"
            )
        return self._unit_tasks[name]

    def __contains__(self, name: str) -> bool:
        """Return True if a unit or composite task with given name exists."""
        return name in self._unit_tasks or name in self._composite_tasks

    def clear(self) -> None:
        """Clear all registered task classes."""
        self._unit_tasks.clear()
        self._composite_tasks.clear()
        self._pending.clear()

    def list_registered_unit_tasks(self) -> list[str]:
        """List names of registered unit tasks."""
        return list(self._unit_tasks.keys())


TASK_CLASS_REGISTRY = TaskClassRegistry()
