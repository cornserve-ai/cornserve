from __future__ import annotations

import base64
import importlib.util
import sys
import types
from importlib.machinery import ModuleSpec
from typing import TYPE_CHECKING

from cornserve.logging import get_logger
from cornserve.task.base import UnitTask, Task
from cornserve.services.task_registry.util import create_package_hierarchy_if_missing

if TYPE_CHECKING:
    from cornserve.task.base import TaskInput, TaskOutput

logger = get_logger(__name__)


class TaskClassRegistry:
    """Registry for dynamically loaded Task classes (both Unit and Composite).

    For unit tasks: store a mapping: task-class-name -> (task class, input model, output model).
    For composite tasks: store a mapping: task-class-name -> (task class, None, None).
    """

    def __init__(self) -> None:
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

    def load_from_source(self, source_code: str, task_class_name: str, module_name: str, is_unit_task: bool = True) -> None:
        """Load a Task class from base64-encoded source and register it.

        Args:
            source_code: Base64-encoded source code file
            task_class_name: Name of the task class to register
            module_name: Full module path of where the task class is defined
            is_unit_task: True for UnitTask, False otherwise
        """
        decoded_source = base64.b64decode(source_code).decode("utf-8")
        try:
            self._exec_install_and_register_task(
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
            if isinstance(missing, str) and (missing.startswith("cornserve_tasklib.") or missing.startswith("cornserve.")):
                self._pending.append((decoded_source, module_name, task_class_name, is_unit_task))
                logger.info(
                    "Queued task module %s for later due to missing dependency: %s",
                    module_name,
                    missing,
                )
                return
            raise

    def _exec_install_and_register_task(
        self,
        decoded_source: str,
        module_name: str,
        task_class_name: str,
        is_unit_task: bool,
    ) -> None:
        """Execute task module source, validate, install, and register (unit or composite)."""
        created_packages: list[str] = []

        # Ensure parent packages
        parent = module_name.rpartition('.')[0]
        if parent:
            create_package_hierarchy_if_missing(parent)
            # Track packages created for rollback
            if parent in sys.modules:
                created_packages.append(parent)

        # Create and pre-insert module
        module = types.ModuleType(module_name)
        module.__spec__ = ModuleSpec(module_name, loader=None, is_package=False)
        module.__package__ = parent or module_name
        module.__file__ = f"<crd:{module_name}>"

        preexisting = module_name in sys.modules
        previous_module = sys.modules.get(module_name)
        sys.modules[module_name] = module
        if parent:
            setattr(sys.modules[parent], module_name.split('.')[-1], module)

        try:
            # Execute the decoded source
            exec(decoded_source, module.__dict__)

            # Validate class presence and type
            if not hasattr(module, task_class_name):
                raise ValueError(f"Task class {task_class_name} not found in source code")
            task_cls = getattr(module, task_class_name)

            if is_unit_task:
                if not issubclass(task_cls, UnitTask):
                    raise ValueError(f"Class {task_class_name} is not a UnitTask subclass")
            else:
                if not issubclass(task_cls, Task):
                    raise ValueError(f"Class {task_class_name} is not a Task subclass")

            if not is_unit_task:
                self._register_composite(task_cls, task_class_name)
                return

            # Extract generic types from MRO
            task_input_cls = None
            task_output_cls = None
            for base in task_cls.__mro__:
                if (
                    hasattr(base, "__name__")
                    and "UnitTask[" in str(base)
                    and hasattr(base, "__pydantic_generic_metadata__")
                ):
                    metadata = base.__pydantic_generic_metadata__
                    if metadata and "args" in metadata and len(metadata["args"]) == 2:
                        task_input_cls, task_output_cls = metadata["args"]
                        break

            if task_input_cls is None or task_output_cls is None:
                # Special case:  the  "LLMBaseUnitTask" is a generic task class
                # so it doesn't have concrete input/output types,
                # so to let it goes through, we register it with dummy TaskInput/Output types
                # TODO: Such special case is definitely not legetimate, what should we do instead?
                if task_class_name == "LLMBaseUnitTask":
                    from cornserve.task.base import TaskInput as DummyInput, TaskOutput as DummyOutput
                    task_input_cls, task_output_cls = DummyInput, DummyOutput
                else:
                    raise ValueError(
                        f"Task class {task_class_name} missing generic type arguments. Expected format: class {task_class_name}(UnitTask[InputType, OutputType])"
                    )

            self._register(task_cls, task_input_cls, task_output_cls, task_class_name)
        except Exception:
            # Roll back package placeholders
            for pkg_name in reversed(created_packages):
                parent_name = pkg_name.rpartition('.')[0]
                child_name = pkg_name.split('.')[-1]
                if parent_name in sys.modules:
                    try:
                        if getattr(sys.modules[parent_name], child_name, None) is sys.modules.get(pkg_name):
                            delattr(sys.modules[parent_name], child_name)
                    except Exception:
                        pass
                sys.modules.pop(pkg_name, None)
            # Roll back pre-inserted module to restore sys.modules state
            try:
                if parent and getattr(sys.modules.get(parent, None), module_name.split('.')[-1], None) is module:
                    delattr(sys.modules[parent], module_name.split('.')[-1])
            except Exception:
                pass
            if preexisting:
                sys.modules[module_name] = previous_module
            else:
                sys.modules.pop(module_name, None)
            raise

    def _bind_pending_tasks(self) -> None:
        """Attempt to load any pending task modules deferred due to missing deps."""
        if not self._pending:
            return
        remaining: list[tuple[str, str, str, bool]] = []
        for decoded_source, module_name, task_class_name, is_unit_task in self._pending:
            try:
                self._exec_install_and_register_task(
                    decoded_source=decoded_source,
                    module_name=module_name,
                    task_class_name=task_class_name,
                    is_unit_task=is_unit_task,
                )
                logger.info("Registered pending task class: %s", task_class_name)
            except ModuleNotFoundError as e:
                missing = getattr(e, "name", "") or str(e)
                if isinstance(missing, str) and (missing.startswith("cornserve_tasklib.") or missing.startswith("cornserve.")):
                    remaining.append((decoded_source, module_name, task_class_name, is_unit_task))
                else:
                    logger.error("Failed to load pending task %s: %r", task_class_name, e)
            except Exception as e:
                logger.error("Failed to load pending task %s: %r", task_class_name, e)
        self._pending = remaining

    def get_unit_task(self, name: str) -> tuple[type[UnitTask], type[TaskInput], type[TaskOutput]]:
        """Return (task class, input model, output model) by registered name for unit tasks."""
        if name not in self._unit_tasks:
            raise KeyError(f"Unit task with name={name} not found. Available unit tasks: {self.list_registered_unit_tasks()}")
        return self._unit_tasks[name]

    def __contains__(self, name: str) -> bool:
        return name in self._unit_tasks or name in self._composite_tasks


    def clear(self) -> None:
        self._unit_tasks.clear()
        self._composite_tasks.clear()

    def list_registered_unit_tasks(self) -> list[str]:
        """List names of registered unit tasks."""
        return list(self._unit_tasks.keys())


TASK_CLASS_REGISTRY = TaskClassRegistry()


