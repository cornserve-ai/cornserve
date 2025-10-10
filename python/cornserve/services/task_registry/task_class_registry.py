from __future__ import annotations

import base64
import importlib.util
import sys
import types
from importlib.machinery import ModuleSpec
from typing import TYPE_CHECKING

from cornserve.logging import get_logger
from cornserve.task.base import UnitTask, Task

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

    def register(
        self,
        task: type[UnitTask],
        task_input: type[TaskInput],
        task_output: type[TaskOutput],
        name: str | None = None,
    ) -> None:
        """Register a unit task class and its IO models under an optional name."""
        name = name or task.__name__
        self._unit_tasks[name] = (task, task_input, task_output)

    def register_composite(
        self,
        task: type[Task],
        name: str | None = None,
    ) -> None:
        """Register a composite task class under an optional name."""
        name = name or task.__name__
        self._composite_tasks[name] = task

    def load_from_source(self, source_code: str, task_class_name: str, module_name: str, is_unit_task: bool = True) -> None:
        """Load a Task class from base64-encoded source and register it.

        Args:
            source_code: Base64-encoded source code file
            task_class_name: Name of the task class to register
            module_name: Full module path of where the task class is defined
            is_unit_task: True for UnitTask, False otherwise
        """
        import sys
        import types
        from importlib.machinery import ModuleSpec
        decoded_source = base64.b64decode(source_code).decode("utf-8")

        created_packages: list[str] = []

        def ensure_package(name: str) -> None:
            """Create a package module if it doesn't exist, ensuring proper package metadata."""
            if name in sys.modules:
                return
            pkg = types.ModuleType(name)
            pkg.__spec__ = ModuleSpec(name, loader=None, is_package=True)
            pkg.__path__ = []  # mark as package
            parent = name.rpartition('.')[0]
            if parent:
                ensure_package(parent)
                # attach as attribute on parent for 'from parent import child'
                setattr(sys.modules[parent], name.split('.')[-1], pkg)
            sys.modules[name] = pkg
            created_packages.append(name)

        # Ensure parent packages are real packages with proper metadata
        parent = module_name.rpartition('.')[0]
        if parent:
            ensure_package(parent)

        # Create the target module with proper metadata (do NOT insert into sys.modules yet)
        module = types.ModuleType(module_name)
        module.__spec__ = ModuleSpec(module_name, loader=None, is_package=False)
        module.__package__ = parent or module_name
        module.__file__ = f"<crd:{module_name}>"

        try:
            # Execute the decoded source in the module namespace
            exec(decoded_source, module.__dict__)

            # Validate the class exists
            if not hasattr(module, task_class_name):
                raise ValueError(f"Task class {task_class_name} not found in source code")
            task_cls = getattr(module, task_class_name)
            
            # Import Task classes for validation
            if is_unit_task:
                if not issubclass(task_cls, UnitTask):
                    raise ValueError(f"Class {task_class_name} is not a UnitTask subclass")
            else:
                if not issubclass(task_cls, Task):
                    raise ValueError(f"Class {task_class_name} is not a Task subclass")

            # Now that validation passed, install module into sys.modules and attach to parent
            sys.modules[module_name] = module
            if parent:
                setattr(sys.modules[parent], module_name.split('.')[-1], module)

            if not is_unit_task:
                # For composite tasks, directly register and return
                self.register_composite(task_cls, task_class_name)
                return

            # Extract generic types from MRO first (preferred)
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
                raise ValueError(
                    f"Task class {task_class_name} missing generic type arguments. Expected format: class {task_class_name}(UnitTask[InputType, OutputType])"
                )

            self.register(task_cls, task_input_cls, task_output_cls, task_class_name)
        except Exception:
            # Roll back any packages we created
            for pkg_name in reversed(created_packages):
                # remove attribute from its parent if present
                parent_name = pkg_name.rpartition('.')[0]
                child_name = pkg_name.split('.')[-1]
                if parent_name in sys.modules:
                    try:
                        if getattr(sys.modules[parent_name], child_name, None) is sys.modules.get(pkg_name):
                            delattr(sys.modules[parent_name], child_name)
                    except Exception:
                        pass
                sys.modules.pop(pkg_name, None)
            raise

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


TASK_CLASS_REGISTRY = TaskClassRegistry()


