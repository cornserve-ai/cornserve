from __future__ import annotations

import base64
import importlib.util
import sys
from typing import TYPE_CHECKING

from cornserve.logging import get_logger

if TYPE_CHECKING:
    from cornserve.task.base import TaskInput, TaskOutput, UnitTask

logger = get_logger(__name__)


class TaskClassRegistry:
    """Registry for dynamically loaded UnitTask classes.

    Stores mapping: task-name -> (task class, input model, output model).
    """

    def __init__(self) -> None:
        self._tasks: dict[str, tuple[type[UnitTask], type[TaskInput], type[TaskOutput]]] = {}

    def register(
        self,
        task: type[UnitTask],
        task_input: type[TaskInput],
        task_output: type[TaskOutput],
        name: str | None = None,
    ) -> None:
        """Register a task class and its IO models under an optional name."""
        name = name or task.__name__
        self._tasks[name] = (task, task_input, task_output)

    def load_from_source(self, source_code: str, task_class_name: str, module_name: str) -> None:
        """Load a UnitTask class from base64-encoded source and register it.

        - Creates a module named module_name and execs decoded source into it
        - Extracts generic type args for UnitTask[InputT, OutputT] via MRO or __orig_bases__
        """
        decoded_source = base64.b64decode(source_code).decode("utf-8")

        # Create the module and place it in sys.modules so future imports work
        spec = importlib.util.spec_from_loader(module_name, loader=None)
        if spec is None:
            raise ImportError(f"Failed to create spec for module {module_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        # Provide symbols tasks expect at exec time
        from cornserve.task.base import UnitTask, TaskInput, TaskOutput, Stream, Task
        from cornserve.task.forward import DataForward, Tensor
        module.UnitTask = UnitTask
        module.TaskInput = TaskInput
        module.TaskOutput = TaskOutput
        module.Stream = Stream
        module.Task = Task
        module.DataForward = DataForward
        module.Tensor = Tensor
        import enum
        module.enum = enum

        # Execute the decoded source in the module namespace
        exec(decoded_source, module.__dict__)

        # Validate the class exists and is a UnitTask subclass
        if not hasattr(module, task_class_name):
            raise ValueError(f"Task class {task_class_name} not found in source code")
        task_cls = getattr(module, task_class_name)
        if not issubclass(task_cls, UnitTask):
            raise ValueError(f"Class {task_class_name} is not a UnitTask subclass")

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

        # Fallback: inspect __orig_bases__
        if task_input_cls is None or task_output_cls is None:
            if hasattr(task_cls, "__orig_bases__"):
                for base in task_cls.__orig_bases__:
                    if hasattr(base, "__origin__") and hasattr(base, "__args__"):
                        from cornserve.task.base import UnitTask as UnitTaskBase
                        if base.__origin__ is UnitTaskBase or (
                            hasattr(base.__origin__, "__mro__") and UnitTaskBase in base.__origin__.__mro__
                        ):
                            args = base.__args__
                            if len(args) == 2:
                                task_input_cls, task_output_cls = args
                                break

        if task_input_cls is None or task_output_cls is None:
            raise ValueError(
                f"Task class {task_class_name} missing generic type arguments. Expected format: class {task_class_name}(UnitTask[InputType, OutputType])"
            )

        self.register(task_cls, task_input_cls, task_output_cls, task_class_name)

    def get(self, name: str) -> tuple[type[UnitTask], type[TaskInput], type[TaskOutput]]:
        """Return (task class, input model, output model) by registered name."""
        if name not in self._tasks:
            raise KeyError(f"Unit task with name={name} not found. Available tasks: {self.list_registered_tasks()}")
        return self._tasks[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tasks

    def list_registered_tasks(self) -> list[str]:
        return list(self._tasks.keys())

    def clear(self) -> None:
        self._tasks.clear()


TASK_CLASS_REGISTRY = TaskClassRegistry()


