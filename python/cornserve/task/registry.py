"""Tasks registered and known to the system."""

from __future__ import annotations

import base64
import importlib.util
import sys
import types
from typing import TYPE_CHECKING

from cornserve.logging import get_logger

if TYPE_CHECKING:
    from cornserve.task.base import TaskInput, TaskOutput, UnitTask

logger = get_logger(__name__)


class TaskRegistry:
    """Registry of unit tasks.

    Composite tasks are not registered here; they can simply be invoked
    and will be decomposed into a series of unit task invocations.
    
    All tasks are loaded dynamically from Custom Resource (CR) instances.
    """

    def __init__(self) -> None:
        """Initialize the task registry."""
        self._tasks: dict[str, tuple[type[UnitTask], type[TaskInput], type[TaskOutput]]] = {}

    def register(
        self,
        task: type[UnitTask],
        task_input: type[TaskInput],
        task_output: type[TaskOutput],
        name: str | None = None,
    ) -> None:
        """Register a task with the given name."""
        name = name or task.__name__
        if name in self._tasks:
            logger.warning("Unit task with name=%s already exists, overriding", name)
        self._tasks[name] = (task, task_input, task_output)
        logger.info("Registered unit task: %s", name)

    def load_from_cr(self, source_code: str, task_class_name: str, module_name: str) -> None:
        """Load and register a unit task from CR source code.
        
        Args:
            source_code: Base64 encoded or plain Python source code
            task_class_name: Name of the main UnitTask class to register
            module_name: Module name to use (from CR spec.moduleName)
        """
        try:
            # Decode source code if it's base64 encoded
            try:
                decoded_source = base64.b64decode(source_code).decode('utf-8')
                logger.debug("Decoded base64 source code")
            except Exception:
                # Assume it's already plain text
                decoded_source = source_code
                logger.debug("Using plain text source code")
            
            logger.info("Loading unit task %s from CR into module %s", task_class_name, module_name)
            
            # Create module and add to sys.modules
            spec = importlib.util.spec_from_loader(module_name, loader=None)
            if spec is None:
                raise ImportError(f"Failed to create spec for module {module_name}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            
            # Make essential imports available in the module namespace
            from cornserve.task.base import UnitTask, TaskInput, TaskOutput, Stream, Task  # Import here to avoid circular imports
            from cornserve.task.forward import DataForward, Tensor
            
            # Add essential imports to module namespace
            module.UnitTask = UnitTask
            module.TaskInput = TaskInput
            module.TaskOutput = TaskOutput
            module.Stream = Stream
            module.Task = Task
            module.DataForward = DataForward
            module.Tensor = Tensor
            
            # Add typing helpers that tasks might need
            from typing import List, Set
            module.List = List
            module.list = list
            module.Set = Set
            module.set = set
            module.str = str
            module.int = int
            
            # Add additional imports that task files commonly need
            import enum
            module.enum = enum
            
            # Try to add pydantic imports that tasks use
            try:
                from pydantic import field_validator
                module.field_validator = field_validator
            except ImportError:
                # Create a mock field_validator if pydantic is not available
                def mock_field_validator(*args, **kwargs):
                    def decorator(func):
                        return func
                    return decorator
                module.field_validator = mock_field_validator
            
            # Execute the source code in the module namespace
            exec(decoded_source, module.__dict__)
            
            # Find the task class in the module
            if not hasattr(module, task_class_name):
                raise ValueError(f"Task class {task_class_name} not found in source code")
            
            task_cls = getattr(module, task_class_name)
            
            # Verify it's actually a UnitTask
            if not issubclass(task_cls, UnitTask):
                raise ValueError(f"Class {task_class_name} is not a UnitTask subclass")
            
            # Get the input/output types from the task class
            # We look for the parameterized UnitTask class in the MRO which has
            # working Pydantic generic metadata, then fall back to __orig_bases__
            task_input_cls = None
            task_output_cls = None
            
            # Look for the parameterized UnitTask class in the MRO
            # The MRO contains a concrete UnitTask[InputType, OutputType] class
            # that has working Pydantic generic metadata
            for base in task_cls.__mro__:
                if (hasattr(base, '__name__') and 'UnitTask[' in str(base) and 
                    hasattr(base, '__pydantic_generic_metadata__')):
                    
                    metadata = base.__pydantic_generic_metadata__
                    if metadata and 'args' in metadata:
                        args = metadata['args']
                        if len(args) == 2:
                            task_input_cls, task_output_cls = args
                            break
            
            # Fallback: Look for generic type information in __orig_bases__
            if task_input_cls is None or task_output_cls is None:
                if hasattr(task_cls, '__orig_bases__'):
                    for base in task_cls.__orig_bases__:
                        if hasattr(base, '__origin__') and hasattr(base, '__args__'):
                            # Check if this base is a generic UnitTask (e.g., UnitTask[InputType, OutputType])
                            if base.__origin__ is UnitTask or (hasattr(base.__origin__, '__mro__') and UnitTask in base.__origin__.__mro__):
                                args = base.__args__
                                if len(args) == 2:
                                    task_input_cls, task_output_cls = args
                                    break
            
            if task_input_cls is None or task_output_cls is None:
                raise ValueError(f"Task class {task_class_name} missing generic type arguments. "
                               f"Expected format: class {task_class_name}(UnitTask[InputType, OutputType])")
            
            # Register the task
            self.register(task_cls, task_input_cls, task_output_cls, task_class_name)
            
        except Exception as e:
            logger.error("Failed to load unit task %s from CR: %s", task_class_name, e)
            raise RuntimeError(f"Failed to load unit task {task_class_name} from CR: {e}") from e

    def get(self, name: str) -> tuple[type[UnitTask], type[TaskInput], type[TaskOutput]]:
        """Get a task by its name.

        Tasks must be loaded from Custom Resources (CRs) before they can be retrieved.
        """
        if name not in self._tasks:
            raise KeyError(f"Unit task with name={name} not found. "
                          f"Available tasks: {self.list_registered_tasks()}. "
                          f"Tasks must be loaded from CRs first.")
        return self._tasks[name]

    def __contains__(self, name: str) -> bool:
        """Check if a task is registered."""
        return name in self._tasks
    
    def list_registered_tasks(self) -> list[str]:
        """List all registered task names."""
        return list(self._tasks.keys())
    
    def clear(self) -> None:
        """Clear all registered tasks. Useful for testing."""
        self._tasks.clear()


TASK_REGISTRY = TaskRegistry()
