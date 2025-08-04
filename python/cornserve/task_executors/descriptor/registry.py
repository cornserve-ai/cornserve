"""Task execution descriptor registry.

Task execution descriptor classes register themselves to the registry
specifying a task they can execute. Exactly one descriptor class per
task is marked as the default descriptor class. The registry is used
to look up the descriptor class for a task when executing it.

`DESCRIPTOR_REGISTRY` is a singleton instance of the registry.
"""

from __future__ import annotations

import base64
import importlib.util
import sys
import types
from collections import defaultdict
from typing import TYPE_CHECKING

from cornserve.logging import get_logger

if TYPE_CHECKING:
    from cornserve.task.base import UnitTask
    from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor

logger = get_logger(__name__)

DEFAULT = "__default_descriptor__"


class TaskExecutionDescriptorRegistry:
    """Registry for task execution descriptors."""

    def __init__(self) -> None:
        """Initialize the registry."""
        # NOTE: Using type[UnitTask] as the key does not work as the reigstered keys,
        # for the same task, has different class instances with the get()'s arg
        self.registry: dict[str, dict[str, type[TaskExecutionDescriptor]]] = defaultdict(dict)
        self.default_registry: dict[str, type[TaskExecutionDescriptor]] = {}

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

        task_name = task.__name__
        
        if name in self.registry[task_name]:
            raise ValueError(f"Descriptor {name} already registered for task {task_name}")

        self.registry[task_name][name] = descriptor

        if default:
            if task_name in self.default_registry:
                raise ValueError(f"Default descriptor already registered for task {task_name}")
            self.default_registry[task_name] = descriptor

    def load_from_cr(self, source_code: str, descriptor_class_name: str, module_name: str, task_class_name: str) -> None:
        """Load and register a task execution descriptor from CR source code.
        
        Args:
            source_code: Base64 encoded or plain Python source code
            descriptor_class_name: Name of the main TaskExecutionDescriptor class to register
            module_name: Module name to use (from CR spec.moduleName)
            task_class_name: Name of the UnitTask class this descriptor is for
        """
        try:
            # Decode source code if it's base64 encoded
            try:
                decoded_source = base64.b64decode(source_code).decode('utf-8')
                logger.debug("Decoded base64 source code for descriptor")
            except Exception:
                # Assume it's already plain text
                decoded_source = source_code
                logger.debug("Using plain text source code for descriptor")
            
            logger.info("Loading descriptor %s from CR into module %s for task %s", 
                       descriptor_class_name, module_name, task_class_name)
            
            # Create module and add to sys.modules
            spec = importlib.util.spec_from_loader(module_name, loader=None)
            if spec is None:
                raise ImportError(f"Failed to create spec for module {module_name}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            
            # Make essential imports available in the module namespace
            from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor
            from cornserve.task_executors.descriptor.registry import DESCRIPTOR_REGISTRY
            from cornserve.services.resource_manager.resource import GPU
            from cornserve import constants
            
            # Add essential imports to module namespace
            module.TaskExecutionDescriptor = TaskExecutionDescriptor
            module.DESCRIPTOR_REGISTRY = DESCRIPTOR_REGISTRY
            module.GPU = GPU
            module.constants = constants
            
            # Add common HTTP client
            import httpx
            module.httpx = httpx
            
            # Add typing helpers that descriptors might need
            from typing import Any, List, Set
            module.Any = Any
            module.List = List
            module.list = list
            module.Set = Set
            module.set = set
            module.str = str
            module.int = int
            
            # Add additional imports that descriptor files commonly need
            import enum
            module.enum = enum
            
            # Execute the source code in the module namespace
            exec(decoded_source, module.__dict__)
            
            # Find the descriptor class in the module
            if not hasattr(module, descriptor_class_name):
                raise ValueError(f"Descriptor class {descriptor_class_name} not found in source code")
            
            descriptor_cls = getattr(module, descriptor_class_name)
            
            # Verify it's actually a TaskExecutionDescriptor
            if not issubclass(descriptor_cls, TaskExecutionDescriptor):
                raise ValueError(f"Class {descriptor_class_name} is not a TaskExecutionDescriptor subclass")
            
            # We need to find the UnitTask class that this descriptor is for
            # Import task registry to get the task class
            from cornserve.task.registry import TASK_REGISTRY
            
            if task_class_name not in TASK_REGISTRY:
                raise ValueError(f"Task class {task_class_name} not found in task registry. "
                               f"Tasks must be loaded before their descriptors.")
            
            task_cls, _, _ = TASK_REGISTRY.get(task_class_name)
            
            # Register the descriptor
            self.register(task_cls, descriptor_cls, descriptor_class_name, default=True)
            logger.info("Successfully loaded and registered descriptor: %s for task: %s", descriptor_class_name, task_class_name)
            
        except Exception as e:
            logger.error("Failed to load descriptor %s from CR: %s", descriptor_class_name, e)
            raise RuntimeError(f"Failed to load descriptor {descriptor_class_name} from CR: {e}") from e

    def get(self, task: type[UnitTask], name: str | None = None) -> type[TaskExecutionDescriptor]:
        """Get the task execution descriptor for a task.

        Descriptors must be loaded from Custom Resources (CRs) before they can be retrieved.

        Args:
            task: The task class to get the descriptor for.
            name: The name of the descriptor. If None, use the default descriptor.
        """
        task_name = task.__name__
        
        if task_name not in self.registry:
            available_tasks = list(self.registry.keys())
            raise ValueError(f"No descriptors registered for task {task_name}. "
                           f"Available descriptor tasks: {available_tasks}. "
                           f"Descriptors must be loaded from CRs first.")

        if name is None:
            if task_name not in self.default_registry:
                available_descriptors = list(self.registry[task_name].keys())
                raise ValueError(f"No default descriptor registered for task {task_name}. "
                               f"Available descriptors: {available_descriptors}. "
                               f"Descriptors must be loaded from CRs first.")
            return self.default_registry[task_name]

        if name not in self.registry[task_name]:
            available_descriptors = list(self.registry[task_name].keys())
            raise ValueError(f"Descriptor {name} not registered for task {task_name}. "
                           f"Available descriptors: {available_descriptors}. "
                           f"Descriptors must be loaded from CRs first.")

        return self.registry[task_name][name]
    
    def list_registered_descriptors(self) -> dict[str, list[str]]:
        """List all registered descriptors by task name."""
        return {task_name: list(descriptors.keys()) 
                for task_name, descriptors in self.registry.items()}
    
    def clear(self) -> None:
        """Clear all registered descriptors. Useful for testing."""
        self.registry.clear()
        self.default_registry.clear()


DESCRIPTOR_REGISTRY = TaskExecutionDescriptorRegistry()
