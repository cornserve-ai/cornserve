from __future__ import annotations

import base64
import importlib.util
import sys
import types
from importlib.machinery import ModuleSpec
from collections import defaultdict
from typing import TYPE_CHECKING

from cornserve.logging import get_logger
from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor
from cornserve.services.task_registry.task_class_registry import TASK_CLASS_REGISTRY

if TYPE_CHECKING:
    from cornserve.task.base import UnitTask

logger = get_logger(__name__)


class TaskExecutionDescriptorRegistry:
    """Registry for dynamically loaded TaskExecutionDescriptor classes per UnitTask."""

    def __init__(self) -> None:
        self.registry: dict[str, dict[str, type[TaskExecutionDescriptor]]] = defaultdict(dict)
        self.default_registry: dict[str, type[TaskExecutionDescriptor]] = {}
        # Descriptors that arrive before their target task class is loaded
        self._pending: dict[str, list[tuple[type[TaskExecutionDescriptor], str, bool]]] = defaultdict(list)

    def register(
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

    def load_from_source(self, source_code: str, descriptor_class_name: str, module_name: str, task_class_name: str) -> None:
        """Load a descriptor class from base64-encoded source.

        Because descriptor code imports task classes, we need to ensure task classes registered before
        their corresponding descriptors are loaded. So, if required task class presents, register immediately;
        otherwise put to the pending queue.
        """
        decoded_source = base64.b64decode(source_code).decode("utf-8")

        created_packages: list[str] = []

        def ensure_package(name: str) -> None:
            if name in sys.modules:
                return
            pkg = types.ModuleType(name)
            pkg.__spec__ = ModuleSpec(name, loader=None, is_package=True)
            pkg.__path__ = []
            parent = name.rpartition('.')[0]
            if parent:
                ensure_package(parent)
                setattr(sys.modules[parent], name.split('.')[-1], pkg)
            sys.modules[name] = pkg
            created_packages.append(name)

        # Determine parent package name (do not create yet)
        parent = module_name.rpartition('.')[0]

        # Prepare the module (do not insert yet)
        module = types.ModuleType(module_name)
        module.__spec__ = ModuleSpec(module_name, loader=None, is_package=False)
        module.__package__ = parent or module_name
        module.__file__ = f"<crd:{module_name}>"

        try:
            # Execute source
            exec(decoded_source, module.__dict__)

            # Validate descriptor exists and is a subclass
            if not hasattr(module, descriptor_class_name):
                raise ValueError(f"Descriptor class {descriptor_class_name} not found in source code")
            descriptor_cls = getattr(module, descriptor_class_name)
            if not issubclass(descriptor_cls, TaskExecutionDescriptor):
                raise ValueError(f"Class {descriptor_class_name} is not a TaskExecutionDescriptor subclass")

            # Create parent packages only now, after validation succeeds
            if parent:
                ensure_package(parent)

            # Install after validation
            sys.modules[module_name] = module
            if parent:
                setattr(sys.modules[parent], module_name.split('.')[-1], module)

            # Register now or queue pending
            if task_class_name in TASK_CLASS_REGISTRY:
                task_cls, _, _ = TASK_CLASS_REGISTRY.get_unit_task(task_class_name)
                self.register(task_cls, descriptor_cls, descriptor_class_name, default=True)
            else:
                self._pending[task_class_name].append((descriptor_cls, descriptor_class_name, True))
                logger.info(
                    "Queued execution descriptor %s for task %s until task class is loaded",
                    descriptor_class_name,
                    task_class_name,
                )
        except Exception:
            # Roll back any packages we created
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
            raise

    def bind_pending_descriptor_for_task_class(self, task: type[UnitTask]) -> None:
        """Bind any queued descriptors to the now-available task class."""
        task_name = task.__name__
        if task_name not in self._pending:
            return
        pending = self._pending.pop(task_name)
        for descriptor_cls, name, is_default in pending:
            self.register(task, descriptor_cls, name, default=is_default)
            logger.info("Registered pending execution descriptor: %s for task: %s", name, task_name)

    def get(self, task: type[UnitTask], name: str | None = None) -> type[TaskExecutionDescriptor]:
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
                    f"No default descriptor registered for task {task_name}. Available descriptors: {available_descriptors}."
                )
            return self.default_registry[task_name]
        if name not in self.registry[task_name]:
            available_descriptors = list(self.registry[task_name].keys())
            raise ValueError(
                f"Descriptor {name} not registered for task {task_name}. Available descriptors: {available_descriptors}."
            )
        return self.registry[task_name][name]

    def list_registered_descriptors(self) -> dict[str, list[str]]:
        return {task_name: list(descriptors.keys()) for task_name, descriptors in self.registry.items()}

    def clear(self) -> None:
        self.registry.clear()
        self.default_registry.clear()


DESCRIPTOR_REGISTRY = TaskExecutionDescriptorRegistry()


