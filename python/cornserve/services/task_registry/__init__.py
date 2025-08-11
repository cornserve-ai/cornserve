"""Runtime task registry service entrypoint."""

from .manager import TaskRegistry
from .task_class_registry import TASK_CLASS_REGISTRY
from .descriptor_registry import DESCRIPTOR_REGISTRY

__all__ = ["TaskRegistry", "TASK_CLASS_REGISTRY", "DESCRIPTOR_REGISTRY"]


