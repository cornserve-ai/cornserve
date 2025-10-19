"""Utility functions for task registry operations."""

import sys
import types
from importlib.machinery import ModuleSpec


def create_package_hierarchy_if_missing(name: str) -> None:
    """Create placeholder package modules in sys.modules if they don't exist.

    This ensures the full package hierarchy exists for a given module name,
    which is required for Python's import system to work properly.

    Args:
        name: The package name to ensure exists (e.g., 'cornserve_tasklib.task.unit')
    """
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__spec__ = ModuleSpec(name, loader=None, is_package=True)
    pkg.__path__ = []
    parent = name.rpartition(".")[0]
    if parent:
        create_package_hierarchy_if_missing(parent)
        setattr(sys.modules[parent], name.split(".")[-1], pkg)
    sys.modules[name] = pkg
