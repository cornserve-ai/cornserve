"""Utilities to discover tasks and descriptors in cornserve_tasklib.

This module scans the installed `cornserve_tasklib` package to produce
deployment-ready entries for unit tasks, composite tasks, and task execution
descriptors. The CLI consumes these entries to create requests to the gateway.
"""

from __future__ import annotations

from typing import Any, Tuple, get_args, get_origin

import base64
import importlib
import inspect
import pkgutil


def _camel_to_kebab(name: str) -> str:
    out: list[str] = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and (
            not name[i - 1].isupper() or (i + 1 < len(name) and not name[i + 1].isupper())
        ):
            out.append("-")
        out.append(ch.lower())
    return "".join(out)


def discover_tasklib() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Discover unit/composite tasks and descriptors from cornserve_tasklib.

    Returns:
        A tuple of (unit_task_entries, composite_task_entries, descriptor_entries),
        where each item is a list of dictionaries ready to be serialized into the
        gateway deployment request payloads.
    """
    try:
        import cornserve_tasklib  # noqa: F401  # ensure package importable
        from cornserve.task.base import Task, UnitTask
        from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor
    except Exception as e:  # pragma: no cover - bubbled to CLI
        raise ImportError(f"Failed to import cornserve_tasklib or cornserve core modules: {e}") from e

    unit_task_entries: list[dict[str, Any]] = []
    composite_task_entries: list[dict[str, Any]] = []
    descriptor_entries: list[dict[str, Any]] = []

    module_source_cache: dict[str, str] = {}

    def get_module_source_b64(module) -> str:
        if module.__name__ not in module_source_cache:
            src = inspect.getsource(module)
            module_source_cache[module.__name__] = base64.b64encode(src.encode("utf-8")).decode("ascii")
        return module_source_cache[module.__name__]

    # Discover tasks (unit and composite)
    task_pkg = importlib.import_module("cornserve_tasklib.task")
    for modinfo in pkgutil.walk_packages(task_pkg.__path__, prefix=task_pkg.__name__ + "."):
        module = importlib.import_module(modinfo.name)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            if issubclass(obj, UnitTask) and obj is not UnitTask:
                unit_task_entries.append(
                    {
                        "source_b64": get_module_source_b64(module),
                        "task_class_name": obj.__name__,
                        "task_definition_name": _camel_to_kebab(obj.__name__),
                        "module_name": module.__name__,
                        "is_unit_task": True,
                    }
                )
            elif issubclass(obj, Task) and (not issubclass(obj, UnitTask)) and obj is not Task:
                composite_task_entries.append(
                    {
                        "source_b64": get_module_source_b64(module),
                        "task_class_name": obj.__name__,
                        "task_definition_name": _camel_to_kebab(obj.__name__),
                        "module_name": module.__name__,
                        "is_unit_task": False,
                    }
                )

    # Discover descriptors
    desc_pkg = importlib.import_module("cornserve_tasklib.task_executors.descriptor")
    for modinfo in pkgutil.walk_packages(desc_pkg.__path__, prefix=desc_pkg.__name__ + "."):
        module = importlib.import_module(modinfo.name)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            if issubclass(obj, TaskExecutionDescriptor) and obj is not TaskExecutionDescriptor:
                task_cls_name = None
                # Preferred: infer from Pydantic model field annotation (robust under generics)
                try:
                    task_field = getattr(obj, "model_fields", {}).get("task")  # type: ignore[attr-defined]
                    if task_field is not None and getattr(task_field, "annotation", None) is not None:
                        task_cls = task_field.annotation
                        task_cls_name = getattr(task_cls, "__name__", None)
                except Exception:
                    pass
                # Fallback: try to infer from generic base declaration
                if task_cls_name is None:
                    for base in getattr(obj, "__orig_bases__", []):
                        origin = get_origin(base)
                        if origin is TaskExecutionDescriptor:
                            args = get_args(base)
                            if args:
                                task_cls = args[0]
                                task_cls_name = getattr(task_cls, "__name__", None)
                                break
                if task_cls_name is None:
                    continue
                descriptor_entries.append(
                    {
                        "source_b64": get_module_source_b64(module),
                        "descriptor_class_name": obj.__name__,
                        "descriptor_definition_name": _camel_to_kebab(obj.__name__),
                        "module_name": module.__name__,
                        "task_class_name": task_cls_name,
                    }
                )

    return unit_task_entries, composite_task_entries, descriptor_entries
