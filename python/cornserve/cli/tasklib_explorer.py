"""Utilities to discover tasks and descriptors in cornserve_tasklib.

This module scans the installed `cornserve_tasklib` package to produce
deployment-ready entries for unit tasks, composite tasks, and task execution
descriptors. The CLI consumes these entries to create requests to the gateway.
"""

from __future__ import annotations

import base64
import importlib
import inspect
import pkgutil
from typing import ForwardRef, get_args, get_origin

from cornserve.services.gateway.models import (
    DescriptorDefinitionPayload,
    TaskDefinitionPayload,
)
from cornserve.task.base import Task, UnitTask
from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor


def _camel_to_kebab(name: str) -> str:
    out: list[str] = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0 and (not name[i - 1].isupper() or (i + 1 < len(name) and not name[i + 1].isupper())):
            out.append("-")
        out.append(ch.lower())
    return "".join(out)


def _sanitize_k8s_name(raw: str) -> str:
    """Sanitize an string into a valid K8s metadata.name.

    - Only lowercase alphanumerics, '-' or '.'
    - Start and end with an alphanumeric
    - Other characters converted to '-'
    """
    sanitized: list[str] = []
    for ch in raw.lower():
        if ("a" <= ch <= "z") or ("0" <= ch <= "9") or ch in {"-", "."}:
            sanitized.append(ch)
        else:
            sanitized.append("-")
    name = "".join(sanitized)
    while "--" in name:
        name = name.replace("--", "-")
    name = name.strip("-.")
    return name


def discover_tasklib() -> tuple[
    list[TaskDefinitionPayload],
    list[TaskDefinitionPayload],
    list[DescriptorDefinitionPayload],
]:
    """Discover unit/composite tasks and descriptors from cornserve_tasklib.

    Returns:
        A tuple of (unit_task_entries, composite_task_entries, descriptor_entries),
        where each item is a list of dictionaries ready to be serialized into the
        gateway deployment request payloads.
    """
    try:
        import cornserve_tasklib  # noqa: F401, PLC0415  # the tasklib is an external package, so we import it carefully
    except Exception as e:  # pragma: no cover - bubbled to CLI
        raise ImportError(f"Failed to import cornserve_tasklib: {e}") from e

    unit_task_entries: list[TaskDefinitionPayload] = []
    composite_task_entries: list[TaskDefinitionPayload] = []
    descriptor_entries: list[DescriptorDefinitionPayload] = []

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
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            if issubclass(obj, UnitTask) and obj is not UnitTask:
                unit_task_entries.append(
                    TaskDefinitionPayload(
                        source_b64=get_module_source_b64(module),
                        task_class_name=obj.__name__,
                        task_definition_name=_sanitize_k8s_name(_camel_to_kebab(obj.__name__)),
                        module_name=module.__name__,
                        is_unit_task=True,
                    )
                )
            elif issubclass(obj, Task) and (not issubclass(obj, UnitTask)) and obj is not Task:
                composite_task_entries.append(
                    TaskDefinitionPayload(
                        source_b64=get_module_source_b64(module),
                        task_class_name=obj.__name__,
                        task_definition_name=_sanitize_k8s_name(_camel_to_kebab(obj.__name__)),
                        module_name=module.__name__,
                        is_unit_task=False,
                    )
                )

    # Discover descriptors
    desc_pkg = importlib.import_module("cornserve_tasklib.task_executors.descriptor")
    for modinfo in pkgutil.walk_packages(desc_pkg.__path__, prefix=desc_pkg.__name__ + "."):
        module = importlib.import_module(modinfo.name)
        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue
            if issubclass(obj, TaskExecutionDescriptor) and obj is not TaskExecutionDescriptor:
                """
                Why two resolution paths?
                1) Postponed annotations and Pydantic generics
                   - With "from __future__ import annotations", annotations are strings at class creation.
                     So the generic arg in TaskExecutionDescriptor[EncoderTask, ...] may be a string/ForwardRef.
                   - Pydantic later resolves those to concrete classes and stores them on
                     __pydantic_generic_metadata__. That metadata can be attached to the class OR to a
                     Pydantic-generated proxy base, so we walk the MRO to find it.
                2) __orig_bases__ (original Python generic bases)
                   - __orig_bases__ records the generic expression exactly as authored. Depending on import
                     order and subclassing, this can be the only place where a concrete class appears (or where
                     a ForwardRef remains when Pydantic metadata isn’t populated yet for the current class).
                   - For those cases, we require __orig_bases__ to already contain a real class object.
                If neither yields a real class, we consider the descriptor invalid and raise.
                """
                task_cls_name = None
                # Path 1: Pydantic generic metadata across MRO
                for base in getattr(obj, "__mro__", (obj,)):
                    meta = getattr(base, "__pydantic_generic_metadata__", None)
                    if not (meta and isinstance(meta, dict) and "args" in meta and meta["args"]):
                        continue
                    task_arg = meta["args"][0]
                    if isinstance(task_arg, str):
                        resolved = getattr(module, task_arg, None)
                        if not isinstance(resolved, type) and "." in task_arg:
                            mod_path, _, cls_name = task_arg.rpartition(".")
                            try:
                                mod = importlib.import_module(mod_path)
                                resolved = getattr(mod, cls_name, None)
                            except Exception:
                                resolved = None
                        if isinstance(resolved, type) and getattr(resolved, "__name__", None):
                            task_cls_name = resolved.__name__
                            break
                    elif isinstance(task_arg, ForwardRef):
                        ref = task_arg.__forward_arg__
                        resolved = getattr(module, ref, None)
                        if not isinstance(resolved, type) and "." in ref:
                            mod_path, _, cls_name = ref.rpartition(".")
                            try:
                                mod = importlib.import_module(mod_path)
                                resolved = getattr(mod, cls_name, None)
                            except Exception:
                                resolved = None
                        if isinstance(resolved, type) and getattr(resolved, "__name__", None):
                            task_cls_name = resolved.__name__
                            break
                    elif isinstance(task_arg, type) and getattr(task_arg, "__name__", None):
                        task_cls_name = task_arg.__name__
                        break
                # Path 2: __orig_bases__ concrete class
                if task_cls_name is None:
                    for base in getattr(obj, "__orig_bases__", []):
                        origin = get_origin(base)
                        if origin is TaskExecutionDescriptor:
                            args = get_args(base)
                            if not args:
                                break
                            task_arg = args[0]
                            if isinstance(task_arg, type) and getattr(task_arg, "__name__", None):
                                task_cls_name = task_arg.__name__
                                break
                if task_cls_name is None:
                    raise ValueError(
                        f"Descriptor {obj.__name__} must derive from "
                        "TaskExecutionDescriptor[<UnitTask>, ...] with concrete types"
                    )
                descriptor_entries.append(
                    DescriptorDefinitionPayload(
                        source_b64=get_module_source_b64(module),
                        descriptor_class_name=obj.__name__,
                        descriptor_definition_name=_sanitize_k8s_name(_camel_to_kebab(obj.__name__)),
                        module_name=module.__name__,
                        task_class_name=task_cls_name,
                    )
                )

    return unit_task_entries, composite_task_entries, descriptor_entries
