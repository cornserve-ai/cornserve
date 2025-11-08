"""Utility functions for task registry operations."""

import sys
import types
from importlib import import_module, invalidate_caches
from pathlib import Path

from cornserve.constants import TASKLIB_DIR


def _ensure_tasklib_base_on_sys_path() -> None:
    """Ensure the base tasklib directory is present on sys.path."""
    if TASKLIB_DIR not in sys.path:
        sys.path.insert(0, TASKLIB_DIR)


def _ensure_fs_package_tree_for_module(module_name: str) -> Path:
    """Check and create the filesystem package tree for the module.

    Example:
        module_name = 'cornserve_tasklib.task.unit.my_task'
        returns: /tmp/cornserve_tasklib/cornserve_tasklib/task/unit/my_task.py
    """
    parts = module_name.split(".")
    # Parent directories represent packages
    package_dirs = Path(TASKLIB_DIR).joinpath(*parts[:-1])
    package_dirs.mkdir(parents=True, exist_ok=True)
    # Ensure __init__.py exists for each package directory to make them traditional packages
    cur = Path(TASKLIB_DIR)
    for part in parts[:-1]:
        cur = cur / part
        init_file = cur / "__init__.py"
        if not init_file.exists():
            init_file.write_text("")

    # Return full module file path
    return package_dirs / f"{parts[-1]}.py"


def write_to_file_and_import(module_name: str, decoded_source: str) -> types.ModuleType:
    """Save module source to the TASKLIB_DIR and import it."""
    _ensure_tasklib_base_on_sys_path()
    target_file = _ensure_fs_package_tree_for_module(module_name)

    bytes_to_write = decoded_source.encode("utf-8")

    same_content = False
    if target_file.exists():
        try:
            same_content = target_file.read_bytes() == bytes_to_write
        except Exception:
            same_content = False

    if not same_content:
        with open(target_file, "wb") as f:
            f.write(bytes_to_write)

    invalidate_caches()

    existing = sys.modules.get(module_name)
    if existing is not None:
        if same_content:
            # Here, we avoid reloading the module to avoid creating a new class identity for the same class.
            return existing
        # TODO: Updating existing modules is not supported yet.
        raise NotImplementedError("Updating existing modules is not supported yet.")

    # Not loaded: import normally
    return import_module(module_name)
