"""Utilities for process management."""

import os
import signal
import contextlib

import psutil


def kill_process_tree(pid: int | None) -> None:
    """Kill all descendant processes of the given pid by sending SIGKILL.

    Args:
        pid: Process ID of the parent process.
    """
    # None might be passed in if mp.Process hasn't been spawned yet
    if pid is None:
        return

    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    # Get all children recursively
    children = parent.children(recursive=True)

    # Send SIGKILL to all children first
    for child in children:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child.pid, signal.SIGKILL)

    # Finally kill the parent
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)
