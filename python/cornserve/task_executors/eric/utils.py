import os
import signal
import contextlib
import tempfile
from uuid import uuid4
from typing import overload

import zmq
import zmq.asyncio
import psutil

from cornserve.logging import get_logger

logger = get_logger(__name__)

TMP_DIR = tempfile.gettempdir()

@overload
def make_zmq_socket(
    ctx: zmq.asyncio.Context,
    path: str,
    sock_type: int,
) -> zmq.asyncio.Socket:
    ...

@overload
def make_zmq_socket(
    ctx: zmq.Context,
    path: str,
    sock_type: int,
) -> zmq.Socket:
    ...

def make_zmq_socket(
    ctx: zmq.Context | zmq.asyncio.Context,
    path: str,
    sock_type: int,
) -> zmq.Socket | zmq.asyncio.Socket:
    s = ctx.socket(sock_type)

    buf_size = int(0.5 * 1024**3)  # 500 MiB

    if sock_type == zmq.PULL:
        s.setsockopt(zmq.RCVHWM, 0)
        s.setsockopt(zmq.RCVBUF, buf_size)
        s.connect(path)
    elif sock_type == zmq.PUSH:
        s.setsockopt(zmq.SNDHWM, 0)
        s.setsockopt(zmq.SNDBUF, buf_size)
        s.bind(path)
    else:
        raise ValueError(f"Unsupported socket type: {sock_type}")

    return s


def get_open_zmq_ipc_path(description: str | None = None) -> str:
    """Get an open IPC path for ZMQ sockets.

    Args:
        description: An optional string description for where the socket is used.
    """
    filename = f"{description}-{uuid4()}" if description is not None else str(uuid4())
    return f"ipc://{TMP_DIR}/{filename}"


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
