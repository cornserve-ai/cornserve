"""Utilities for creating and managing ZMQ sockets."""

import contextlib
import tempfile
from uuid import uuid4
from typing import overload, Iterator

import zmq
import zmq.asyncio

from cornserve.logging import get_logger

logger = get_logger(__name__)

TMP_DIR = tempfile.gettempdir()


@overload
def make_zmq_socket(
    ctx: zmq.asyncio.Context,
    path: str,
    sock_type: int,
) -> zmq.asyncio.Socket: ...


@overload
def make_zmq_socket(
    ctx: zmq.Context,
    path: str,
    sock_type: int,
) -> zmq.Socket: ...


def make_zmq_socket(
    ctx: zmq.Context | zmq.asyncio.Context,
    path: str,
    sock_type: int,
) -> zmq.Socket | zmq.asyncio.Socket:
    """Create a ZMQ socket.

    Args:
        ctx: The ZMQ context. Can be either a sync or async context.
        path: Socket path prefixed with protocol.
        sock_type: Socket type, like `zmq.PULL` or `zmq.PUSH`.
    """
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


@contextlib.contextmanager
def zmq_sync_socket(path: str, sock_type: int) -> Iterator[zmq.Socket]:
    """Context manager that creates and cleans up a ZMQ socket."""
    ctx = zmq.Context(io_threads=2)
    try:
        yield make_zmq_socket(ctx, path, sock_type)

    finally:
        ctx.destroy(linger=0)
