from typing import overload

import zmq
import zmq.asyncio

from cornserve.logging import get_logger

logger = get_logger(__name__)

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
