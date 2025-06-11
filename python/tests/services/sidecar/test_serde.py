import ctypes
import hashlib
import multiprocessing
import random
from multiprocessing.process import BaseProcess
from typing import Generator

import pytest
import torch
import zmq

from cornserve.sidecar.serde import MsgpackDecoder, MsgpackEncoder

random.seed(0)

NUM_GPUS = 4
IPC_PATH = "ipc:///tmp/test_serde"


def tensor_hash(tensor: torch.Tensor) -> str:
    """
    Generate a hash of a CUDA tensor's content.

    Args:
        tensor: Input tensor (should be on CUDA device)

    Returns:
        Hexadecimal hash string of the tensor content
    """
    tensor_cpu = tensor.detach().cpu()
    data_ptr = tensor_cpu.data_ptr()
    nbytes = tensor_cpu.element_size() * tensor_cpu.numel()
    buffer = (ctypes.c_byte * nbytes).from_address(data_ptr)
    tensor_bytes = bytes(buffer)
    return hashlib.sha256(tensor_bytes).hexdigest()


@pytest.fixture(scope="module")
def server() -> Generator[BaseProcess, None, None]:
    """Fixture to start a ZeroMQ server for testing tensor serialization and deserialization."""
    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(target=run_server)
    process.start()
    yield process
    process.join(timeout=3)


def run_server() -> None:
    """A server that listens for tensor data, deserializes it, and sends back a hash of the tensor."""
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.bind(IPC_PATH)
    while True:
        i, data = socket.recv_pyobj()
        if i < 0:
            break
        print(f"Received data {data} for GPU {i}")
        torch.cuda.set_device(i)
        tensor = MsgpackDecoder().decode(data)
        hash = tensor_hash(tensor)
        print(f"Received tensor on GPU {i}: {tensor}, hash: {hash}")
        socket.send_pyobj((i, hash))
    socket.close()
    context.term()


def test_serde(server: BaseProcess) -> None:
    """Test the Msgpack serialization and deserialization of tensors share IPC handles."""
    assert server.is_alive(), "Server process is not alive"
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.connect(IPC_PATH)
    for i in range(NUM_GPUS):
        torch.manual_seed(i)
        tensor = torch.randn(2, 3, 4, device=f"cuda:{i}", dtype=torch.bfloat16)
        data = MsgpackEncoder().encode(tensor)
        socket.send_pyobj((i, data))
        hash = tensor_hash(tensor)
        n, hash_received = socket.recv_pyobj()
        assert i == int(n)
        assert hash_received == hash, f"Hash mismatch for GPU {i}"
    socket.send_pyobj((-1, None))  # Signal to stop the server
    socket.close()
    context.term()
