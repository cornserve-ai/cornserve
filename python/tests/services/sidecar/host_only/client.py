#!/usr/bin/env python3
"""
Standalone client for tensor serialization testing.
This client sends tensors from a specific GPU and verifies responses.
"""

import ctypes
import hashlib
import random

import torch
import zmq


def tensor_hash(tensor: torch.Tensor) -> str:
    """Generate a hash of a CUDA tensor's content."""
    tensor_cpu = tensor.detach().cpu()
    data_ptr = tensor_cpu.data_ptr()
    nbytes = tensor_cpu.element_size() * tensor_cpu.numel()
    buffer = (ctypes.c_byte * nbytes).from_address(data_ptr)
    tensor_bytes = bytes(buffer)
    return hashlib.sha256(tensor_bytes).hexdigest()


def run_client(
    client_id: int,
    device_id: int | None = None,
    server_host: str = "localhost",
    server_port: int = 5555,
    num_iterations: int = 3,
):
    """Client process that sends tensors from a specific GPU and verifies responses.

    Args:
        client_id: GPU device ID to use
        device_id: Physical GPU device ID to use (if different from client_id)
        server_host: Server hostname or IP
        server_port: Server port
        num_iterations: Number of tensors to send and verify
    """
    from cornserve.sidecar.serde import MsgpackEncoder

    print(f"Starting client for GPU {client_id}")

    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.connect(f"tcp://{server_host}:{server_port}")
    torch.manual_seed(client_id * 1000)

    device = torch.device(f"cuda:{device_id}" if device_id is not None else f"cuda:{client_id}")
    print(f"Client GPU {client_id} using device {device}")

    try:
        for iteration in range(num_iterations):
            num_dims = random.randint(2, 4)
            shape = tuple(random.randint(1, 5) for _ in range(num_dims))
            tensor = torch.randn(shape, device=device, dtype=torch.bfloat16)
            data = MsgpackEncoder().encode(tensor)
            original_hash = tensor_hash(tensor)
            message = (client_id, data)

            socket.send_pyobj(message)
            print(f"Client GPU {client_id}: Sent tensor {tensor} of hash {original_hash} iteration {iteration}")

            response = socket.recv_pyobj()
            received_client_id, received_hash = response

            assert received_client_id == client_id, f"GPU ID mismatch: expected {client_id}, got {received_client_id}"
            assert received_hash == original_hash, f"Hash mismatch for GPU {client_id}, iteration {iteration}"
            print(f"Client GPU {client_id}: Verified tensor iteration {iteration}")

        # Send shutdown signal
        shutdown_message = (client_id, None)
        socket.send_pyobj(shutdown_message)
        print(f"Client GPU {client_id}: Sent shutdown signal")

    except Exception as e:
        print(f"Client GPU {client_id} error: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        socket.close()
        context.term()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tensor serialization client")
    parser.add_argument("client_id", type=int, help="Client id (default GPU device ID to use)")
    parser.add_argument("--device-id", type=int, help="The physical GPU device ID to use", default=None)
    parser.add_argument("--host", default="localhost", help="Server hostname")
    parser.add_argument("--port", type=int, default=5555, help="Server port")
    parser.add_argument("--iterations", type=int, default=3, help="Number of test iterations")

    args = parser.parse_args()

    run_client(args.client_id, args.device_id, args.host, args.port, args.iterations)
