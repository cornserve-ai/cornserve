import ctypes
import hashlib
import multiprocessing
import random
from collections.abc import Generator
from multiprocessing.process import BaseProcess

import pytest
import torch
import zmq

from cornserve.sidecar.serde import MsgpackDecoder, MsgpackEncoder

random.seed(0)

NUM_GPUS = 4
IPC_PATH = "ipc:///tmp/test_serde"


def tensor_hash(tensor: torch.Tensor) -> str:
    """Generate a hash of a CUDA tensor's content.

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
    process.join(timeout=5)


def run_server() -> None:
    """A server that listens for tensor data, deserializes it, and sends back a hash of the tensor."""
    import pickle

    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind(IPC_PATH)

    shutdown_count = 0
    expected_clients = NUM_GPUS

    print(f"Server started, expecting {expected_clients} clients")

    while True:
        try:
            # ROUTER receives: [client_identity, serialized_message] when DEALER uses send_pyobj()
            message_parts = socket.recv_multipart()

            if len(message_parts) != 2:
                print(f"Unexpected message format: got {len(message_parts)} parts, expected 2")
                continue

            client_identity, serialized_message = message_parts

            # Deserialize the actual message
            gpu_id, data = pickle.loads(serialized_message)

            if data is None:  # Shutdown signal
                print(f"Received shutdown from client {client_identity.hex()[:8]} (GPU {gpu_id})")
                shutdown_count += 1

                if shutdown_count >= expected_clients:
                    print("All clients have sent shutdown signals")
                    break
                continue

            print(f"Processing request from client {client_identity.hex()[:8]} for GPU {gpu_id}")

            # Set device and deserialize tensor
            torch.cuda.set_device(gpu_id)
            tensor = MsgpackDecoder().decode(data)
            hash_value = tensor_hash(tensor)
            print(f"Received tensor on GPU {gpu_id}: {tensor.shape}, hash: {hash_value}")

            # Send response back to specific client
            response = (gpu_id, hash_value)
            serialized_response = pickle.dumps(response)
            socket.send_multipart([client_identity, serialized_response])

        except Exception as e:
            print(f"Server error: {e}")
            import traceback

            traceback.print_exc()
            break

    print("Server shutting down")
    socket.close()
    context.term()


def run_client(gpu_id: int, num_iterations: int = 1) -> None:
    """Client process that sends tensors from a specific GPU and verifies responses.

    Args:
        gpu_id: GPU device ID to use
        num_iterations: Number of tensors to send and verify
    """
    print(f"Starting client for GPU {gpu_id}")

    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.connect(IPC_PATH)
    torch.manual_seed(gpu_id * 1000)

    try:
        for iteration in range(num_iterations):
            num_dims = random.randint(1, 4)
            shape = tuple(random.randint(1, 10) for _ in range(num_dims))
            tensor = torch.randn(shape, device=f"cuda:{gpu_id}", dtype=torch.bfloat16)
            data = MsgpackEncoder().encode(tensor)
            original_hash = tensor_hash(tensor)
            message = (gpu_id, data)

            # DEALER sends single-part message, ROUTER receives [client_identity, message]
            socket.send_pyobj(message)
            print(f"Client GPU {gpu_id}: Sent tensor iteration {iteration}")

            response = socket.recv_pyobj()
            received_gpu_id, received_hash = response

            assert received_gpu_id == gpu_id, f"GPU ID mismatch: expected {gpu_id}, got {received_gpu_id}"
            assert received_hash == original_hash, f"Hash mismatch for GPU {gpu_id}, iteration {iteration}"
            print(f"Client GPU {gpu_id}: Verified tensor iteration {iteration}")

        # Send shutdown signal
        shutdown_message = (gpu_id, None)
        socket.send_pyobj(shutdown_message)
        print(f"Client GPU {gpu_id}: Sent shutdown signal")

    except Exception as e:
        print(f"Client GPU {gpu_id} error: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        socket.close()
        context.term()


def test_multi_client_serde(server: BaseProcess) -> None:
    """Test multiple client processes, each using a dedicated GPU."""
    assert server.is_alive(), "Server process is not alive"

    # Create multiple client processes
    ctx = multiprocessing.get_context("spawn")
    client_processes: list[BaseProcess] = []

    try:
        # Start client processes
        for gpu_id in range(NUM_GPUS):
            process = ctx.Process(target=run_client, args=(gpu_id, 3))
            process.start()
            client_processes.append(process)

        # Wait for all clients to complete
        for i, process in enumerate(client_processes):
            process.join(timeout=15)
            if process.exitcode != 0:
                raise RuntimeError(f"Client process for GPU {i} failed with exit code {process.exitcode}")
            print(f"Client process for GPU {i} completed successfully")

        print("All client processes completed successfully")

    except Exception as e:
        print(f"Test failed: {e}")
        # Clean up any remaining processes
        for process in client_processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
        raise

    finally:
        # Ensure all processes are cleaned up
        for process in client_processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
