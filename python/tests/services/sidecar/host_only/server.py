#!/usr/bin/env python3
"""
Standalone server for tensor serialization testing.
This server listens for tensor data, deserializes it, and sends back a hash of the tensor.
"""

import ctypes
import hashlib
import pickle

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


def run_server(port: int = 5555, expected_clients: int = 4):
    """Run the tensor serialization server.

    Args:
        port: Port to listen on
        expected_clients: Number of clients expected before shutdown
    """
    from cornserve.sidecar.serde import MsgpackDecoder

    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind(f"tcp://*:{port}")

    # CUDA manual init

    for i in range(expected_clients):
        try:
            torch.cuda.set_device(i)
            # Perform a small, innocuous operation to force context creation
            _ = torch.tensor([1.0], device=f"cuda:{i}")
            print(f"Successfully initialized context on cuda:{i}")
        except Exception as e:
            print(f"Could not initialize cuda:{i}. Error: {e}")

    shutdown_count = 0

    print(f"Server started on port {port}, expecting {expected_clients} clients")

    while True:
        try:
            message_parts = socket.recv_multipart()

            if len(message_parts) != 2:
                print(f"Unexpected message format: got {len(message_parts)} parts, expected 2")
                continue

            client_identity, serialized_message = message_parts
            gpu_id, data = pickle.loads(serialized_message)
            print(f"Received message from client {client_identity.hex()[:8]} for GPU {gpu_id}")

            if data is None:  # Shutdown signal
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
            print(f"Received tensor on GPU {gpu_id}: {tensor}-{tensor.shape}, hash: {hash_value}")
            del tensor

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tensor serialization server")
    parser.add_argument("--port", type=int, default=5555, help="Port to listen on")
    parser.add_argument("--clients", type=int, default=4, help="Number of expected clients")

    args = parser.parse_args()

    run_server(port=args.port, expected_clients=args.clients)
