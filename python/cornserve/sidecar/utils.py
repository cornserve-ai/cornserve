"""Sidecar utility functions and constants."""

from __future__ import annotations

import ctypes
import os

import torch

from cornserve import constants
from cornserve.logging import get_logger

logger = get_logger(__name__)

RANK_OFFSET = 1000000
CHUNK_OFFSET = 1000

GRPC_BASE_PORT = 10000
UCX_BASE_PORT = 12000


def chunk_tag(id: str, rank: int, chunk_id: int, shard_rank: int) -> int:
    """Generate a tag for the chunk.

    The tag is a unique id for a chunk during transmission.
    """
    # convert the hex uuid to int
    base = int(id, 16)
    return base + RANK_OFFSET * (rank) + CHUNK_OFFSET * (chunk_id) + shard_rank


def buffer_from_tensor(t: torch.Tensor) -> ctypes.Array:
    """Convert a torch tensor to a ctypes buffer for ucx-py."""
    data_ptr = t.data_ptr()
    size_bytes = t.numel() * t.element_size()
    buffer = (ctypes.c_byte * size_bytes).from_address(data_ptr)
    return buffer


def shm_filename() -> str:
    """Shared memory filename in each node."""
    return "/dev/shm/sc_shm"


def device_from_rank(rank: int) -> torch.device:
    """Torch device from rank."""
    assert rank >= 0, "Rank should be non-negative"
    return torch.device(f"cuda:{rank}")


def grpc_url_from_rank(rank: int) -> str:
    """GRPC channel url from rank."""
    assert rank >= 0, "Rank should be non-negative"
    is_local = os.environ.get("SIDECAR_IS_LOCAL", "false").lower() == "true"
    if is_local:
        return f"localhost:{GRPC_BASE_PORT + rank}"
    parts = [
        f"sidecar-{rank}",
        constants.K8S_SIDECAR_SERVICE_NAME,
        constants.K8S_NAMESPACE,
        "svc.cluster.local",
    ]
    return ".".join(parts) + f":{GRPC_BASE_PORT + rank}"


def ucx_url_from_rank(rank: int) -> str:
    """UCX connection host url from rank."""
    assert rank >= 0, "Rank should be non-negative"
    is_local = os.environ.get("SIDECAR_IS_LOCAL", "false").lower() == "true"
    if is_local:
        return "127.0.0.1"
    parts = [
        f"sidecar-{rank}",
        constants.K8S_SIDECAR_SERVICE_NAME,
        constants.K8S_NAMESPACE,
        "svc.cluster.local",
    ]
    return ".".join(parts)


def ucx_port_from_rank(rank: int) -> int:
    """UCX connection host port from rank."""
    assert rank >= 0, "Rank should be non-negative"
    return UCX_BASE_PORT + rank


def init_shmem(
    filename: str,
    local_ranks: list[int],
    num_local_sidecars: int,
    partition_numel: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize a shared memory buffer between the sidecar client and server.

    All sidecars within the same node will share the same buffer but at different offsets.
    Each sidecar will only access its own slice of the buffer, and each slice has the same size.

    Args:
        filename: The filename of the shared memory buffer.
        local_ranks: The local ranks of the sidecars that will share the buffer, must be consecutive.
        num_local_sidecars: Total number of sidecars within the same node.
        partition_numel: Number of elements of given dtype in the shared memory buffer used by each device/sidecar.
        dtype: Data type of the shared memory buffer.
    """
    # sanity check device_ids
    for i in range(len(local_ranks) - 1):
        assert local_ranks[i] + 1 == local_ranks[i + 1], "Device IDs must be consecutive"
    total_element_count = partition_numel * num_local_sidecars
    full_tensor = torch.from_file(
        filename=filename,
        shared=True,
        size=total_element_count,
        dtype=dtype,
    )
    start = partition_numel * local_ranks[0]
    end = partition_numel * (local_ranks[-1] + 1)
    return full_tensor, full_tensor[start:end]
