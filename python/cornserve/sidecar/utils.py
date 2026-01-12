"""Sidecar utility functions and constants."""

from __future__ import annotations

import ctypes

import torch

from cornserve.logging import get_logger

logger = get_logger(__name__)


def buffer_from_tensor(t: torch.Tensor) -> ctypes.Array:
    """Convert a torch tensor to a ctypes buffer for ucx-py."""
    data_ptr = t.data_ptr()
    size_bytes = t.numel() * t.element_size()
    buffer = (ctypes.c_byte * size_bytes).from_address(data_ptr)
    return buffer


def device_from_rank(rank: int) -> torch.device:
    """Torch device from rank."""
    assert rank >= 0, "Rank should be non-negative"
    return torch.device(f"cuda:{rank}")


def init_shmem(
    filename: str,
    local_ranks: list[int],
    num_local_sidecars: int,
    partition_bytes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize a shared memory buffer between the sidecar client and server.

    All sidecars within the same node will share the same buffer but at different offsets.
    Each sidecar will only access its own slice of the buffer, and each slice has the same size.

    Args:
        filename: The filename of the shared memory buffer.
        local_ranks: The local ranks of the sidecars that will share the buffer, must be consecutive.
        num_local_sidecars: Total number of sidecars within the same node.
        partition_bytes: Number of bytes in the shared memory buffer used by each device/sidecar.
    """
    # sanity check device_ids
    for i in range(len(local_ranks) - 1):
        assert local_ranks[i] + 1 == local_ranks[i + 1], "Device IDs must be consecutive"

    element_size = dtype.itemsize
    assert partition_bytes % element_size == 0, "Partition bytes must be divisible by element size"
    partition_numel = partition_bytes // element_size

    total_element_count = partition_numel * num_local_sidecars
    full_tensor = torch.from_file(
        filename=filename,
        shared=True,
        size=total_element_count,
        dtype=torch.uint8,
    )
    start = partition_numel * local_ranks[0]
    end = partition_numel * (local_ranks[-1] + 1)
    return full_tensor, full_tensor[start:end]
