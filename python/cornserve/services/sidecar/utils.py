"""Sidecar utility functions and constants."""

from __future__ import annotations

from enum import Enum

import torch

from cornserve import constants

RANK_OFFSET = 1000000
CHUNK_OFFSET = 1000


def chunk_tag(rank: int, chunk_id: int, shard_rank: int) -> int:
    """Generate a tag for the chunk.

    The tag is a unique id for a chunk during gloo transmission.
    """
    return RANK_OFFSET * (rank) + CHUNK_OFFSET * (chunk_id) + shard_rank


def shm_fn() -> str:
    """Shared memory filename in each node."""
    return "/dev/shm/sc_shm"


def device_from_rank(rank: int) -> torch.device:
    """Torch device from rank."""
    assert rank >= 0, "Rank should be non-negative"
    return torch.device(f"cuda:{rank}")


def grpc_channel_from_rank(rank: int) -> str:
    """GRPC channel from rank."""
    assert rank >= 0, "Rank should be non-negative"
    parts = [
        f"sidecar-{rank}",
        constants.K8S_SIDECAR_SERVICE_NAME,
        constants.K8S_NAMESPACE,
        "svc.cluster.local",
    ]
    return ".".join(parts) + f":{10000 + rank}"


def init_shmem(
    fn: str,
    local_ranks: list[int],
    num_local_sidecars: int,
    size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Initialize a shared memory buffer between the sidecar client and server.

    All sidecars within the same node will share the same buffer but at different offsets.
    Each sidecar will only access its own slice of the buffer, and each slice has the same size.

    Args:
        fn: Shared memory filename.
        local_ranks: The local ranks of the sidecars that will share the buffer, must be consecutive.
        num_local_sidecars: Total number of sidecars within the same node.
        size: Number of elements of given dtype in the shared memory buffer used by each device/sidecar.
        dtype: Data type of the shared memory buffer.
    """
    # sanity check device_ids
    for i in range(len(local_ranks) - 1):
        assert local_ranks[i] + 1 == local_ranks[i + 1], "Device IDs must be consecutive"
    total_element_count = size * num_local_sidecars
    full_tensor = torch.from_file(
        filename=fn,
        shared=True,
        size=total_element_count,
        dtype=dtype,
    )
    start = size * local_ranks[0]
    end = size * (local_ranks[-1] + 1)
    return full_tensor[start:end]


class TensorLayout(Enum):
    """Tensor layout/slicing dimention for the data transferred by a sidecar."""

    FULL = 0
