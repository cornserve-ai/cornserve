"""Sidecar utility functions and constants."""
from __future__ import annotations
import torch

from enum import Enum

RANK_OFFSET = 1000000
CHUNK_OFFSET = 1000

def chunk_tag(rank: int, chunk_id: int, shard_rank: int) -> int:
    """Generate a tag for the chunk.

    The tag is a unique id for a chunk during gloo transmission.
    """
    return RANK_OFFSET * (rank) + CHUNK_OFFSET * (chunk_id) + shard_rank


def shm_fn_from_rank(rank: int) -> str:
    """Unique shared memory filename for each rank."""
    assert rank >= 0, "Rank should be non-negative"
    return f"/dev/shm/sc_shm_{rank}"


def device_from_rank(rank: int) -> torch.device:
    """Torch device from rank."""
    assert rank >= 0, "Rank should be non-negative"
    return torch.device(f"cuda:{rank}")


def grpc_channel_from_rank(rank: int) -> str:
    """GRPC channel from rank."""
    assert rank >= 0, "Rank should be non-negative"
    return f"sidecar-{rank}.torch-headless.cornserve.svc.cluster.local:{10000 + rank}"


def init_shmem(shm_fn: str, size: int, dtype: torch.dtype) -> torch.Tensor:
    """Initialize shared memory buffer from filename, size and dtype."""
    shared_tensor = torch.from_file(
        filename=shm_fn,
        shared=True,
        size=size,
        dtype=dtype,
    )
    return shared_tensor


class TensorLayout(Enum):
    """Tensor layout/slicing dimention for the data transferred by a sidecar."""
    FULL = 0
