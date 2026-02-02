"""Sidecar constants."""

from __future__ import annotations

import os

from cornserve import constants

RANK_OFFSET = 1000000
CHUNK_OFFSET = 1000
GRPC_BASE_PORT = 10000
UCX_BASE_PORT = 12000

# Retry configuration for gRPC calls
GRPC_RETRY_MAX_ATTEMPTS = 3  # 3 total attempts (1 initial + 2 retries)
GRPC_RETRY_INITIAL_BACKOFF_SECONDS = 0.05  # Initial backoff delay in seconds
GRPC_RETRY_BACKOFF_MULTIPLIER = 2.0  # Exponential backoff multiplier


def chunk_tag(id: str, rank: int, chunk_id: int, shard_rank: int) -> int:
    """Generate a tag for the chunk.

    The tag is a unique id for a chunk during transmission.
    """
    # convert the hex uuid to int
    base = int(id, 16)
    return base + RANK_OFFSET * (rank) + CHUNK_OFFSET * (chunk_id) + shard_rank


def shm_filename() -> str:
    """Shared memory filename in each node."""
    return "/dev/shm/sc_shm"


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
