"""PyTorch Distributed management and collectives for sequence parallelism.

Mirrors Eric's distributed/parallel.py but for sequence parallelism (SP)
instead of tensor parallelism (TP). SP splits the sequence (spatial token)
dimension across GPUs, using All-to-All for attention redistribution.
"""

from __future__ import annotations

import torch
import torch.distributed

from cornserve.logging import get_logger

logger = get_logger(__name__)


class SPGroup:
    """A group of devices participating in sequence parallelism.

    ``torch.distributed`` must be initialized before creating an SP group.
    The only collective exposed on the group is :meth:`all_to_all`, which is
    the core primitive for Ulysses SP attention redistribution.

    Attributes:
        rank: Rank of the current process in the SP group.
        world_size: Number of ranks in the SP group.
        process_group: The underlying PyTorch distributed process group (NCCL).
    """

    def __init__(self, ranks: list[int], name: str) -> None:
        """Initialize the SP group.

        Args:
            ranks: Global ranks that belong to this SP group.
            name: Human-readable name for logging.
        """
        self.name = name
        self.ranks = ranks
        self.world_size = len(ranks)

        if self.world_size == 1:
            self.process_group = None
            self.rank = 0
            return

        if not torch.distributed.is_initialized():
            raise RuntimeError(f"Distributed process group is not initialized. Cannot create SP group {name}.")

        self.process_group = torch.distributed.new_group(ranks=ranks, backend="nccl")
        global_rank = torch.distributed.get_rank()
        self.rank = ranks.index(global_rank)

        logger.info(
            "SP group %s initialized with ranks %s (local rank %d, world size %d).",
            name,
            ranks,
            self.rank,
            self.world_size,
        )

    def all_to_all(self, input_: torch.Tensor, scatter_dim: int, gather_dim: int) -> torch.Tensor:
        """Perform an All-to-All collective.

        Splits ``input_`` along ``scatter_dim``, sends the i-th chunk to rank i,
        and gathers received chunks along ``gather_dim``.

        Used by Ulysses SP attention:
        - Before attention: scatter_dim=seq, gather_dim=head  (seq-split → head-split)
        - After attention:  scatter_dim=head, gather_dim=seq  (head-split → seq-split)

        Args:
            input_: Input tensor with shape [B, H, S, D] or similar.
            scatter_dim: Dimension to split and scatter.
            gather_dim: Dimension along which to concatenate received chunks.

        Returns:
            Redistributed tensor.
        """
        if self.world_size == 1:
            return input_

        if scatter_dim < 0:
            scatter_dim += input_.dim()
        if gather_dim < 0:
            gather_dim += input_.dim()

        assert input_.size(scatter_dim) % self.world_size == 0, (
            f"scatter_dim size ({input_.size(scatter_dim)}) must be divisible "
            f"by world_size ({self.world_size})"
        )

        # Split input along scatter_dim into world_size chunks.
        input_chunks = input_.chunk(self.world_size, dim=scatter_dim)
        # Each chunk must be contiguous for NCCL.
        input_chunks = [chunk.contiguous() for chunk in input_chunks]

        # Prepare output buffers with the same shape as each input chunk.
        output_chunks = [torch.empty_like(input_chunks[0]) for _ in range(self.world_size)]

        torch.distributed.all_to_all(output_chunks, input_chunks, group=self.process_group)

        # Concatenate along gather_dim.
        return torch.cat(output_chunks, dim=gather_dim)

    def scatter(self, input_: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """Split a tensor along ``dim`` and keep only this rank's chunk.

        This is a local slicing operation — no communication involved.

        Args:
            input_: Full tensor to split.
            dim: Dimension along which to split.

        Returns:
            This rank's chunk of the tensor.
        """
        if self.world_size == 1:
            return input_

        if dim < 0:
            dim += input_.dim()

        chunks = input_.chunk(self.world_size, dim=dim)
        return chunks[self.rank].contiguous()

    def all_gather(self, input_: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """All-gather tensors along the given dimension.

        Args:
            input_: Local tensor shard.
            dim: Dimension along which to gather.

        Returns:
            Gathered tensor with ``dim`` size multiplied by ``world_size``.
        """
        if self.world_size == 1:
            return input_

        if dim < 0:
            dim += input_.dim()

        input_size = input_.size()
        flat_size = list(input_size)
        flat_size[0] *= self.world_size
        output_tensor = torch.empty(flat_size, dtype=input_.dtype, device=input_.device)
        torch.distributed.all_gather_into_tensor(output_tensor, input_.contiguous(), group=self.process_group)

        if dim == 0:
            return output_tensor

        output_tensor = output_tensor.reshape((self.world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        merged_size = list(input_size)
        merged_size[dim] *= self.world_size
        return output_tensor.reshape(merged_size)

    def broadcast(self, input_: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast a tensor from ``src`` rank to all ranks.

        Args:
            input_: Tensor to broadcast (only meaningful on src rank).
            src: Source rank within the SP group (0-indexed).

        Returns:
            Broadcast tensor on all ranks.
        """
        if self.world_size == 1:
            return input_

        torch.distributed.broadcast(input_, src=self.ranks[src], group=self.process_group)
        return input_

    def barrier(self) -> None:
        """Synchronize all ranks in the SP group."""
        if self.world_size == 1:
            return
        torch.distributed.barrier(group=self.process_group)


# ---------------------------------------------------------------------------
# Global SP group singleton
# ---------------------------------------------------------------------------

_SP_GROUP: SPGroup | None = None


def get_sp_group() -> SPGroup:
    """Get the global sequence parallelism group.

    Works even when ``sp_size=1``; collective calls will be no-ops.
    """
    if _SP_GROUP is None:
        raise RuntimeError("Sequence parallel group is not initialized.")
    return _SP_GROUP


def init_sp_distributed(
    world_size: int,
    rank: int,
    backend: str = "nccl",
    init_method: str = "tcp://127.0.0.1:29500",
) -> None:
    """Initialize the distributed process group and the global SP group.

    Args:
        world_size: Total number of SP workers.
        rank: Rank of this worker.
        backend: PyTorch distributed backend (default ``"nccl"``).
        init_method: URL for rendezvous.
    """
    if torch.distributed.is_initialized():
        logger.warning("Distributed process group is already initialized. Skipping initialization.")
        return

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
    else:
        logger.warning("CUDA is not available. Continuing to initialize distributed environment without CUDA.")

    if world_size > 1:
        torch.distributed.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
        logger.info(
            "Distributed process group initialized (world_size=%d, rank=%d).",
            world_size,
            rank,
        )

    global _SP_GROUP
    _SP_GROUP = SPGroup(
        ranks=list(range(world_size)),
        name="sequence_parallel_group",
    )


def destroy_sp_distributed() -> None:
    """Destroy the distributed process group."""
    if not torch.distributed.is_initialized():
        logger.warning("Distributed process group is not initialized. Skipping destruction.")
        return

    torch.distributed.destroy_process_group()
    logger.info("Destroyed distributed process groups.")
