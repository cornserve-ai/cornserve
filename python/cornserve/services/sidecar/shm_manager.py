"""Shared memory manager for sidecar communication."""

from __future__ import annotations

import torch

from cornserve.logging import get_logger

logger = get_logger(__name__)


class SharedMemoryChunk:
    """A tensor could be chunked during processing and transmission."""

    def __init__(self, size: int, data: torch.Tensor, num_shards: int) -> None:
        """Initialize a shared memory chunk, which will be viewed as a list of shards."""
        self.size = size
        self.data = data
        self.shard_availability = [False for _ in range(num_shards)]
        self.ready = False
        self.received_size = 0

    def mark_shard_ready(self, shard_rank: int, shard_size) -> None:
        """Mark a shard as ready, will set the chunk as ready if all shards are ready."""
        self.shard_availability[shard_rank] = True
        self.received_size += shard_size
        if self.received_size == self.size:
            logger.info("All shards are ready, marking chunk as ready")
            self.ready = True

    def __repr__(self):
        """Return a string representation of the shared memory chunk."""
        return f"SharedMemoryChunk(size={self.size}, shard_availability={self.shard_availability}, ready={self.ready})"


class SharedMemoryBuffer:
    """A shared memory buffer that could be sliced into multiple shards.

    Only the sidecar receiver will use the shards and track the availablity information.
    """

    def __init__(self, size: int, data: torch.Tensor, slots: list[int]):
        """Initialize a shared memory buffer, no chunking by default."""
        self.size = size
        self.data = data
        self.slots = slots
        self.is_chunked = False

    def create_chunks(self, num_chunks: int, num_shards: int):
        """Chunking the buffer with each chunk having `num_shards` shards."""
        self.is_chunked = True
        self.ready = False
        self.chunks = []
        chunk_size = self.size // num_chunks
        for i in range(num_chunks):
            self.chunks.append(
                SharedMemoryChunk(chunk_size, self.data[i * chunk_size : (i + 1) * chunk_size], num_shards)
            )
        self.chunk_availability = [False for _ in range(num_chunks)]

    def mark_chunk_ready(self, chunk_id: int):
        """Mark a chunk as ready, will set the buffer as ready if all chunks are ready."""
        assert self.is_chunked
        self.chunk_availability[chunk_id] = True
        if all(self.chunk_availability):
            self.ready = True

    def is_ready(self) -> bool:
        """Check if the buffer is ready."""
        if self.is_chunked:
            return self.ready
        else:
            return True

    def __repr__(self):
        """Return a string representation of the shared memory buffer."""
        if self.is_chunked:
            return (
                f"SharedMemoryBuffer(size={self.size}, slots={len(self.slots)}, "
                f"is_chunked={self.is_chunked}, ready={self.ready}, chunk_availability={self.chunk_availability})"
            )
        else:
            return f"SharedMemoryBuffer(size={self.size}, slots={len(self.slots)})"


class SharedMemoryManager:
    """A shared memory manager that manages a shared memory buffer backed by file.

    This class is used by the SidecarSender frontend and the SidecarReceiver backend.
    Note this class is not thread-safe, so locking and back pressure should be handled by the caller.
    """

    def __init__(self, shm: torch.Tensor, slot_size: int):
        """Initialize a shared memory manager with the given shared memory tensor.

        Args:
            shm: the shared memory tensor
            shm_size: the size of the shared memory tensor (TODO: remove this, can be inferred)
            slot_size: the size of each slot in the shared memory
        """
        self.shm = shm
        self.shm_size = shm.numel()
        self.slot_size = slot_size
        self.num_slots = self.shm_size // self.slot_size
        self.occupancy = [0 for _ in range(self.num_slots)]
        logger.info("Shared memory manager initialized with %d slots", self.num_slots)

    def allocate(self, size: int) -> SharedMemoryBuffer | None:
        """Allocate a shared memory buffer of the given size.

        This method will find a contiguous slot in the shared memory and return a buffer.
        """
        if size > self.shm_size:
            raise ValueError("Requested size exceeds shared memory size")
        if size < self.slot_size:
            raise ValueError("Requested size is smaller than slot size")
        if size % self.slot_size != 0:
            raise ValueError("Requested size is not a multiple of slot size")

        slots_needed = size // self.slot_size
        cur_free = 0
        for i in range(self.num_slots):
            if self.occupancy[i] == 0:
                cur_free += 1
                if cur_free == slots_needed:
                    for j in range(i - slots_needed + 1, i + 1):
                        self.occupancy[j] = 1
                    return SharedMemoryBuffer(
                        size=size,
                        data=self.shm[(i - slots_needed + 1) * self.slot_size : (i + 1) * self.slot_size],
                        slots=list(range(i - slots_needed + 1, i + 1)),
                    )
            else:
                cur_free = 0
        return None

    def free(self, buffer: SharedMemoryBuffer):
        """Free a shared memory buffer."""
        for slot in buffer.slots:
            self.occupancy[slot] = 0

    def free_slots(self) -> int:
        """Return the number of free slots in the shared memory."""
        return self.num_slots - sum(self.occupancy)
