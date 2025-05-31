"""Shared memory manager for sidecar communication."""

from __future__ import annotations

import torch

from cornserve.logging import get_logger
from cornserve.sidecar.serde import SharedTensorHandle

logger = get_logger(__name__)


class SharedMemoryShard:
    """A shard of a shared memory buffer."""

    def __init__(self, offset: int, length: int, data: torch.Tensor) -> None:
        """Initialize a shared memory shard.

        Args:
            offset: the offset of the shard in the shared memory buffer
            length: the length of the shard
            data: the backing tensor storing the data
        """
        self.offset = offset
        self.length = length
        self.data = data

    def __repr__(self) -> str:
        """Return a string representation of the shared memory shard."""
        return f"SharedMemoryShard(offset={self.offset}, length={self.length}, data={self.data})"


class SharedMemoryBuffer:
    """A shared memory buffer that could be sliced into multiple shards."""

    def __init__(self, size: int, data: torch.Tensor, slots: list[int]) -> None:
        """Initialize a shared memory buffer, no sharding by default.

        Args:
            size: the real size (numel) of the data
            data: the tensor storing the data, could be larger than size
            slots: the list of slots in the shared memory manager view
        """
        self.size = size
        self.data = data
        self.slots = slots
        self.is_sharded = False
        self.shards: list[SharedMemoryShard] = []
        self.shard_availability: list[bool] = []

    def create_shards(self, num_shards: int) -> None:
        """Create shards from the buffer."""
        if self.is_sharded:
            raise ValueError("Buffer is already sharded")
        # shard based on the shard rank and num shards
        for shard_rank in range(num_shards):
            quotient, remainder = divmod(self.size, num_shards)
            start_pos = shard_rank * quotient + min(shard_rank, remainder)
            end_pos = start_pos + quotient + (1 if shard_rank < remainder else 0)
            shard_data = self.data[start_pos:end_pos]
            self.shards.append(
                SharedMemoryShard(
                    offset=start_pos,
                    length=end_pos - start_pos,
                    data=shard_data,
                )
            )
            self.shard_availability.append(False)
        self.is_sharded = True
        self.ready = False

    def mark_shard_ready(self, shard_rank: int) -> None:
        """Mark a shard as ready, will set the buffer as ready if all shards are ready."""
        assert self.is_sharded
        self.shard_availability[shard_rank] = True
        if all(self.shard_availability):
            self.ready = True

    def is_ready(self) -> bool:
        """Check if the buffer is ready."""
        if self.is_sharded:
            return self.ready
        else:
            return True

    def __repr__(self) -> str:
        """Return a string representation of the shared memory buffer."""
        if self.is_sharded:
            return (
                f"SharedMemoryBuffer(size={self.size}, slots={len(self.slots)}, "
                f"is_sharded={self.is_sharded}, ready={self.ready}, shard_availability={self.shard_availability}, "
                f"shards={self.shards})"
            )
        else:
            return f"SharedMemoryBuffer(size={self.size}, slots={len(self.slots)})"

    def create_handle(self, base_ptr: int) -> SharedTensorHandle:
        """Create a handle for the shared memory buffer."""
        return SharedTensorHandle(
            offset=self.data.data_ptr() - base_ptr,
            numel=self.size,
        )


class SharedMemoryManager:
    """A shared memory manager that manages a shared memory buffer backed by file.

    This class is used by the SidecarSender frontend and the SidecarReceiver backend.
    Note this class is not thread-safe, so locking and back pressure should be handled by the caller.
    """

    def __init__(self, shm: torch.Tensor, slot_size: int) -> None:
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
        logger.info("Shared memory manager initialized with %d slots of slot size%d", self.num_slots, self.slot_size)

    def allocate(self, size: int) -> SharedMemoryBuffer | None:
        """Allocate a shared memory buffer of the given size.

        This method will find a contiguous slot in the shared memory and return a buffer.
        """
        if size > self.shm_size:
            raise ValueError("Requested size exceeds shared memory size")
        if size < self.slot_size:
            raise ValueError("Requested size is smaller than slot size")
        if size % self.slot_size != 0:
            raise ValueError(f"Requested size {size} is not a multiple of slot size {self.slot_size}")

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

    def free(self, buffer: SharedMemoryBuffer) -> None:
        """Free a shared memory buffer."""
        for slot in buffer.slots:
            self.occupancy[slot] = 0

    def free_slots(self) -> int:
        """Return the number of free slots in the shared memory."""
        return self.num_slots - sum(self.occupancy)
