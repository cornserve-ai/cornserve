"""The client-side data structures."""

import dataclasses
from functools import reduce
from operator import mul

import torch


@dataclasses.dataclass
class SidecarConfig:
    """The config to initialize the sidecar server.

    Attributes:
        sidecar_rank: The rank of the sidecar server.
        group: The sidecar ranks in the TP group.
        max_workers: The maximum number of workers in the thread pool.
        send_tensor_dtype: The dtype of the tensor to be sent.
        send_tensor_shape: The shape of the tensor to be sent.
        recv_tensor_dtype: The dtype of the tensor to be received.
        recv_tensor_shape: The shape of the tensor to be received.
        concurrent_copy: Whether to enable concurrent copy in the sidecar server.
    """

    sidecar_rank: int
    # TP group of all sidecar ranks
    # used to deduct TP rank
    group: list[int] | None = None

    # performance tuning
    max_workers: int = 8

    ## Memory management
    # tensor size hint to reduce internal fragmentation
    send_tensor_dtype: torch.dtype | None = None
    send_tensor_shape: tuple[int, ...] | int | None = None
    recv_tensor_dtype: torch.dtype | None = None
    recv_tensor_shape: tuple[int, ...] | int | None = None

    concurrent_copy: bool = True

    # read_only = False

    def __post_init__(self) -> None:
        """Post-initialization checks for the SidecarConfig class."""
        if self.group is None:
            self.group = [self.sidecar_rank]
        self.group.sort()
        if self.sidecar_rank not in self.group:
            raise ValueError("Sidecar rank should be in the group")
        if self.max_workers <= 0:
            raise ValueError("Max workers should be positive")
        if self.send_tensor_shape is None and self.recv_tensor_shape is None:
            raise ValueError("Either send tensor shape or recv tensor shape should be set")
        if (self.send_tensor_shape is None) ^ (self.send_tensor_dtype is None):
            raise ValueError("Send tensor shape and dtype should be set together")
        if (self.recv_tensor_shape is None) ^ (self.recv_tensor_dtype is None):
            raise ValueError("Recv tensor shape and dtype should be set together")
        if (
            self.send_tensor_dtype is not None
            and self.recv_tensor_dtype is not None
            and self.send_tensor_dtype != self.recv_tensor_dtype
        ):
            raise ValueError("Send and recv tensor dtypes should be the same for now")

    def get_send_tensor_shape(self) -> tuple[int, ...]:
        """Return the send tensor shape."""
        if self.send_tensor_shape is not None:
            return self.send_tensor_shape if isinstance(self.send_tensor_shape, tuple) else (self.send_tensor_shape,)
        if self.recv_tensor_shape is not None:
            return self.recv_tensor_shape if isinstance(self.recv_tensor_shape, tuple) else (self.recv_tensor_shape,)
        raise ValueError("Either send tensor shape or recv tensor shape should be set")

    def get_send_slot_numel(self) -> int:
        """Return the slot_numel to use for the sender shared memory manager."""
        if self.send_tensor_shape is not None:
            return (
                reduce(mul, self.send_tensor_shape[1:], 1)
                if isinstance(self.send_tensor_shape, tuple)
                else self.send_tensor_shape
            )
        if self.recv_tensor_shape is not None:
            return (
                reduce(mul, self.recv_tensor_shape[1:], 1)
                if isinstance(self.recv_tensor_shape, tuple)
                else self.recv_tensor_shape
            )
        raise ValueError("Either send tensor shape or recv tensor shape should be set")

    def get_recv_tensor_shape(self) -> tuple[int, ...]:
        """Return the recv tensor shape."""
        if self.recv_tensor_shape is not None:
            return self.recv_tensor_shape if isinstance(self.recv_tensor_shape, tuple) else (self.recv_tensor_shape,)
        if self.send_tensor_shape is not None:
            return self.send_tensor_shape if isinstance(self.send_tensor_shape, tuple) else (self.send_tensor_shape,)
        raise ValueError("Either send tensor shape or recv tensor shape should be set")

    def get_recv_slot_numel(self) -> int:
        """Return the slot_numel to use for the receiver shared memory manager."""
        if self.recv_tensor_shape is not None:
            return (
                reduce(mul, self.recv_tensor_shape[1:], 1)
                if isinstance(self.recv_tensor_shape, tuple)
                else self.recv_tensor_shape
            )
        if self.send_tensor_shape is not None:
            return (
                reduce(mul, self.send_tensor_shape[1:], 1)
                if isinstance(self.send_tensor_shape, tuple)
                else self.send_tensor_shape
            )
        raise ValueError("Either send tensor shape or recv tensor shape should be set")

    def get_dtype(self) -> torch.dtype:
        """Return the dtype to use for the sender and receiver shared memory manager."""
        if self.send_tensor_dtype is not None:
            return self.send_tensor_dtype
        if self.recv_tensor_dtype is not None:
            return self.recv_tensor_dtype
        raise ValueError("Either send tensor dtype or recv tensor dtype should be set")
