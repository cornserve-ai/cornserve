import dataclasses
from functools import reduce
from operator import mul
import torch

@dataclasses.dataclass
class SidecarConfig:
    sidecar_rank: int
    # TP group of all sidecar ranks
    # used to deduct TP rank
    group: list[int] | None = None

    # performance tuning
    max_workers: int = 8

    ## Memory management
    # tensor size hint to reduce internal fragmentation
    send_tensor_dtype: torch.dtype | None = None
    send_tensor_shape: tuple[int, ...] | None = None
    recv_tensor_dtype: torch.dtype | None = None
    recv_tensor_shape: tuple[int, ...] | None = None

    concurrent_copy: bool = True

    # read_only = False
    def __post_init__(self):
        if self.group is None:
            self.group = [self.sidecar_rank]
        self.group.sort()
        if self.sidecar_rank not in self.group:
            raise ValueError("Sidecar rank should be in the group")
        if self.send_tensor_shape is not None and self.send_tensor_shape[0] != -1:
            raise ValueError("The first dimension of the send tensor shape should be -1")
        if self.recv_tensor_shape is not None and self.recv_tensor_shape[0] != -1:
            raise ValueError("The first dimension of the recv tensor shape should be -1")
        if self.max_workers <= 0:
            raise ValueError("Max workers should be positive")
        if self.send_tensor_shape is None and self.recv_tensor_shape is None:
            raise ValueError("Either send tensor shape or recv tensor shape should be set")
        if (self.send_tensor_shape is None) ^ (self.send_tensor_dtype is None):
            raise ValueError("Send tensor shape and dtype should be set together")
        if (self.recv_tensor_shape is None) ^ (self.recv_tensor_dtype is None):
            raise ValueError("Recv tensor shape and dtype should be set together")
        if self.send_tensor_dtype is not None and self.recv_tensor_dtype is not None and self.send_tensor_dtype != self.recv_tensor_dtype:
            raise ValueError("Send and recv tensor dtypes should be the same for now")

    def get_send_tensor_shape(self) -> tuple[int, ...]:
        if self.send_tensor_shape is not None:
            return self.send_tensor_shape
        if self.recv_tensor_shape is not None:
            return self.recv_tensor_shape
        raise ValueError("Either send tensor shape or recv tensor shape should be set")

    def get_send_slot_numel(self) -> int:
        if self.send_tensor_shape is not None:
            return reduce(mul, self.send_tensor_shape[1:], 1)
        if self.recv_tensor_shape is not None:
            return reduce(mul, self.recv_tensor_shape[1:], 1)
        raise ValueError("Either send tensor shape or recv tensor shape should be set")

    def get_recv_tensor_shape(self) -> tuple[int, ...]:
        if self.recv_tensor_shape is not None:
            return self.recv_tensor_shape
        if self.send_tensor_shape is not None:
            return self.send_tensor_shape
        raise ValueError("Either send tensor shape or recv tensor shape should be set")

    def get_recv_slot_numel(self) -> int:
        if self.recv_tensor_shape is not None:
            return reduce(mul, self.recv_tensor_shape[1:], 1)
        if self.send_tensor_shape is not None:
            return reduce(mul, self.send_tensor_shape[1:], 1)
        raise ValueError("Either send tensor shape or recv tensor shape should be set")

    def get_dtype(self) -> torch.dtype:
        if self.send_tensor_dtype is not None:
            return self.send_tensor_dtype
        if self.recv_tensor_dtype is not None:
            return self.recv_tensor_dtype
        raise ValueError("Either send tensor dtype or recv tensor dtype should be set")
