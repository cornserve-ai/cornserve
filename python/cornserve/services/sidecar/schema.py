from __future__ import annotations
from dataclasses import dataclass
from cornserve.services.sidecar.shm_manager import SharedMemoryBuffer
from cornserve.sidecar.utils import init_shmem, shm_filename
from ucxx._lib_async.endpoint import Endpoint
import torch

# To allow grouping, we need to bookkeep the mapping between global rank and local rank
@dataclass
class SidecarNodeInfo:
    """Local Sidecar status within node."""
    sidecar_ranks: list[int]

    def __post_init__(self):
        # check all different
        s = set(self.sidecar_ranks)
        assert len(s) == len(self.sidecar_ranks), "Sidecarranks should be unique"
        self.sidecar_ranks.sort()

    def get_device_id(self, sidecar_rank: int) -> int:
        """Get the device id of the sidecar, the same as local rank."""
        return self.sidecar_ranks.index(sidecar_rank)

    def get_sidecar_num(self) -> int:
        """Get the number of sidecars on the node."""
        return len(self.sidecar_ranks)

    def contains(self, sidecar_rank: int) -> bool:
        """Check if the sidecar rank is in the node."""
        return sidecar_rank in self.sidecar_ranks

    def get_local_ranks(self, sidecar_ranks: list[int]) -> list[int]:
        """Get the local ranks of the sidecars."""
        return [self.sidecar_ranks.index(rank) for rank in sidecar_ranks]

@dataclass
class SidecarReceiverConfig:
    sidecar_rank: int
    node_info: SidecarNodeInfo
    peers: dict[int, Endpoint]
    # TP group, when enabled, only the leader sidecar will perform action
    group: list[int]
    base_ptr: int
    shared_tensor: torch.Tensor
    shm_numel: int
    slot_numel: int
    dtype: torch.dtype
    full_tensor: torch.Tensor

@dataclass
class SidecarSenderConfig:
    sidecar_rank: int
    node_info: SidecarNodeInfo
    peers: dict[int, Endpoint]
    # TP group, when enabled, only the leader sidecar will perform action
    # the sidecar_ranks in the group
    group: list[int]
    base_ptr: int
    shared_tensor: torch.Tensor
    shm_numel: int
    slot_numel: int
    dtype: torch.dtype
    full_tensor: torch.Tensor
    concurrent_copy: bool = False

@dataclass
class SidecarServerConfig:
    sidecar_rank: int
    node_info: SidecarNodeInfo
    # TP group, when enabled, only the leader sidecar will perform action
    peers: dict[int, Endpoint]
    # need to be sorted
    group: list[int]

    tensor_dtype: torch.dtype
    slab_numel: int

    send_slot_numel: int
    recv_slot_numel: int
    concurrent_copy: bool = True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SidecarServerConfig):
            return False
        # ignores sidecar_rank due to grouping
        return self.group == sorted(other.group) \
                and self.tensor_dtype == other.tensor_dtype \
                and self.slab_numel == other.slab_numel \
                and self.send_slot_numel == other.send_slot_numel \
                and self.recv_slot_numel == other.recv_slot_numel \
                and self.concurrent_copy == other.concurrent_copy

    def sender_config(self) -> SidecarSenderConfig:
        full_tensor, slab = init_shmem(
            filename=shm_filename(),
            local_ranks=list(range(self.node_info.get_sidecar_num())),
            num_local_sidecars=self.node_info.get_sidecar_num(),
            partition_numel=self.slab_numel*2,
            dtype=self.tensor_dtype,
        )
        return SidecarSenderConfig(
            sidecar_rank=self.sidecar_rank,
            node_info=self.node_info,
            peers=self.peers,
            group=self.group,
            base_ptr=full_tensor.data_ptr(),
            shared_tensor=slab[:slab.numel()//2],
            shm_numel=self.slab_numel,
            slot_numel=self.send_slot_numel,
            dtype=self.tensor_dtype,
            concurrent_copy=self.concurrent_copy,
            full_tensor=full_tensor,
        )

    def receiver_config(self) -> SidecarReceiverConfig:
        full_tensor, slab = init_shmem(
            filename=shm_filename(),
            local_ranks=list(range(self.node_info.get_sidecar_num())),
            num_local_sidecars=self.node_info.get_sidecar_num(),
            partition_numel=self.slab_numel*2,
            dtype=self.tensor_dtype,
        )
        return SidecarReceiverConfig(
            sidecar_rank=self.sidecar_rank,
            node_info=self.node_info,
            peers=self.peers,
            group=self.group,
            base_ptr=full_tensor.data_ptr(),
            shared_tensor=slab[slab.numel()//2:],
            shm_numel=self.slab_numel,
            slot_numel=self.send_slot_numel,
            dtype=self.tensor_dtype,
            full_tensor=full_tensor,
        )


@dataclass
class SendTransferRequestState:
    """Internal data structure to keep track of a tansfer request's state.

    Attributes:
        - id: The concatenation of request_id and data_id
        - buffer: The shared memory buffer used to recv the data
        - done: A flag to indicate if the transfer is fully sent
    """

    id: str
    buffer: SharedMemoryBuffer
    shards_sent: list[bool]
    done: bool = False

@dataclass
class RecvTransferRequestState:
    """Internal data structure to keep track of a tansfer request's state.

    Attributes:
        - id: The concatenation of request_id and data_id
        - buffer: The shared memory buffer used to recv the data
        - done: A flag to indicate if the transfer is done
    """

    id: str
    buffer: SharedMemoryBuffer
    intra_node_rank: int = -1
    done: bool = False
