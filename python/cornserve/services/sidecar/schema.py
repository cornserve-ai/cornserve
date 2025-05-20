"""Bookkeeping data structures for Sidecar."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import torch
from ucxx._lib_async.endpoint import Endpoint  # type: ignore

from cornserve.services.sidecar.shm_manager import SharedMemoryBuffer
from cornserve.sidecar.utils import init_shmem, shm_filename


# To allow grouping, we need to bookkeep the mapping between global rank and local rank
@dataclass
class SidecarNodeInfo:
    """Local Sidecar status within node.

    Attributes:
        sidecar_ranks: The sidecars on the current node.
    """

    sidecar_ranks: list[int]

    def __post_init__(self) -> None:
        """Check the sidecar ranks are unique and make sure they are sorted."""
        # check all different
        s = set(self.sidecar_ranks)
        assert len(s) == len(self.sidecar_ranks), "sidecar ranks should be unique"
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
    """Sidecar receiver configuration.

    Attributes:
        sidecar_rank: The rank of the sidecar in the group.
        node_info: The node info of the sidecar server.
        peers: The peers of in the sidecar server.
        group: The current group of sidecars.
        base_ptr: The base pointer of the shared memory buffer.
        shared_tensor: The shared tensor allocated to the receiver.
        slot_numel: The number of elements in each slot.
    """

    sidecar_rank: int
    node_info: SidecarNodeInfo
    peers: dict[int, Endpoint]
    # TP group, when enabled, only the leader sidecar will perform action
    group: list[int]
    base_ptr: int
    shared_tensor: torch.Tensor
    slot_numel: int


@dataclass
class SidecarSenderConfig:
    """Sidecar sender configuration.

    Attributes:
        sidecar_rank: The rank of the sidecar in the group.
        node_info: The node info of the sidecar server.
        peers: The peers of in the sidecar server.
        group: The current group of sidecars.
        base_ptr: The base pointer of the shared memory buffer.
        shared_tensor: The shared tensor allocated to the sender.
        slot_numel: The number of elements in each slot.
        concurrent_copy: Whether to use concurrent copy when TP is enbaled.
    """

    sidecar_rank: int
    node_info: SidecarNodeInfo
    peers: dict[int, Endpoint]
    # TP group, when enabled, only the leader sidecar will perform action
    # the sidecar_ranks in the group
    group: list[int]
    base_ptr: int
    shared_tensor: torch.Tensor
    slot_numel: int
    concurrent_copy: bool = False


@dataclass
class SidecarServerConfig:
    """Sidecar server configuration.

    Attributes:
        sidecar_rank: The rank of the sidecar in the group.
        node_info: The node info of the sidecar server.
        peers: The peers of in the sidecar server.
        group: The current group of sidecars.
        tensor_dtype: The data type of the tensor.
        slab_numel: The number of elements in each slab.
        send_slot_numel: The number of elements in each send slot.
        recv_slot_numel: The number of elements in each recv slot.
        concurrent_copy: Whether to use concurrent copy when TP is enbaled.
    """

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

    def __post_init__(self) -> None:
        """Check the group is sorted and unique."""
        s = set(self.group)
        assert len(s) == len(self.group), "Sidecar ranks should be unique"
        self.group.sort()

    def __eq__(self, other: object) -> bool:
        """Check if two SidecarServerConfig are equal."""
        if not isinstance(other, SidecarServerConfig):
            return False
        # ignores sidecar_rank due to grouping
        return (
            self.group == sorted(other.group)
            and self.tensor_dtype == other.tensor_dtype
            and self.slab_numel == other.slab_numel
            and self.send_slot_numel == other.send_slot_numel
            and self.recv_slot_numel == other.recv_slot_numel
            and self.concurrent_copy == other.concurrent_copy
        )

    def sender_config(self) -> SidecarSenderConfig:
        """Create the sender config."""
        full_tensor, slab = init_shmem(
            filename=shm_filename(),
            local_ranks=list(range(self.node_info.get_sidecar_num())),
            num_local_sidecars=self.node_info.get_sidecar_num(),
            partition_numel=self.slab_numel * 2,
            dtype=self.tensor_dtype,
        )
        return SidecarSenderConfig(
            sidecar_rank=self.sidecar_rank,
            node_info=self.node_info,
            peers=self.peers,
            group=self.group,
            base_ptr=full_tensor.data_ptr(),
            shared_tensor=slab[: slab.numel() // 2],
            slot_numel=self.send_slot_numel,
            concurrent_copy=self.concurrent_copy,
        )

    def receiver_config(self) -> SidecarReceiverConfig:
        """Create the receiver config."""
        full_tensor, slab = init_shmem(
            filename=shm_filename(),
            local_ranks=list(range(self.node_info.get_sidecar_num())),
            num_local_sidecars=self.node_info.get_sidecar_num(),
            partition_numel=self.slab_numel * 2,
            dtype=self.tensor_dtype,
        )
        return SidecarReceiverConfig(
            sidecar_rank=self.sidecar_rank,
            node_info=self.node_info,
            peers=self.peers,
            group=self.group,
            base_ptr=full_tensor.data_ptr(),
            shared_tensor=slab[slab.numel() // 2 :],
            slot_numel=self.send_slot_numel,
        )


@dataclass
class SendTransferRequestState:
    """Internal data structure to keep track of a tansfer request's state.

    Attributes:
        id: The concatenation of request_id and data_id
        buffer: The shared memory buffer used to recv the data
        shards_sent: A list of flags to indicate if each shard is sent
        done: A flag to indicate if the transfer is fully sent
    """

    id: str
    buffer: SharedMemoryBuffer
    shards_sent: list[bool]
    done: bool = False


@dataclass
class RecvTransferRequestState:
    """Internal data structure to keep track of a tansfer request's state.

    Attributes:
        id: The concatenation of request_id and data_id
        buffer: The shared memory buffer used to recv the data
        intra_node_rank: The intra node source sidecar rank
        done: A flag to indicate if the transfer is done
    """

    id: str
    buffer: SharedMemoryBuffer
    intra_node_rank: int = -1
    done: bool = False


@dataclass
class RecvTensorState:
    """Internal data structure to keep track of a chunk of tensor's transfer state.

    Attributes:
        id: The chunk id
        buffer: The shared memory buffer used to recv the tensor
        intra_node_rank: The intra node source sidecar rank
        done: A flag to indicate if the transfer is done
    """

    id: int
    buffer: SharedMemoryBuffer
    intra_node_rank: int = -1
    done: bool = False


@dataclass
class RecvObjState:
    """Internal data structure to keep track of a chunk of obj's transfer state.

    Attributes:
        id: The chunk id
        data: Any deserialized data
    """

    id: int
    data: Any


@dataclass
class RecvRequestState:
    """Internal data structure to keep track of a transfer request's state.

    Attributes:
        id: The request id of the receive request
        num_chunks: The number of chunks in the request
        chunks: A dictionary of chunk id to RecvChunkState
    """

    id: str
    num_chunks: int = -1
    chunks: dict[int, RecvTensorState | RecvObjState] = field(default_factory=dict)
    done_event: asyncio.Event = field(default_factory=asyncio.Event)
