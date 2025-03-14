"""Sidecar api to be usd by the task exucutors: Enc Server, vLLM Server, etc.

TensorSidecarSender is used by the producer task executors to send data to a sidecar
consumer, all methods are blocking.
TensorSidecarAsyncReceiver is used by the consumer task executors to receive data from
a sidecar producer, all methods are async.
"""

from __future__ import annotations

import os
import pickle
import weakref
import threading
from functools import reduce
from operator import mul
from typing import List, Optional, cast
from concurrent.futures import ThreadPoolExecutor

import grpc
import torch

from cornserve.logging import get_logger
from cornserve.services.pb import comm_sidecar_pb2, comm_sidecar_pb2_grpc, common_pb2

from .shm_manager import SharedMemoryBuffer, SharedMemoryManager
from .utils import (
    TensorLayout,
    device_from_rank,
    grpc_channel_from_rank,
    init_shmem,
    shm_fn,
)

logger = get_logger(__name__)


class TensorSidecarReceiverBase:
    """Factory class for sidecar receiver.

    Receiver doesn't need to store device information, as it only reads from the shared memory buffer.

    Attributes:
        shm_size: The shared memory size (number of elements of given dtype).
        dtype: The data type of the tensor to be received.
        shape: The shape of the tensor to be received, must be (-1, ...)
    """

    def __init__(self, sidecar_rank: int, peers: list[int], shape: tuple[int, ...], dtype: torch.dtype) -> None:
        """Initialize the sidecar receiver client that receives data from a sender sidecar.

        Args:
            sidecar_rank: The global rank of the sidecar server.
            peers: The sidecar_ranks of the TP group to receive the tensor.
            shape: The shape of the tensor to be received, should be (-1, slot_shape), where
                slot_shape is used in the shared memory buffer.
            dtype: The data type of the tensor to be received.
            local_ranks: The local ranks of the sidecar servers.
        """
        if shape[0] != -1:
            raise ValueError("The first dimension of the shape should be -1")

        self.sidecar_rank = sidecar_rank
        self.peers = peers
        self.dtype = dtype
        self.tensor_shape = shape

        # register the to the sidecar server
        # TODO: change to group
        self.channel = grpc.insecure_channel(grpc_channel_from_rank(self.sidecar_rank))
        self.stub = comm_sidecar_pb2_grpc.CommSidecarStub(self.channel)

        slot_size = -1 * reduce(mul, self.tensor_shape)
        request = comm_sidecar_pb2.RegisterReceiverRequest(
            slot_size=slot_size,
            dtype=str(self.dtype).split(".")[-1],
            peers=peers,
        )
        response = self.stub.RegisterReceiver(request)
        assert response.shm_size > 0, "Failed to register sidecar"
        # set up shared memory buffer
        self.shared_tensor = init_shmem(
            shm_fn(),
            response.local_ranks,
            response.num_local_sidecars,
            response.shm_size,
            self.dtype,
        )

        # this is total size
        self.shm_size = response.shm_size * len(response.local_ranks)


class TensorSidecarAsyncReceiverController(TensorSidecarReceiverBase):
    """Async receiver client for interacting with the sidecar receiver server.

    This class lives in the control plane.
    """

    pass


class TensorSidecarReceiverExecutor:
    """Receiver client that reads data from the shared memory buffer.

    This class lives in the data plane.
    """

    def __init__(self, sidecar_rank: int, shape: tuple[int, ...], dtype: torch.dtype) -> None:
        """Initialize the sidecar receiver client that only reads the data."""
        self.sidecar_rank = sidecar_rank
        self.channel = grpc.insecure_channel(grpc_channel_from_rank(self.sidecar_rank))
        self.stub = comm_sidecar_pb2_grpc.CommSidecarStub(self.channel)
        self.dtype = dtype
        self.tensor_shape = shape
        self._finalizer = weakref.finalize(self, self.shutdown)

        request = comm_sidecar_pb2.RegisterReaderRequest()
        response = self.stub.RegisterReader(request)

        self.shared_tensor = init_shmem(
            shm_fn(),
            response.local_ranks,
            response.num_local_sidecars,
            response.shm_size,
            self.dtype,
        )

    def recv(self, id: str) -> torch.Tensor:
        """Read the tensor corresponding to the given id from the shared memory buffer."""
        recv_req = comm_sidecar_pb2.ReceiveRequest(id=id)
        response = self.stub.Receive(recv_req)
        assert response.offset >= 0 and response.size >= 0, "Failed to receive data"
        chunk = self.shared_tensor[response.offset : response.offset + response.size]
        return chunk.view(self.tensor_shape)

    def shutdown(self) -> None:
        """Unlink the shared memory buffer."""
        try:
            del self.shared_tensor
            os.unlink(shm_fn())
        except Exception:
            pass


class TensorSidecarAsyncReceiver(TensorSidecarReceiverBase):
    """Async receiver client for interacting with the sidecar receiver server.

    A legacy class that is not used in the current codebase, but it is kept for reference.
    """

    def __init__(
        self,
        sidecar_rank: int,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        peers: Optional[list[int]] = None,
    ) -> None:
        """Constructor for the sidecar receiver.

        Args:
            sidecar_rank: The rank of the sidecar server.
            device_id: The device rank the the receiver will use.
            shape: The shape expected for the received tensor, should be (-1, hidden_size).
                hidden_size will be used as the slot size in the shared memory buffer.
            dtype: The data type of the tensor to be received.
            peers: The sidecar ranks of the TP group to receiver the tensor.
        """
        if peers is None:
            peers = [sidecar_rank]
        super().__init__(sidecar_rank, peers, shape, dtype)

        # casting is to make pyright happy
        cast(grpc.Channel, self.channel).close()
        # overwrite the channel and the stub with async ones
        # we always talk to the controller sidecar
        controller_rank = min(peers)
        self.channel = grpc.aio.insecure_channel(grpc_channel_from_rank(controller_rank))
        self.stub = comm_sidecar_pb2_grpc.CommSidecarStub(self.channel)
        self._finalizer = weakref.finalize(self, self.__del__)

    def __del__(self) -> None:
        """Clean up the channels and unlink the shared memory buffer."""
        if not hasattr(self, "channel"):
            return
        logger.warning("Sidecar receiver not shutdown properly, remember to call shutdown")
        try:
            del self.channel
            del self.shared_tensor
            os.unlink(shm_fn())
        except Exception:
            pass

    async def shutdown(self) -> None:
        """Close the gRPC channel."""
        try:
            await self.channel.close()
            del self.channel
            del self.shared_tensor
            os.unlink(shm_fn())
        except Exception:
            pass

    async def recv(self, id: str) -> torch.Tensor:
        """Receive a tensor from the sidecar server.

        Args:
            id: the id of expected data, should be the concatenation of request id and data id.
        """
        recv_req = comm_sidecar_pb2.ReceiveRequest(id=id)
        response = await self.stub.Receive(recv_req)
        assert response.offset >= 0 and response.size >= 0, "Failed to receive data"
        chunk = self.shared_tensor[response.offset : response.offset + response.size]
        return chunk.view(self.tensor_shape)

    async def mark_done(self, id: str) -> None:
        """Mark a tensor as done in the sidecar server, which will free the shared memory buffer."""
        request = comm_sidecar_pb2.MarkDoneRequest(id=id)
        response = await self.stub.MarkDone(request)
        if response.status == common_pb2.Status.STATUS_OK:
            logger.debug("Request %s marked done", id)


class TensorSidecarSenderBase:
    """Factory class for sidecar sender."""

    def __init__(  # noqa: PLR0913
        self,
        sidecar_rank: int,
        slot_shape: tuple[int, ...],
        dtype: torch.dtype,
        shard_rank: int = 0,
        num_shards: int = 1,
        layout: TensorLayout = TensorLayout.FULL,
    ) -> None:
        """Initialize the sidecar sender client that sends data to a receiver sidecar.

        Attributes:
            shm_size: The shared memory size (number of elements of given dtype).
        """
        self.sidecar_rank = sidecar_rank
        self.dtype = dtype
        assert shard_rank < num_shards, "Invalid shard rank"
        self.slot_shape = slot_shape
        self.slot_size = reduce(mul, self.slot_shape)
        self.shard_rank = shard_rank
        self.num_shards = num_shards
        self.layout = layout
        # will be set in _register
        self.shm_size = -1

        # Register the sender to the sidecar server
        logger.debug("Registering sidecar to %s", grpc_channel_from_rank(self.sidecar_rank))
        self.channel = grpc.insecure_channel(grpc_channel_from_rank(self.sidecar_rank))
        self.stub = comm_sidecar_pb2_grpc.CommSidecarStub(self.channel)
        request = comm_sidecar_pb2.RegisterSenderRequest(
            slot_size=self.slot_size,
            dtype=str(self.dtype).split(".")[-1],
            shard_rank=self.shard_rank,
            num_shards=self.num_shards,
            layout=self.layout.value,
        )

        response = self.stub.RegisterSender(request)
        assert response.shm_size > 0, "Failed to register sidecar"
        self.shm_size = response.shm_size
        self.shm_fn = shm_fn()
        self.shared_tensor = init_shmem(
            self.shm_fn,
            response.local_ranks,
            response.num_local_sidecars,
            self.shm_size,
            self.dtype,
        )
        logger.debug("Registered sidecar with host device cuda:%d", response.local_ranks[0])
        # Post initialization, set GPU device and shared memory buffer
        self.device = device_from_rank(self.shard_rank)
        # cast to make pyright happy
        self.stream = cast(torch.cuda.Stream, torch.cuda.Stream(device=self.device))
        # this is used to keep track of the memory pressure events
        self.mem_pressure_count = 0

        self._finalizer = weakref.finalize(self, self.shutdown)

    def __del__(self) -> None:
        """Clean up, unlink the shared memory buffer."""
        if not hasattr(self, "shared_tensor"):
            return
        try:
            del self.shared_tensor
            os.unlink(shm_fn())
        except Exception:
            pass

    def shutdown(self) -> None:
        """Unlink the shared memory buffer."""
        try:
            del self.shared_tensor
            os.unlink(shm_fn())
        except Exception:
            pass


class TensorSidecarSender(TensorSidecarSenderBase):
    """Sender client for interacting with the sidecar sender server. All methods are blocking.

    Chunk is the logical smallest unit of data during transmission.
    Slot is the physical smallest unit of data during transmission.
    """

    def __init__(  # noqa: PLR0913
        self,
        sidecar_rank: int,
        slot_shape: tuple[int, ...],
        dtype: torch.dtype,
        shard_rank: int = 0,
        num_shards: int = 1,
        layout: TensorLayout = TensorLayout.FULL,
        max_workers: int = 8,
        report_interval: int = 100,
    ) -> None:
        """Initialize the a sidecar sender client that instructs the sidecar server to send data.

        This client should be instantiated by the producer task executors, e.g. Erics.
        Chunk is the LOGICAL smallest unit of data during transmission.
        Slot is the PHYSICAL smallest unit of data during transmission.

        Transfer Data:
        Chunks is called over multiple `send`, it is expected that every chunk has the same size(shape).
        There is no mechanism to detect the size mismatch. Chunking is to support fixed resolution ViTs,
        directly send the whole tensor if using dynamic resolution ViTs.

        |-----CHUNK0-----|-----CHUNK1-----|-----CHUNK2-----|-----CHUNK3-----|
           send(chunk0)     send(chunk1)     send(chunk2)     send(chunk3)

        Each send shards the chunk during transmission. Sharding is oblivious to the caller,
        and shard sizes could be imbalanced by maximum of 1 slot.
        send(chunk0):
            Shard RANK0: |-SAHRD0-|-------| -> Receiver
            Shard RANK1: |--------|-SHRAD1| -> Receiver

        Args:
            sidecar_rank: The rank of the sidecar server.
            slot_shape: The slot shape (size) used in the shared memory buffer,
                typically the embedding size one image token or a multiple of it.
                The slot_shape should align with encoder's hidden size, even for fixed resolution ViTs.
                Note if you use a tile shape as the slot_shape for fixed resolution ViTs, all `send`
                will dont by the `shard_rank=0` sender due to sharding imbalance.
            dtype: The data type of the chunk.
            shard_rank: The rank of the shard this sender will send, this is the same as tp_rank.
            num_shards: The number of shards, this is the same as tp_size.
            layout: The layout of the tensor, currently only support slicing over the first dimension.
            max_workers: The maximum number of worker threads.
            report_interval: The interval (s) to report memory pressure.
        """
        super().__init__(sidecar_rank, slot_shape, dtype, shard_rank, num_shards, layout)
        self.shm_manager = SharedMemoryManager(self.shared_tensor, self.slot_size)
        self.memory_lock = threading.Lock()
        self.memory_freed = threading.Condition(self.memory_lock)
        # we use one more worker to report memory pressure
        self.shutdown_event = threading.Event()
        self.worker_pool = ThreadPoolExecutor(max_workers=max_workers + 1, thread_name_prefix="sidecar-sender")

        # create a background thread to report memory pressure periodically
        def _report_memory_pressure():
            import time

            while not self.shutdown_event.is_set():
                logger.debug("Sidecar sender %d Memory pressure count %d", self.sidecar_rank, self.mem_pressure_count)
                request = comm_sidecar_pb2.ReportMemoryRequest(pressure=self.mem_pressure_count)
                response = self.stub.ReportMemory(request)
                if response.status != common_pb2.Status.STATUS_OK:
                    logger.error("Sidecar server is unhealthy")
                for _ in range(report_interval):
                    if self.shutdown_event.is_set():
                        break
                    time.sleep(1)

        self.worker_pool.submit(_report_memory_pressure)

    def shutdown(self) -> None:
        """Shutdown the sender client."""
        self.shutdown_event.set()
        self.worker_pool.shutdown(wait=True, cancel_futures=True)
        try:
            del self.shared_tensor
            os.unlink(shm_fn())
        except Exception:
            pass

    def __del__(self) -> None:
        """Clean up worker threads and the shared memory buffer."""
        try:
            if not self.shutdown_event.is_set():
                self.shutdown()
        except Exception:
            pass

    def _allocate(self, size: int) -> SharedMemoryBuffer:
        """Allocate a shared memory buffer in thread safe."""
        with self.memory_lock:
            while True:
                buffer = self.shm_manager.allocate(size)
                if buffer is not None:
                    logger.debug(
                        "Allcoated buffer of size %d using %d slots, free slots left %d",
                        size,
                        len(buffer.slots),
                        self.shm_manager.free_slots(),
                    )
                    self.mem_pressure_count += 1
                    return buffer
                logger.warning(
                    "SHARD RANK %d: Shared memory pressure, try increasing shm_size. Free slots %d/%d",
                    self.shard_rank,
                    self.shm_manager.free_slots(),
                    self.shm_manager.num_slots,
                )
                self.memory_freed.wait()

    def _free(self, buffer: SharedMemoryBuffer) -> None:
        """Free a shared memory buffer in thread safe."""
        with self.memory_lock:
            self.shm_manager.free(buffer)
            self.memory_freed.notify_all()

    def send(
        self,
        chunk: torch.Tensor,
        id: str,
        dst_sidecar_ranks: List[int],
        chunk_id: int = 0,
        num_chunks: int = 1,
    ) -> None:
        """Schedules a background thread that sends a tensor to destination sidecar ranks.

        TODO: microbenchmark overhead to determine if we need to batch multiple chunks.

        Args:
            chunk: the tensor to be sent, must be a multiple of self.slot_size,
                the chunk will be flattened and sharded.
            id: the concatenation of the request id and the data id.
                `data_id` is for images in the same request, for chunking
                an image in fixed resolution ViTs, use chunk_id instead.
            dst_sidecar_ranks: the destination sidecar ranks.
            chunk_id: the chunk id of the data.
            num_chunks: the total number of chunks the data is split into.
        """
        assert all(dst_sidecar_rank != self.sidecar_rank for dst_sidecar_rank in dst_sidecar_ranks), (
            "Cannot send to self"
        )
        assert all(dst_sidecar_rank >= 0 for dst_sidecar_rank in dst_sidecar_ranks), "Invalid sidecar rank"
        assert len(id) % 2 == 0, "id must be the concatenation of request id and data id"
        assert chunk.device == self.device, "Device mismatch"
        size = chunk.numel()
        assert size % self.slot_size == 0, "Chunk size should be a multiple of slot size"
        chunk = chunk.view(-1)

        # shard based on the shard rank and num shards
        num_slots = size // self.slot_size
        quotient, remainder = divmod(num_slots, self.num_shards)
        start_slot = self.shard_rank * quotient + min(self.shard_rank, remainder)
        end_slot = start_slot + quotient + (1 if self.shard_rank < remainder else 0)
        if start_slot == end_slot:
            # no responsibility to send
            return

        logger.debug(
            "SHARD RANK %d: %s:Chunk is using %d num_slots, shard %d, start_slot %d, end_slot %d, slot_size=%d",
            self.shard_rank,
            id,
            num_slots,
            self.shard_rank,
            start_slot,
            end_slot,
            self.slot_size,
        )
        shard = chunk[start_slot * self.slot_size : end_slot * self.slot_size]
        shard_offset = start_slot * self.slot_size
        logger.debug("SHRAD RANK %d: Shard %s", self.shard_rank, shard.shape)
        self.worker_pool.submit(
            self._send_worker,
            id=id,
            shard=shard,
            chunk_size=size,
            shard_offset=shard_offset,
            dst_sidecar_ranks=dst_sidecar_ranks,
            chunk_id=chunk_id,
            num_chunks=num_chunks,
        )

    def _send_worker(
        self,
        id: str,
        shard: torch.Tensor,
        chunk_size: int,
        shard_offset: int,
        dst_sidecar_ranks: List[int],
        chunk_id: int,
        num_chunks: int,
    ) -> None:
        """The worker function that sends a shard to destination sidecar ranks.

        Args:
            id: the id of the data transfer request
            shard: the shard to be sent, must be a 1D tensor.
            chunk_size: the complete size of the chunk, used by the server to allocate buffer on the receiver side.
            shard_offset: the offset of the shard in the data, passed in to avoid recomputation.
            dst_sidecar_ranks: the destination sidecar ranks.
            chunk_id: the chunk id of the data.
            num_chunks: the total number of chunks the data is split into.
        """
        buffer = self._allocate(shard.numel())
        try:
            cuda_event = torch.cuda.Event(interprocess=True)
            with torch.cuda.stream(self.stream):
                buffer.data.copy_(shard, non_blocking=True)
                cuda_event.record(self.stream)
            ipc_handle = cuda_event.ipc_handle()
            logger.debug("SHARD RANK: %d: Sending send request to sidecar", self.shard_rank)
            request = comm_sidecar_pb2.SendRequest(
                id=id,
                size=buffer.size,
                slot=buffer.slots[0],
                ipc_handle=pickle.dumps(ipc_handle),
                chunk_id=chunk_id,
                num_chunks=num_chunks,
                dst_ranks=dst_sidecar_ranks,
                chunk_size=chunk_size,
                shard_offset=shard_offset,
            )
            response = self.stub.Send(request)

            if response.status == common_pb2.Status.STATUS_OK:
                logger.debug("Sent shard %d of chunk %d in req %s successfully", self.shard_rank, chunk_id, id)
            else:
                logger.error("Failed to send data")
        except Exception:
            logger.exception("Failed to send data")
        finally:
            # used to simulate a delay for testing back pressure
            # import time
            # time.sleep(1)
            self._free(buffer)
