"""Sidecar server implementniation.

The sidecar server is a gRPC service that runs on each node in the cluster. This service
is the backend for the `SidecarSender` and `SidecarReceiver` classes in the `api` module.
It has two corresponding components, `CommSidecarSender` and `CommSidecarReceiver`, which
together provide the functionality to send and receive tensors between ranks.
"""

from __future__ import annotations
import asyncio
from dataclasses import dataclass
import os
import pickle
from typing import Dict

import grpc
import kubernetes_asyncio.client as kclient
import kubernetes_asyncio.config as kconfig
import torch
import torch.distributed as dist
import tyro
import multiprocessing as mp

from cornserve.logging import SidcarAdapter, get_logger
from cornserve.services.pb import comm_sidecar_pb2, comm_sidecar_pb2_grpc, common_pb2

from .shm_manager import SharedMemoryBuffer, SharedMemoryManager
from .utils import (
    TensorLayout,
    chunk_tag,
    device_from_rank,
    grpc_channel_from_rank,
    init_shmem,
    shm_fn,
)

logger = get_logger(__name__, [SidcarAdapter])
cleanup_coroutines = []


class CommSidecarReceiver:
    """The receiver sidecar server supports receiving tensors from other ranks using gloo backend."""

    @dataclass
    class TransferRequestState:
        """Internal data structure to keep track of a tansfer request's state.

        Attributes:
            - id: The concatenation of request_id and data_id
            - buffer: The shared memory buffer used to recv the data
            - done: A flag to indicate if the transfer is done
        """

        id: str
        buffer: SharedMemoryBuffer
        done: bool = False

    def __init__(
        self,
        sidecar_rank: int,
        peers: list[int],
        node_info: SidecarNodeInfo,
        shm_size: int,
        slot_size: int,
        dtype: str,
    ) -> None:
        """Initialize the receiver sidecar.

        Args:
            sidecar_rank: The sidecar rank, aka global rank.
            peers: The ranks of the TP group to receive tensors from.
            node_info: The node information.
            shm_size: The shared memory size (number of elements of given dtype).
            slot_size: The shape of the tensor to be received, currently fixed.
            dtype: The data type of the receiving tensor.
        """
        self.sidecar_rank = sidecar_rank
        self.peers = peers

        self.dtype = getattr(torch, dtype)
        self.shm_fn = shm_fn()
        self.node_info = node_info
        self.local_ranks = [self.node_info.get_device_id(i) for i in peers]
        self.shm_size = shm_size
        self.shared_tensor = init_shmem(
            self.shm_fn,
            local_ranks=self.local_ranks,
            num_local_sidecars=self.node_info.get_sidecar_num(),
            size=self.shm_size,
            dtype=self.dtype,
        )
        self.shm_manager = SharedMemoryManager(shm=self.shared_tensor, slot_size=slot_size)
        self.has_memory = asyncio.Condition()
        self.malloc_events: dict[str, asyncio.Event] = {}

        # a legder to keep the transfer status of each transfer request
        self.ledger: dict[str, CommSidecarReceiver.TransferRequestState] = {}
        # per req event, recieve will wait on this event, recv_task will try to set this event
        self.req_events: dict[str, asyncio.Event] = {}

        # we use a multiprocessing lock to protect the done flag, as this lock is used in the recv_task,
        # which is running in a separate thread to avoid blocking on dist.recv
        self.recv_done_lock = mp.Lock()

        # this is used to keep track of the memory pressure events
        self.mem_pressure_count = 0

    async def shutdown(self):
        """Cleanup routines for the receiver."""
        # remove the shared memory file, used async to unify the interface
        del self.shared_tensor
        del self.shm_manager
        os.unlink(self.shm_fn)

    async def prepare_receive(
        self,
        request: comm_sidecar_pb2.PrepareReceiveRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.PrepareReceiveResponse:
        """Prepare to receive a tensor from another rank, called by the sender sidecar server.

        This function allocates a shared memory buffer if not already allocated,
        and queues up a receive task to receive the tensor.
        """
        logger.info(
            (
                "Prepare receive for request id %s, shard_size %d, dtype %s, src_rank %d, shard_rank %d, "
                "num_shards %d, chunk_size %d, num_chunks %d, chunk_id %d, shard_offset %d"
            ),
            request.id,
            request.shard_size,
            request.dtype,
            request.src_rank,
            request.shard_rank,
            request.num_shards,
            request.chunk_size,
            request.num_chunks,
            request.chunk_id,
            request.shard_offset,
        )
        dtype = getattr(torch, request.dtype)
        if self.dtype != dtype:
            logger.error("Data type mismatch")
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Data type mismatch")

        async with self.has_memory:
            # acquire the underlying lock
            if request.id not in self.ledger:
                # some prepare_recv needs to allocate the buffer and update the ledger
                if request.id not in self.malloc_events:
                    # this is the first prepare_recv call for request id
                    buffer = self.shm_manager.allocate(request.chunk_size * request.num_chunks)
                    # note: if this succeeds, self.ledger[request.id] will be created, so all future prepare_recv
                    # will not enter this if block
                    if buffer is None:
                        # this means all future prepare_recv will also fail
                        event = asyncio.Event()
                        self.malloc_events[request.id] = event

                    # keep retry
                    while buffer is None:
                        self.mem_pressure_count += 1
                        logger.info("Memory pressure detected, current prssure count %d", self.mem_pressure_count)
                        await self.has_memory.wait()
                        buffer = self.shm_manager.allocate(request.chunk_size * request.num_chunks)

                    buffer.create_chunks(request.num_chunks, request.num_shards)
                    self.ledger[request.id] = CommSidecarReceiver.TransferRequestState(request.id, buffer)

                    if request.id in self.malloc_events:
                        # wake up all the waiting prepare_recv calls
                        self.malloc_events[request.id].set()
                        del self.malloc_events[request.id]
                else:
                    # some previous prepare_recv call is blocking on the allocation
                    event = self.malloc_events[request.id]
                    self.has_memory.release()
                    try:
                        await event.wait()
                    finally:
                        # Make sure to re-acquire the lock after waiting
                        await self.has_memory.acquire()

        state = self.ledger[request.id]
        chunk = state.buffer.chunks[request.chunk_id]
        tag = chunk_tag(request.src_rank, request.chunk_id, request.shard_rank)

        # TODO: allow batch recv
        def recv_task():
            """The task to receive the tensor."""
            logger.info("Queuing recv task for chunk %d of request %s tag %d", request.chunk_id, request.id, tag)
            dist.recv(
                chunk.data[request.shard_offset : request.shard_offset + request.shard_size],
                src=request.src_rank,
                tag=tag,
            )
            chunk.mark_shard_ready(request.shard_rank, request.shard_size)
            logger.info(
                "Received shard %d of chunk %d of request %s tag %d",
                request.shard_rank,
                request.chunk_id,
                request.id,
                tag,
            )
            if chunk.ready:
                state.buffer.mark_chunk_ready(request.chunk_id)
            if state.buffer.is_ready():
                self.recv_done_lock.acquire()
                state.done = True
                if request.id in self.req_events:
                    self.req_events[request.id].set()
                self.recv_done_lock.release()

        asyncio.create_task(asyncio.to_thread(recv_task))
        return comm_sidecar_pb2.PrepareReceiveResponse(status=common_pb2.Status.STATUS_OK)

    async def receive(
        self,
        recv_req: comm_sidecar_pb2.ReceiveRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.ReceiveResponse:
        """Receive the tensor of a request from other ranks, returns a slot number in the shared memory.

        If all chunks are received, return the slot number imediately.
        Else, queues up an event for the request id and waits for all chunks to be received.
        """
        logger.info("==> Receive request for request id %s", recv_req.id)
        self.recv_done_lock.acquire()
        if recv_req.id in self.ledger and self.ledger[recv_req.id].done:
            self.recv_done_lock.release()
        else:
            # still waiting for chunks/shards
            event = asyncio.Event()
            self.req_events[recv_req.id] = event
            self.recv_done_lock.release()
            await event.wait()

        logger.info("==> All chunks received for request id %s", recv_req.id)
        offset = self.ledger[recv_req.id].buffer.slots[0] * self.shm_manager.slot_size
        size = self.ledger[recv_req.id].buffer.size
        return comm_sidecar_pb2.ReceiveResponse(offset=offset, size=size)

    async def mark_done(
        self,
        mark_done_req: comm_sidecar_pb2.MarkDoneRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.MarkDoneResponse:
        """Mark a tensor as consumed, free up the shared memory used."""
        if mark_done_req.id not in self.ledger:
            await context.abort(grpc.StatusCode.NOT_FOUND, "mark_done_req not found")
        logger.info(
            "mark_done: Freeing up %d slots from %s",
            len(self.ledger[mark_done_req.id].buffer.slots),
            mark_done_req.id,
        )
        async with self.has_memory:
            self.shm_manager.free(self.ledger[mark_done_req.id].buffer)
            self.has_memory.notify_all()
        del self.ledger[mark_done_req.id]
        if mark_done_req.id in self.req_events:
            del self.req_events[mark_done_req.id]
        return comm_sidecar_pb2.MarkDoneResponse(status=common_pb2.Status.STATUS_OK)


class CommSidecarSender:
    """Sidecar sender gRPC service backend.

    Implements the gRPC service for the sender sidecar.
    """

    def __init__(  # noqa: PLR0913
        self,
        sidecar_rank: int,
        node_info: SidecarNodeInfo,
        shm_size: int,
        slot_size: int,
        dtype: torch.dtype,
        shard_rank: int = 0,
        num_shards: int = 1,
        layout: TensorLayout = TensorLayout.FULL,
    ) -> None:
        """Initialize the sender sidecar server.

        Args:
            gpu_rank: The local GPU rank.
            sidecar_rank: The sidecar rank, aka global rank.
            shm_size: The shared memory size (number of elements of given dtype).
            slot_size: The slot_size of the shared memory buffer.
            dtype: The data type of the sending tensor.
            shard_rank: The rank of the shard, default to 0.
            num_shards: The number of shards, default to 1.
            layout: The layout of the tensor, default to FULL.
            node_info: The node information.
        """
        self.sidecar_rank = sidecar_rank
        self.node_info = node_info
        self.local_rank = self.node_info.get_device_id(self.sidecar_rank)
        self.shm_size = shm_size
        self.dtype = dtype
        self.slot_size = slot_size
        self.shard_rank = shard_rank
        self.num_shards = num_shards
        self.layout = layout
        self.device = device_from_rank(self.local_rank)
        self.shared_tensor = init_shmem(
            fn=shm_fn(),
            local_ranks=[self.local_rank],
            num_local_sidecars=self.node_info.get_sidecar_num(),
            size=shm_size,
            dtype=self.dtype,
        )

        self.dst_channels: Dict[int, grpc.aio.Channel] = {}
        self.dst_stubs: Dict[int, comm_sidecar_pb2_grpc.CommSidecarStub] = {}
        self.mem_pressure_count = 0

    async def report_memory(
        self, request: comm_sidecar_pb2.ReportMemoryRequest, context: grpc.aio.ServicerContext
    ) -> comm_sidecar_pb2.ReportMemoryResponse:
        """Updates the memory pressure count."""
        self.mem_pressure_count = request.pressure
        return comm_sidecar_pb2.ReportMemoryResponse(status=common_pb2.Status.STATUS_OK)

    async def shutdown(self) -> None:
        """Cleanup routines for the sender sidecar."""
        for channel in self.dst_channels.values():
            await channel.close()
        # remove the shared memory file
        del self.shared_tensor
        os.unlink(shm_fn())

    async def send(
        self,
        send_request: comm_sidecar_pb2.SendRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.SendResponse:
        """Send a tensor to another rank.

        First use prepare_receive to send control signals to the destination sidecar,
        then queue up the send tasks.
        """
        # sanity check
        if send_request.slot < 0 or send_request.slot * self.slot_size + send_request.size > self.shm_size:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"slot out of range {send_request.slot}*{self.slot_size}+{send_request.size} {self.shm_size}",
            )

        if any(r < 0 or r == self.sidecar_rank for r in send_request.dst_ranks):
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid destination rank")
        # only send to the head receiver when TP is enabled (min sidecar rank)
        dst_rank = min(send_request.dst_ranks)

        # inform destination sidecar
        # lazily create channel
        if dst_rank not in self.dst_channels:
            self.dst_channels[dst_rank] = grpc.aio.insecure_channel(grpc_channel_from_rank(dst_rank))
            self.dst_stubs[dst_rank] = comm_sidecar_pb2_grpc.CommSidecarStub(self.dst_channels[dst_rank])
            logger.info("Connected to sidecar-%d", dst_rank)

        stub = self.dst_stubs[dst_rank]
        logger.info(
            "Calling prepare receive on sidecar-%d for request %s chunk id %s out of %d",
            dst_rank,
            send_request.id,
            send_request.chunk_id,
            send_request.num_chunks,
        )
        prepare_receive_request = comm_sidecar_pb2.PrepareReceiveRequest(
            id=send_request.id,
            shard_size=send_request.size,
            chunk_size=send_request.chunk_size,
            chunk_id=send_request.chunk_id,
            num_chunks=send_request.num_chunks,
            dtype=str(self.dtype).split(".")[-1],
            src_rank=self.sidecar_rank,
            shard_rank=self.shard_rank,
            num_shards=self.num_shards,
            shard_offset=send_request.shard_offset,
            layout=self.layout.value,
        )
        response = await stub.PrepareReceive(prepare_receive_request)
        if response.status != common_pb2.Status.STATUS_OK:
            logger.error("Failed to prepare receive")
            # TODO: clean up by canceling the previous prepare_receive calls
            return comm_sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_ERROR)

        ipc_handle = pickle.loads(send_request.ipc_handle)
        cuda_event = torch.cuda.Event.from_ipc_handle(self.device, ipc_handle)

        while not cuda_event.query():
            await asyncio.sleep(0.0)  # Allows other futures to make process.

        logger.info("Sending chunk %d for req %s", send_request.chunk_id, send_request.id)
        tag = chunk_tag(self.sidecar_rank, send_request.chunk_id, self.shard_rank)
        req = dist.isend(
            self.shared_tensor[
                send_request.slot * self.slot_size : send_request.slot * self.slot_size + send_request.size
            ],
            dst=dst_rank,
            tag=tag,
        )
        if req is None:
            logger.error("Failed to send chunk %d for request %s", send_request.chunk_id, send_request.id)
            return comm_sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_ERROR)

        def wait_fn():
            req.wait()

        await asyncio.to_thread(wait_fn)
        logger.info(
            "SHARD RANK %d: sent chunk %d for request %s", self.shard_rank, send_request.chunk_id, send_request.id
        )

        return comm_sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_OK)


class CommSidecarServicer(comm_sidecar_pb2_grpc.CommSidecarServicer):
    """A unified wrapper for both sender and receiver sidecar servers. Entry point for the gRPC service."""

    def __init__(self, sidecar_rank: int, mem_pressure_threshold=500) -> None:
        """Initialize the sidecar service.

        This creates an offline sidecar server that only has the CheckHealth endpoint available.

        Args:
            sidecar_rank: The global rank of the sidecar.
            mem_pressure_threshold: The threshold of memory pressure count to trigger the memory pressure status.
        """
        self.sidecar_rank = sidecar_rank
        self.sidecar: CommSidecarSender | CommSidecarReceiver | None = None
        self.live = False
        self.mem_pressure_threshold = mem_pressure_threshold

    def online(self, node_info: SidecarNodeInfo, shm_size: int) -> None:
        """Mark the sidecar as online.

        Args:
            node_info: The sidecar information within the node.
            shm_size: The size of the shared memory buffer used by each sidecar server.
        """
        self.node_info = node_info
        self.device_id = self.node_info.get_device_id(self.sidecar_rank)
        self.num_devices = self.node_info.get_sidecar_num()
        self.shm_size = shm_size
        self.live = True
        logger.info("Sidecar online")

    def add_mapping(self, mapping: dict[int, int]) -> None:
        """Adds a mapping of global rank to local rank."""
        self.mapping = mapping

    async def CheckHealth(  # noqa: N802
        self,
        request: comm_sidecar_pb2.CheckHealthRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.CheckHealthResponse:
        """Health check for the sidecar."""
        if not self.live or self.sidecar is None:
            return comm_sidecar_pb2.CheckHealthResponse(status=comm_sidecar_pb2.HealthStatus.HEALTH_OFFLINE)
        if self.sidecar.mem_pressure_count > self.mem_pressure_threshold:
            return comm_sidecar_pb2.CheckHealthResponse(status=comm_sidecar_pb2.HealthStatus.HEALTH_MEMORY_PRESSURE)
        return comm_sidecar_pb2.CheckHealthResponse(status=comm_sidecar_pb2.HealthStatus.HEALTH_ALL_GOOD)

    async def RegisterSender(  # noqa: N802
        self,
        request: comm_sidecar_pb2.RegisterSenderRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.RegisterResponse:
        """Called by the sender server to register the sidecar."""
        if not self.live:
            logger.error("Sidecar not online")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not online")

        if self.sidecar is not None:
            logger.warning("Overwriting existing sidecar")

        dtype = getattr(torch, request.dtype)
        # calculate the number of shared elements to return
        shm_size = self.shm_size // dtype.itemsize

        self.sidecar = CommSidecarSender(
            sidecar_rank=self.sidecar_rank,
            shm_size=shm_size,
            slot_size=request.slot_size,
            dtype=dtype,
            shard_rank=request.shard_rank,
            num_shards=request.num_shards,
            node_info=self.node_info,
            layout=TensorLayout(request.layout),
        )

        # gpu_rank is local rank, which is the gpu_rank on the host
        logger.info("Registered sender of gpu_rank %s, sidecar_rank %s", self.device_id, self.sidecar_rank)

        return comm_sidecar_pb2.RegisterResponse(
            shm_size=shm_size,
            local_ranks=[self.device_id],
            num_local_sidecars=self.num_devices,
        )

    async def RegisterReceiver(  # noqa: N802
        self,
        request: comm_sidecar_pb2.RegisterReceiverRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.RegisterResponse:
        """Called by the receiver server to register the sidecar."""
        if not self.live:
            logger.error("Sidecar not online")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not online")
        if self.sidecar is not None:
            logger.warning("Overwriting existing sidecar")

        if self.sidecar_rank not in request.peers:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid peers rank")
        for r in request.peers:
            if not self.node_info.contains(r):
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid peers rank")

        dtype = getattr(torch, request.dtype)
        shm_size = self.shm_size // dtype.itemsize

        self.sidecar = CommSidecarReceiver(
            sidecar_rank=self.sidecar_rank,
            peers=list(request.peers),
            shm_size=shm_size,
            slot_size=request.slot_size,
            dtype=request.dtype,
            node_info=self.node_info,
        )
        logger.info(
            "Registered receiver of gpu_rank %s, sidecar_rank %s, slot_size %d, dtype %s",
            self.device_id,
            self.sidecar_rank,
            request.slot_size,
            request.dtype,
        )

        return comm_sidecar_pb2.RegisterResponse(
            shm_size=shm_size,
            local_ranks=[self.node_info.get_device_id(i) for i in request.peers],
            num_local_sidecars=self.num_devices,
        )

    async def RegisterReader(  # noqa: N802
        self,
        request: comm_sidecar_pb2.RegisterReaderRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.RegisterResponse:
        """Register a read-only sidecar. This is temporary."""
        if not self.live or self.sidecar is None:
            logger.error("Sidecar not online")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not online")

        if not isinstance(self.sidecar, CommSidecarReceiver):
            logger.error("Invalid sidecar mode")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Invalid sidecar mode")

        logger.info("Registered reader of sidecar_rank %s", self.sidecar_rank)
        return comm_sidecar_pb2.RegisterResponse(
            shm_size=self.sidecar.shm_size,
            local_ranks=self.sidecar.local_ranks,
            num_local_sidecars=self.num_devices,
        )

    async def Send(  # noqa: N802
        self, request: comm_sidecar_pb2.SendRequest, context: grpc.aio.ServicerContext
    ) -> comm_sidecar_pb2.SendResponse:
        """Called by the sender server to send a tensor to some other rank."""
        if self.sidecar is None:
            logger.error("Sidecar not registered")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not registered")
        if not isinstance(self.sidecar, CommSidecarSender):
            logger.error("Invalid sidecar mode")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Invalid sidecar mode")
        return await self.sidecar.send(request, context)

    async def PrepareReceive(  # noqa: N802
        self,
        request: comm_sidecar_pb2.PrepareReceiveRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.PrepareReceiveResponse:
        """Called by the sender sidercar to the receiver sidecar to prepare receiving a tensor."""
        if not self.live:
            logger.error("Sidecar not online")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not online")
        if self.sidecar is None:
            logger.error("Sidecar not registered")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not registered")
        if not isinstance(self.sidecar, CommSidecarReceiver):
            logger.error("Invalid sidecar mode")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Invalid sidecar mode")
        return await self.sidecar.prepare_receive(request, context)

    async def Receive(  # noqa: N802
        self,
        request: comm_sidecar_pb2.ReceiveRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.ReceiveResponse:
        """Initiate receiving a tensor from some other rank."""
        if not self.live:
            logger.error("Sidecar not online")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not online")
        if self.sidecar is None:
            logger.error("Sidecar not registered")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not registered")
        if not isinstance(self.sidecar, CommSidecarReceiver):
            logger.error("Invalid sidecar mode")
            return comm_sidecar_pb2.ReceiveResponse(offset=-1, size=-1)

        return await self.sidecar.receive(request, context)

    async def MarkDone(  # noqa: N802
        self,
        request: comm_sidecar_pb2.MarkDoneRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.MarkDoneResponse:
        """Called by the receiver server to mark a request as done."""
        if not self.live:
            logger.error("Sidecar not online")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not online")
        if self.sidecar is None:
            logger.error("Sidecar not registered")
            return comm_sidecar_pb2.MarkDoneResponse(status=common_pb2.Status.STATUS_ERROR)
        if not isinstance(self.sidecar, CommSidecarReceiver):
            logger.error("Invalid sidecar mode")
            return comm_sidecar_pb2.MarkDoneResponse(status=common_pb2.Status.STATUS_ERROR)

        return await self.sidecar.mark_done(request, context)

    async def shutdown(self):
        """Shutdown the sidecar."""
        if self.sidecar is not None:
            await self.sidecar.shutdown()

    async def ReportMemory(
        self,
        request: comm_sidecar_pb2.ReportMemoryRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.ReportMemoryResponse:
        """Report memory pressure to the sidecar."""
        if not self.live:
            logger.error("Sidecar not online")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not online")
        if self.sidecar is None:
            logger.error("Sidecar not registered")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not registered")
        if not isinstance(self.sidecar, CommSidecarSender):
            logger.error("Invalid sidecar mode")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Invalid sidecar mode")
        return await self.sidecar.report_memory(request, context)


NAMESPACE = "cornserve"


# To allow grouping, we need to bookkeep the mapping between global rank and local rank
@dataclass
class SidecarNodeInfo:
    """Local Sidecar status within node."""

    sidecar_ranks: list[int]

    def get_device_id(self, sidecar_rank: int) -> int:
        """Get the device id of the sidecar, the same as local rank."""
        return self.sidecar_ranks.index(sidecar_rank)

    def get_sidecar_num(self) -> int:
        """Get the number of sidecars on the node."""
        return len(self.sidecar_ranks)

    def contains(self, sidecar_rank: int) -> bool:
        """Check if the sidecar rank is in the node."""
        return sidecar_rank in self.sidecar_ranks


async def _get_node_info(pod_name: str) -> SidecarNodeInfo | None:
    kconfig.load_incluster_config()
    async with kclient.ApiClient() as api_client:
        v1 = kclient.CoreV1Api(api_client)
        pod = await v1.read_namespaced_pod(name=pod_name, namespace=NAMESPACE)  # pyright: ignore
        node_name = pod.spec.node_name  # pyright: ignore
        label_selector = "app=sidecar"
        pods = await v1.list_namespaced_pod(namespace=NAMESPACE, label_selector=label_selector)
        same_node_pod_names = [p.metadata.name for p in pods.items if p.spec.node_name == node_name]
        sorted_pod_names = sorted(same_node_pod_names)
        if pod_name not in sorted_pod_names:
            logger.error("Current pod not found in the list of sidecar pods on the node. K8s issue?")
            return None
        return SidecarNodeInfo([int(pod_name.split("-")[-1]) for pod_name in sorted_pod_names])


async def main(
    ip: str = "[::]",
    base_port: int = 10000,
) -> None:
    """Main entrypoint for the sidecar server.

    The entrypoint uses the following environment variables to configure the sidecar server.
    When launched in a Kubernetes cluster, the `SIDECAR_POD_NAME` environment variable is used
    to determine the global rank and the GPU device used for each sidecar.

    When launched outside of a Kubernetes cluster, the `SIDECAR_RANK` environment variable is used
    to determine the global rank and the GPU device used for each sidecar, which will be the same.
    Note this means that outside of k8s, only single node is supported.

    Environment variables:
        - SIDECAR_WORLD_SIZE: The total number of sidecars in the cluster.
        - SIDECAR_MASTER_ADDR: The address of the master node.
        - SIDECAR_MASTER_PORT: The port of the master node.
        - SIDECAR_SHM_SIZE: The size of the shared memory buffer in bytes in each sidecar,
            this will be divided by the dtype size so it should be a multiple of the dtype size.
        K8s only:
        - SIDECAR_POD_NAME: The name of the pod the sidecar is running in.
        Outside of k8s:
        - SIDECAR_RANK: The global rank of the sidecar
        - SIDECAR_DEVICE_ID: The device id of the GPU used by the sidecar, will use SIDECAR_RANK if not set.
        - SIDECAR_NUM_DEVICES: Optional. The number of devices on the node, will use SIDECAR_WORLD_SIZE if not set.
    """
    world_size = int(os.environ.get("SIDECAR_WORLD_SIZE", 1))
    master_addr = os.environ.get("SIDECAR_MASTER_ADDR", "localhost")
    master_port = os.environ.get("SIDECAR_MASTER_PORT", 48105)
    shm_size = int(os.environ.get("SIDECAR_SHM_SIZE", 2**30))

    assert world_size > 0, "Invalid SIDECAR_WORLD_SIZE"
    pod_name = os.environ.get("SIDECAR_POD_NAME")

    if pod_name:
        try:
            sidecar_rank = int(pod_name.split("-")[-1])
        except ValueError:
            sidecar_rank = -1
    else:
        sidecar_rank = int(os.environ.get("SIDECAR_RANK", -1))

    assert sidecar_rank >= 0, "Invalid sidecar rank"

    # We start the server so the health check gRPC is always available
    server = grpc.aio.server()
    servicer = CommSidecarServicer(sidecar_rank=sidecar_rank)
    comm_sidecar_pb2_grpc.add_CommSidecarServicer_to_server(servicer, server)
    port = base_port + sidecar_rank
    listen_addr = f"{ip}:{port}"
    server.add_insecure_port(listen_addr)
    await server.start()
    logger.info("Sidecar server started on %s", listen_addr)

    async def server_graceful_shutdown():
        logger.info("Starting graceful shutdown...")
        await servicer.shutdown()
        await server.stop(5)
        logger.info("Server stopped")

    init_url = f"tcp://{master_addr}:{master_port}"
    dist.init_process_group(
        backend="gloo",
        init_method=init_url,
        rank=sidecar_rank,
        world_size=world_size,
    )

    # now that every sidecar server has started, we query the cluster to retrieve
    # the device_id and num_devices within the node when using k8s
    if pod_name:
        node_info = await _get_node_info(pod_name)
    else:
        # outside of k8s, currently limited to identity mapping
        node_info = SidecarNodeInfo([i for i in range(world_size)])

    assert node_info is not None, "Failed to get node info"

    assert shm_size % torch.cdouble.itemsize == 0, (
        "shm_size should be a multiple of num_devices * max(torch.cdouble) dtype itemsize"
    )
    # dist process group is initialized, now we can mark the server live
    servicer.online(node_info=node_info, shm_size=shm_size)

    cleanup_coroutines.append(server_graceful_shutdown())
    await server.wait_for_termination()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(tyro.cli(main))
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(asyncio.gather(*cleanup_coroutines))
        loop.close()
