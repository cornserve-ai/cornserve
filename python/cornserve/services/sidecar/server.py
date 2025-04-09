"""Sidecar server implementniation.

The sidecar server is a gRPC service that runs on each node in the cluster. This service
is the backend for the `SidecarSender` and `SidecarReceiver` classes in the `api` module.
It has two corresponding components, `SidecarSender` and `SidecarReceiver`, which
together provide the functionality to send and receive tensors between ranks.
"""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import os
import pickle
from dataclasses import dataclass

import grpc
import kubernetes_asyncio.client as kclient
import kubernetes_asyncio.config as kconfig
import numpy as np
import torch
import tyro
import ucxx
from opentelemetry import trace
from opentelemetry.instrumentation.grpc import (
    GrpcAioInstrumentorClient,
    GrpcAioInstrumentorServer,
    GrpcInstrumentorClient,
    GrpcInstrumentorServer,
)
from ucxx._lib_async.endpoint import Endpoint

from cornserve.logging import SidcarAdapter, get_logger
from cornserve.services.pb import sidecar_pb2, sidecar_pb2_grpc, common_pb2
from cornserve.tracing import configure_otel

from .shm_manager import SharedMemoryBuffer, SharedMemoryManager
from .utils import (
    GRPC_BASE_PORT,
    UCX_BASE_PORT,
    TensorLayout,
    buffer_from_tensor,
    chunk_tag,
    device_from_rank,
    grpc_channel_from_rank,
    init_shmem,
    shm_fn,
    ucx_port_from_rank,
    ucx_url_from_rank,
)

logger = get_logger(__name__, [SidcarAdapter])
tracer = trace.get_tracer(__name__)
cleanup_coroutines = []


class SidecarReceiver:
    """The receiver sidecar server supports receiving tensors from other ranks using ucx-py backend."""

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
        group: list[int],
        node_info: SidecarNodeInfo,
        shm_size: int,
        slot_size: int,
        dtype: str,
        peers: dict[int, Endpoint],
    ) -> None:
        """Initialize the receiver sidecar.

        Args:
            sidecar_rank: The sidecar rank, aka global rank.
            group: The ranks of the TP group to receive tensors from.
            node_info: The node information.
            shm_size: The shared memory size (number of elements of given dtype).
            slot_size: The shape of the tensor to be received, currently fixed.
            dtype: The data type of the receiving tensor.
            peers: The peers to receive the tensor from.
        """
        self.sidecar_rank = sidecar_rank
        self.group = group

        self.dtype = getattr(torch, dtype)
        self.shm_fn = shm_fn()
        self.node_info = node_info
        self.local_ranks = [self.node_info.get_device_id(i) for i in group]
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
        self.ledger: dict[str, SidecarReceiver.TransferRequestState] = {}
        # per req event, recieve will wait on this event, recv_task will try to set this event
        self.req_events: dict[str, asyncio.Event] = {}

        # we use a multiprocessing lock to protect the done flag, as this lock is used in the recv_task,
        # which is running in a separate thread to avoid blocking on recv
        self.recv_done_lock = mp.Lock()

        # this is used to keep track of the memory pressure events
        self.mem_pressure_count = 0

        self.peers = peers

    async def shutdown(self):
        """Cleanup routines for the receiver."""
        # remove the shared memory file, used async to unify the interface
        del self.shared_tensor
        del self.shm_manager
        with contextlib.suppress(Exception):
            os.unlink(self.shm_fn)

    async def prepare_receive(
        self,
        request: sidecar_pb2.PrepareReceiveRequest,
        context: grpc.aio.ServicerContext,
    ) -> sidecar_pb2.PrepareReceiveResponse:
        """Prepare to receive a tensor from another rank, called by the sender sidecar server.

        This function allocates a shared memory buffer if not already allocated,
        and queues up a receive task to receive the tensor.
        """
        logger.debug(
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

        span = trace.get_current_span()
        async with self.has_memory:
            # acquire the underlying lock
            if request.id not in self.ledger:
                # some prepare_recv needs to allocate the buffer and update the ledger
                if request.id not in self.malloc_events:
                    # this is the first prepare_recv call for request id
                    span.add_event("allocate.start")
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
                    span.add_event("allocate.done")

                    buffer.create_chunks(request.num_chunks, request.num_shards)
                    self.ledger[request.id] = SidecarReceiver.TransferRequestState(request.id, buffer)

                    if request.id in self.malloc_events:
                        # wake up all the waiting prepare_recv calls
                        self.malloc_events[request.id].set()
                        del self.malloc_events[request.id]
                else:
                    # some previous prepare_recv call is blocking on the allocation
                    span.add_event("allocate_wait.start")
                    event = self.malloc_events[request.id]
                    self.has_memory.release()
                    try:
                        await event.wait()
                        span.add_event("allocate_wait.done")
                    finally:
                        # Make sure to re-acquire the lock after waiting
                        await self.has_memory.acquire()

        state = self.ledger[request.id]
        chunk = state.buffer.chunks[request.chunk_id]
        tag = chunk_tag(request.src_rank, request.chunk_id, request.shard_rank)

        # TODO: allow batch recv
        async def recv_task():
            """The task to receive the tensor."""
            logger.info("Queuing recv task for chunk %d of request %s tag %d", request.chunk_id, request.id, tag)
            peer = self.peers[request.src_rank]
            await peer.recv(
                buffer_from_tensor(chunk.data[request.shard_offset : request.shard_offset + request.shard_size]),
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

        asyncio.create_task(recv_task())
        return sidecar_pb2.PrepareReceiveResponse(status=common_pb2.Status.STATUS_OK)

    async def receive(
        self,
        recv_req: sidecar_pb2.ReceiveRequest,
        context: grpc.aio.ServicerContext,
    ) -> sidecar_pb2.ReceiveResponse:
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
        return sidecar_pb2.ReceiveResponse(offset=offset, size=size)

    async def mark_done(
        self,
        mark_done_req: sidecar_pb2.MarkDoneRequest,
        context: grpc.aio.ServicerContext,
    ) -> sidecar_pb2.MarkDoneResponse:
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
        return sidecar_pb2.MarkDoneResponse(status=common_pb2.Status.STATUS_OK)


class SidecarSender:
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
        peers: dict[int, Endpoint],
        shard_rank: int = 0,
        num_shards: int = 1,
        layout: TensorLayout = TensorLayout.FULL,
    ) -> None:
        """Initialize the sender sidecar server.

        Args:
            sidecar_rank: The sidecar rank, aka global rank.
            shm_size: The shared memory size (number of elements of given dtype).
            slot_size: The slot_size of the shared memory buffer.
            dtype: The data type of the sending tensor.
            peers: The peers to send the tensor to.
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

        self.dst_channels: dict[int, grpc.aio.Channel] = {}
        self.dst_stubs: dict[int, sidecar_pb2_grpc.SidecarStub] = {}
        self.mem_pressure_count = 0
        self.peers = peers

    async def report_memory(
        self, request: sidecar_pb2.ReportMemoryRequest, context: grpc.aio.ServicerContext
    ) -> sidecar_pb2.ReportMemoryResponse:
        """Updates the memory pressure count."""
        self.mem_pressure_count = request.pressure
        return sidecar_pb2.ReportMemoryResponse(status=common_pb2.Status.STATUS_OK)

    async def shutdown(self) -> None:
        """Cleanup routines for the sender sidecar."""
        for channel in self.dst_channels.values():
            await channel.close()
        # remove the shared memory file
        del self.shared_tensor
        os.unlink(shm_fn())

    async def send(
        self,
        send_request: sidecar_pb2.SendRequest,
        context: grpc.aio.ServicerContext,
    ) -> sidecar_pb2.SendResponse:
        """Send a tensor to another rank.

        First use prepare_receive to send control signals to the destination sidecar,
        then queue up the send tasks.
        """
        span = trace.get_current_span()
        # sanity check
        if send_request.slot < 0 or send_request.slot * self.slot_size + send_request.size > self.shm_size:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"slot out of range {send_request.slot}*{self.slot_size}+{send_request.size} {self.shm_size}",
            )

        if not send_request.dst_ranks:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "dst_ranks cannot be empty")
        if any(r < 0 or r == self.sidecar_rank for r in send_request.dst_ranks):
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid destination rank")
        # only send to the head receiver when TP is enabled (min sidecar rank)
        dst_rank = min(send_request.dst_ranks)
        if dst_rank not in self.peers:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Peer does not exist")

        # inform destination sidecar
        # lazily create channel
        if dst_rank not in self.dst_channels:
            self.dst_channels[dst_rank] = grpc.aio.insecure_channel(grpc_channel_from_rank(dst_rank))
            self.dst_stubs[dst_rank] = sidecar_pb2_grpc.SidecarStub(self.dst_channels[dst_rank])
            logger.info("Connected to sidecar-%d", dst_rank)

        stub = self.dst_stubs[dst_rank]
        logger.info(
            "Calling prepare receive on sidecar-%d for request %s chunk id %s out of %d",
            dst_rank,
            send_request.id,
            send_request.chunk_id,
            send_request.num_chunks,
        )
        prepare_receive_request = sidecar_pb2.PrepareReceiveRequest(
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
            return sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_ERROR)

        ipc_handle = pickle.loads(send_request.ipc_handle)
        cuda_event = torch.cuda.Event.from_ipc_handle(self.device, ipc_handle)

        await asyncio.to_thread(cuda_event.synchronize)
        span.add_event("copy.done")

        logger.info("Sending chunk %d for req %s", send_request.chunk_id, send_request.id)
        tag = chunk_tag(self.sidecar_rank, send_request.chunk_id, self.shard_rank)

        peer = self.peers[dst_rank]
        to_send = self.shared_tensor[
            send_request.slot * self.slot_size : send_request.slot * self.slot_size + send_request.size
        ]
        span.add_event("send.start")
        await peer.send(buffer_from_tensor(to_send), tag=tag)
        span.add_event("send.done")
        span.set_attribute("sidecar_sender_server.send.size", send_request.size)
        logger.info(
            "SHARD RANK %d: sent chunk %d for request %s", self.shard_rank, send_request.chunk_id, send_request.id
        )

        return sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_OK)


class SidecarServicer(sidecar_pb2_grpc.SidecarServicer):
    """A unified wrapper for both sender and receiver sidecar servers. Entry point for the gRPC service."""

    def __init__(
        self,
        sidecar_rank: int,
        ucx_port: int,
        world_size: int,
        mem_pressure_threshold=500,
    ) -> None:
        """Initialize the sidecar service.

        This creates an offline sidecar server that only has the CheckHealth endpoint available.

        Args:
            sidecar_rank: The global rank of the sidecar.
            ucx_port: The UCX port to use for communication.
            world_size: The total number of sidecars in the cluster.
            mem_pressure_threshold: The threshold of memory pressure count to trigger the memory pressure status.
        """
        self.sidecar_rank = sidecar_rank
        self.sidecar: SidecarSender | SidecarReceiver | None = None
        self.live = False
        self.mem_pressure_threshold = mem_pressure_threshold
        self.ucx_port = ucx_port
        self.world_size = world_size
        self.peers = dict[int, Endpoint]()

        async def _ucxx_listener_callback(ep: Endpoint) -> None:
            """Callback for the UCX listener."""
            id = np.empty(1, dtype=np.int32)
            await ep.recv(id)
            if id[0] in self.peers:
                logger.warning("Overwriting endpoint %d", id[0])
            self.peers[id[0]] = ep

        self.ucx_listener = ucxx.create_listener(_ucxx_listener_callback, port=self.ucx_port)

    async def p2p_connect(self) -> None:
        """Connect to other peers using UCX.

        Connects to peers with lower sidecar ranks, and wait to be connected by peers with higher sidecar ranks.
        """
        for i in range(self.world_size):
            if i < self.sidecar_rank:
                while i not in self.peers:
                    try:
                        logger.info(
                            "Connecting to sidecar-%d - url %s - port %d",
                            i,
                            ucx_url_from_rank(i),
                            ucx_port_from_rank(i),
                        )
                        ep = await ucxx.create_endpoint(ucx_url_from_rank(i), ucx_port_from_rank(i))
                        msg = np.array([self.sidecar_rank], dtype=np.int32)
                        await ep.send(msg)
                        self.peers[i] = ep
                    except Exception:
                        await asyncio.sleep(0.5)
        while len(self.peers) < self.world_size - 1:
            await asyncio.sleep(0.5)
        logger.info("Connected to all peers")

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
        request: sidecar_pb2.CheckHealthRequest,
        context: grpc.aio.ServicerContext,
    ) -> sidecar_pb2.CheckHealthResponse:
        """Health check for the sidecar."""
        if not self.live or self.sidecar is None:
            return sidecar_pb2.CheckHealthResponse(status=sidecar_pb2.HealthStatus.HEALTH_OFFLINE)
        if self.sidecar.mem_pressure_count > self.mem_pressure_threshold:
            return sidecar_pb2.CheckHealthResponse(status=sidecar_pb2.HealthStatus.HEALTH_MEMORY_PRESSURE)
        return sidecar_pb2.CheckHealthResponse(status=sidecar_pb2.HealthStatus.HEALTH_ALL_GOOD)

    async def RegisterSender(  # noqa: N802
        self,
        request: sidecar_pb2.RegisterSenderRequest,
        context: grpc.aio.ServicerContext,
    ) -> sidecar_pb2.RegisterResponse:
        """Called by the sender server to register the sidecar."""
        if not self.live:
            logger.error("Sidecar not online")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not online")

        if self.sidecar is not None:
            logger.warning("Overwriting existing sidecar")

        dtype = getattr(torch, request.dtype)
        # calculate the number of shared elements to return
        shm_size = self.shm_size // dtype.itemsize

        self.sidecar = SidecarSender(
            sidecar_rank=self.sidecar_rank,
            shm_size=shm_size,
            slot_size=request.slot_size,
            dtype=dtype,
            peers=self.peers,
            shard_rank=request.shard_rank,
            num_shards=request.num_shards,
            node_info=self.node_info,
            layout=TensorLayout(request.layout),
        )

        logger.info("Registered sender of local_rank %s, sidecar_rank %s", self.device_id, self.sidecar_rank)

        return sidecar_pb2.RegisterResponse(
            shm_size=shm_size,
            local_ranks=[self.device_id],
            num_local_sidecars=self.num_devices,
        )

    async def RegisterReceiver(  # noqa: N802
        self,
        request: sidecar_pb2.RegisterReceiverRequest,
        context: grpc.aio.ServicerContext,
    ) -> sidecar_pb2.RegisterResponse:
        """Called by the receiver server to register the sidecar."""
        if not self.live:
            logger.error("Sidecar not online")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not online")
        if self.sidecar is not None:
            logger.warning("Overwriting existing sidecar")

        if self.sidecar_rank not in request.group:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid group rank")
        for r in request.group:
            if not self.node_info.contains(r):
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid group rank")

        dtype = getattr(torch, request.dtype)
        shm_size = self.shm_size // dtype.itemsize

        self.sidecar = SidecarReceiver(
            sidecar_rank=self.sidecar_rank,
            group=list(request.group),
            shm_size=shm_size,
            slot_size=request.slot_size,
            dtype=request.dtype,
            peers=self.peers,
            node_info=self.node_info,
        )
        logger.info(
            "Registered receiver of local_rank %s, sidecar_rank %s, slot_size %d, dtype %s",
            self.device_id,
            self.sidecar_rank,
            request.slot_size,
            request.dtype,
        )

        return sidecar_pb2.RegisterResponse(
            shm_size=shm_size,
            local_ranks=[self.node_info.get_device_id(i) for i in request.group],
            num_local_sidecars=self.num_devices,
        )

    async def RegisterReader(  # noqa: N802
        self,
        request: sidecar_pb2.RegisterReaderRequest,
        context: grpc.aio.ServicerContext,
    ) -> sidecar_pb2.RegisterResponse:
        """Register a read-only sidecar. This is temporary."""
        if not self.live or self.sidecar is None:
            logger.error("Sidecar not online")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not online")

        if not isinstance(self.sidecar, SidecarReceiver):
            logger.error("Invalid sidecar mode")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Invalid sidecar mode")

        logger.info("Registered reader of sidecar_rank %s", self.sidecar_rank)
        return sidecar_pb2.RegisterResponse(
            shm_size=self.sidecar.shm_size,
            local_ranks=self.sidecar.local_ranks,
            num_local_sidecars=self.num_devices,
        )

    async def Send(  # noqa: N802
        self, request: sidecar_pb2.SendRequest, context: grpc.aio.ServicerContext
    ) -> sidecar_pb2.SendResponse:
        """Called by the sender server to send a tensor to some other rank."""
        if self.sidecar is None:
            logger.error("Sidecar not registered")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not registered")
        if not isinstance(self.sidecar, SidecarSender):
            logger.error("Invalid sidecar mode")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Invalid sidecar mode")
        return await self.sidecar.send(request, context)

    async def PrepareReceive(  # noqa: N802
        self,
        request: sidecar_pb2.PrepareReceiveRequest,
        context: grpc.aio.ServicerContext,
    ) -> sidecar_pb2.PrepareReceiveResponse:
        """Called by the sender sidercar to the receiver sidecar to prepare receiving a tensor."""
        if not self.live:
            logger.error("Sidecar not online")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not online")
        if self.sidecar is None:
            logger.error("Sidecar not registered")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not registered")
        if not isinstance(self.sidecar, SidecarReceiver):
            logger.error("Invalid sidecar mode")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Invalid sidecar mode")
        return await self.sidecar.prepare_receive(request, context)

    async def Receive(  # noqa: N802
        self,
        request: sidecar_pb2.ReceiveRequest,
        context: grpc.aio.ServicerContext,
    ) -> sidecar_pb2.ReceiveResponse:
        """Initiate receiving a tensor from some other rank."""
        if not self.live:
            logger.error("Sidecar not online")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not online")
        if self.sidecar is None:
            logger.error("Sidecar not registered")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not registered")
        if not isinstance(self.sidecar, SidecarReceiver):
            logger.error("Invalid sidecar mode")
            return sidecar_pb2.ReceiveResponse(offset=-1, size=-1)

        return await self.sidecar.receive(request, context)

    async def MarkDone(  # noqa: N802
        self,
        request: sidecar_pb2.MarkDoneRequest,
        context: grpc.aio.ServicerContext,
    ) -> sidecar_pb2.MarkDoneResponse:
        """Called by the receiver server to mark a request as done."""
        if not self.live:
            logger.error("Sidecar not online")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not online")
        if self.sidecar is None:
            logger.error("Sidecar not registered")
            return sidecar_pb2.MarkDoneResponse(status=common_pb2.Status.STATUS_ERROR)
        if not isinstance(self.sidecar, SidecarReceiver):
            logger.error("Invalid sidecar mode")
            return sidecar_pb2.MarkDoneResponse(status=common_pb2.Status.STATUS_ERROR)

        return await self.sidecar.mark_done(request, context)

    async def shutdown(self):
        """Shutdown the sidecar."""
        if self.sidecar is not None:
            await self.sidecar.shutdown()
        for peer in self.peers.values():
            await peer.close()
        self.ucx_listener.close()
        logger.info("Sidecar shutdown")

    async def ReportMemory(
        self,
        request: sidecar_pb2.ReportMemoryRequest,
        context: grpc.aio.ServicerContext,
    ) -> sidecar_pb2.ReportMemoryResponse:
        """Report memory pressure to the sidecar."""
        if not self.live:
            logger.error("Sidecar not online")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not online")
        if self.sidecar is None:
            logger.error("Sidecar not registered")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not registered")
        if not isinstance(self.sidecar, SidecarSender):
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
    world_size = int(os.environ.get("SIDECAR_WORLD_SIZE", "1"))
    shm_size = int(os.environ.get("SIDECAR_SHM_SIZE", str(2**30)))

    assert world_size > 0, "Invalid SIDECAR_WORLD_SIZE"
    pod_name = os.environ.get("SIDECAR_POD_NAME")

    if pod_name:
        try:
            sidecar_rank = int(pod_name.split("-")[-1])
        except ValueError:
            sidecar_rank = -1
    else:
        sidecar_rank = int(os.environ.get("SIDECAR_RANK", "-1"))

    assert sidecar_rank >= 0, "Invalid sidecar rank"

    # OpenTelemetry setup
    configure_otel(name=f"sidecar[{sidecar_rank}]")

    GrpcInstrumentorClient().instrument()
    GrpcInstrumentorServer().instrument()
    GrpcAioInstrumentorClient().instrument()
    GrpcAioInstrumentorServer().instrument()

    # We start the server so the health check gRPC is always available
    server = grpc.aio.server()
    ucx_port = UCX_BASE_PORT + sidecar_rank
    servicer = SidecarServicer(
        sidecar_rank=sidecar_rank,
        ucx_port=ucx_port,
        world_size=world_size,
    )
    sidecar_pb2_grpc.add_SidecarServicer_to_server(servicer, server)
    port = GRPC_BASE_PORT + sidecar_rank
    listen_addr = f"{ip}:{port}"
    server.add_insecure_port(listen_addr)
    await server.start()
    logger.info("Sidecar server started on %s", listen_addr)

    async def server_graceful_shutdown():
        logger.info("Starting graceful shutdown...")
        await servicer.shutdown()
        await server.stop(5)
        logger.info("Server stopped")

    logger.info("Starting sidecar server %s", sidecar_rank)
    await servicer.p2p_connect()

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
    # sidecar group p2p connected, now we can mark the server live
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
