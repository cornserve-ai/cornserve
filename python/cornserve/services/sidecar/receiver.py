"""Sidecar Receiver."""

import asyncio
import contextlib
import ctypes
import os

import grpc
import torch
from opentelemetry import trace

from cornserve.logging import SidcarAdapter, get_logger
from cornserve.services.pb import common_pb2, sidecar_pb2, sidecar_pb2_grpc
from cornserve.services.sidecar.schema import (
    RecvObjState,
    RecvRequestState,
    RecvTensorState,
    SidecarReceiverConfig,
)
from cornserve.services.sidecar.shm_manager import (
    SharedMemoryBuffer,
    SharedMemoryManager,
)
from cornserve.sidecar.serde import (
    ForwardTensorHandle,
    MsgpackDecoder,
    MsgpackEncoder,
    SharedTensorHandle,
)
from cornserve.sidecar.utils import (
    buffer_from_tensor,
    chunk_tag,
    grpc_url_from_rank,
    shm_filename,
)

logger = get_logger(__name__, [SidcarAdapter])
tracer = trace.get_tracer(__name__)


class SidecarReceiver:
    """The receiver sidecar server supports receiving tensors from other ranks using ucx-py backend."""

    def __init__(
        self,
        config: SidecarReceiverConfig,
    ) -> None:
        """Initialize the receiver sidecar.

        Args:
            config: The configuration for the receiver sidecar.
        """
        self.config = config
        self.sidecar_rank = config.sidecar_rank
        self.group = config.group

        self.shm_fn = shm_filename()
        self.node_info = config.node_info
        self.local_ranks = [self.node_info.get_device_id(i) for i in config.group]
        self.shm_manager = SharedMemoryManager(
            shm=config.shared_tensor,
            slot_size=config.slot_numel,
        )
        self.dtype = config.shared_tensor.dtype
        self.memory_freed = asyncio.Condition()

        self.has_memory = asyncio.Condition()

        self.malloc_events: dict[str, asyncio.Event] = {}
        # This lock guards access over the self.malloc_events
        self.malloc_lock = asyncio.Lock()

        # a legder to keep the transfer status of each transfer request
        self.ledger: dict[str, RecvRequestState] = {}
        # This lock guards access over the self.ledger when necessary
        self.recv_done_lock = asyncio.Lock()

        # this is used to keep track of the memory pressure events
        self.mem_pressure_count = 0

        self.peers = config.peers

        self.dst_channels: dict[int, grpc.aio.Channel] = {}
        self.dst_stubs: dict[int, sidecar_pb2_grpc.SidecarStub] = {}

        self.encoder = MsgpackEncoder()
        self.decoder = MsgpackDecoder()

        logger.info("SidecarReceiver initialized, using slot size %s", config.slot_numel)

    async def _allocate(self, size: int) -> SharedMemoryBuffer:
        """Allocate a shared memory buffer of the given size.

        size: The number of elements to allocate.
        """
        async with self.memory_freed:
            buffer = self.shm_manager.allocate(size)
            while buffer is None:
                logger.warning("Memory pressure detected, waiting for memory to be freed")
                self.mem_pressure_count += 1
                await self.memory_freed.wait()
                buffer = self.shm_manager.allocate(size)
            return buffer

    async def _free(self, buffer: SharedMemoryBuffer) -> None:
        """Free a shared memory buffer.

        Args:
            buffer: The shared memory buffer to free.
        """
        async with self.memory_freed:
            self.shm_manager.free(buffer)
            self.memory_freed.notify_all()

    async def shutdown(self) -> None:
        """Cleanup routines for the receiver."""
        # remove the shared memory file, used async to unify the interface
        del self.shm_manager
        for channel in self.dst_channels.values():
            await channel.close()
        with contextlib.suppress(Exception):
            os.unlink(shm_filename())

    async def prepare_receive(
        self,
        request: sidecar_pb2.PrepareReceiveRequest,
        context: grpc.aio.ServicerContext,
    ) -> sidecar_pb2.PrepareReceiveResponse:
        """Prepare to receive a tensor from another rank, called by the sender sidecar server.

        This function allocates a shared memory buffer if not already allocated,
        and queues up a receive task to receive the tensor.
        """
        span = trace.get_current_span()
        # malloc is per chunk
        malloc_id = request.id + f"-{request.chunk_id}"
        span.set_attribute("SidecarReceiver.prepare_receive.id", request.id)
        span.set_attribute("SidecarReceiver.prepare_receive.chunk_id", request.chunk_id)
        obj = self.decoder.decode(request.data)
        if isinstance(obj, ForwardTensorHandle):
            span.set_attribute("SidecarReceiver.prepare_receive.type", "ForwardTensorHandle")
            # inter-node
            is_first = False
            async with self.malloc_lock:
                if malloc_id not in self.malloc_events:
                    # first call
                    is_first = True
                    self.malloc_events[malloc_id] = asyncio.Event()

            if is_first:
                span.add_event("allocate.start")
                buffer = await self._allocate(obj.total_numel)
                span.add_event("allocate.done")
                buffer.create_shards(obj.num_shards)

                if request.id in self.ledger:
                    chunk_state = RecvTensorState(request.chunk_id, buffer)
                    self.ledger[request.id].chunks[request.chunk_id] = chunk_state
                    if self.ledger[request.id].num_chunks == -1:
                        self.ledger[request.id].num_chunks = request.num_chunks
                else:
                    req_state = RecvRequestState(request.id, request.num_chunks)
                    req_state.chunks[request.chunk_id] = RecvTensorState(request.chunk_id, buffer)
                    self.ledger[request.id] = req_state

                self.malloc_events[malloc_id].set()
            else:
                # wait for the first call to finish allocating
                span.add_event("wait_for_allocate.start")
                await self.malloc_events[malloc_id].wait()
                span.add_event("wait_for_allocate.done")
                chunk_state = self.ledger[request.id].chunks[request.chunk_id]
                assert isinstance(chunk_state, RecvTensorState)
                buffer = chunk_state.buffer

            tag = chunk_tag(
                request.id,
                request.src_rank,
                request.chunk_id,
                obj.shard_rank,
            )

            async def recv_task() -> None:
                peer = self.peers[request.src_rank]
                logger.debug(
                    "receiving to data_ptr %s for shard %s req id %s",
                    buffer.shards[obj.shard_rank].data.data_ptr(),
                    obj.shard_rank,
                    request.id,
                )
                span.add_event("recv.start")
                await peer.recv(
                    buffer_from_tensor(buffer.shards[obj.shard_rank].data),
                    tag=tag,
                )
                span.add_event("recv.done")
                buffer.mark_shard_ready(obj.shard_rank)
                if buffer.is_ready():
                    # cleanup the malloc event
                    async with self.malloc_lock:
                        if malloc_id in self.malloc_events:
                            del self.malloc_events[malloc_id]

                    # mark the request as done
                    async with self.recv_done_lock:
                        req_state = self.ledger[request.id]
                        chunk_state = req_state.chunks[request.chunk_id]
                        assert isinstance(chunk_state, RecvTensorState)
                        chunk_state.done = True
                        req_state.done_event.set()

            asyncio.create_task(recv_task())
            return sidecar_pb2.PrepareReceiveResponse(status=common_pb2.Status.STATUS_OK)

        elif isinstance(obj, SharedTensorHandle):
            span.set_attribute("SidecarReceiver.prepare_receive.type", "SharedTensorHandle")
            # intra-node
            logger.info("Intra node prepare receive request for request id %s", request.id)
            cbuf = (ctypes.c_byte * obj.numel * self.dtype.itemsize).from_address(self.config.base_ptr + obj.offset)
            tensor = torch.frombuffer(cbuf, dtype=self.dtype, count=obj.numel)
            dummy_buffer = SharedMemoryBuffer(size=obj.numel, data=tensor, slots=[])
            chunk_state = RecvTensorState(request.chunk_id, dummy_buffer, request.src_rank, True)

            async with self.recv_done_lock:
                if request.id in self.ledger:
                    # check if first
                    self.ledger[request.id].chunks[request.chunk_id] = chunk_state
                    if self.ledger[request.id].num_chunks == -1:
                        self.ledger[request.id].num_chunks = request.num_chunks
                else:
                    req_state = RecvRequestState(request.id, request.num_chunks)
                    req_state.chunks[request.chunk_id] = chunk_state
                    self.ledger[request.id] = req_state
                self.ledger[request.id].done_event.set()

            return sidecar_pb2.PrepareReceiveResponse(status=common_pb2.Status.STATUS_OK)
        else:
            span.set_attribute("SidecarReceiver.prepare_receive.type", "Object")

            chunk_state = RecvObjState(request.chunk_id, obj)
            async with self.recv_done_lock:
                if request.id in self.ledger:
                    # check if first
                    self.ledger[request.id].chunks[request.chunk_id] = chunk_state
                    if self.ledger[request.id].num_chunks == -1:
                        self.ledger[request.id].num_chunks = request.num_chunks
                else:
                    req_state = RecvRequestState(request.id, request.num_chunks)
                    req_state.chunks[request.chunk_id] = chunk_state
                    self.ledger[request.id] = req_state
                self.ledger[request.id].done_event.set()
            return sidecar_pb2.PrepareReceiveResponse(status=common_pb2.Status.STATUS_OK)

    async def receive(
        self,
        recv_req: sidecar_pb2.ReceiveRequest,
        context: grpc.aio.ServicerContext,
    ) -> sidecar_pb2.ReceiveResponse:
        """Receive the tensor of a request from other ranks.

        If all shards are received, return the slot number immediately.
        Else, queues up an event for the request id and waits for all shards to be received.

        Every request may have many chunks, each chunk's recv_done will wake up all recv_done tasks,
        but only the correct one will return, the rest will continue waiting for the event

        even if say the first set wakes up all awaiting ones, and upon some called clear
        (some still woken but not run yet), another routine calls set, The sleeping ones will be
        waken up again, and the already woken ones can still check the chunk_id
        """
        span = trace.get_current_span()
        logger.info("==> Receive request for request id %s", recv_req.id)
        span.set_attribute("SidecarReceiver.receive.id", recv_req.id)
        span.set_attribute("SidecarReceiver.receive.chunk_id", recv_req.chunk_id)

        async with self.recv_done_lock:
            if recv_req.id not in self.ledger:
                req_state = RecvRequestState(recv_req.id, -1)
                self.ledger[recv_req.id] = req_state

        req_state = self.ledger[recv_req.id]
        while True:
            await req_state.done_event.wait()
            # some chunk of this request is already received
            if recv_req.chunk_id > req_state.num_chunks:
                # check out of bound
                return sidecar_pb2.ReceiveResponse(
                    status=common_pb2.Status.STATUS_OK,
                    data=self.encoder.encode(None),
                )

            async with self.recv_done_lock:
                if recv_req.chunk_id not in req_state.chunks:
                    # chunk not received yet
                    req_state.done_event.clear()
                    continue

                chunk_state = req_state.chunks[recv_req.chunk_id]
                if isinstance(chunk_state, RecvTensorState) and not chunk_state.done:
                    # still waiting for shards of the tensor
                    req_state.done_event.clear()
                    continue

                if isinstance(chunk_state, RecvObjState):
                    obj = chunk_state.data
                elif isinstance(chunk_state, RecvTensorState):
                    obj = chunk_state.buffer.create_handle(self.config.base_ptr)
                else:
                    raise ValueError("Unknown chunk state")

                return sidecar_pb2.ReceiveResponse(
                    status=common_pb2.Status.STATUS_OK,
                    data=self.encoder.encode(obj),
                )

    def _get_grpc_stub(self, rank: int) -> sidecar_pb2_grpc.SidecarStub:
        """Get the stub for the given rank.

        Args:
            rank: The rank of the sidecar server.
        """
        if rank not in self.dst_stubs:
            self.dst_channels[rank] = grpc.aio.insecure_channel(grpc_url_from_rank(rank))
            self.dst_stubs[rank] = sidecar_pb2_grpc.SidecarStub(self.dst_channels[rank])
        return self.dst_stubs[rank]

    async def mark_done(
        self,
        mark_done_req: sidecar_pb2.MarkDoneRequest,
        context: grpc.aio.ServicerContext,
    ) -> sidecar_pb2.MarkDoneResponse:
        """Mark a tensor as consumed, free up the shared memory used.

        If the shared buffer is from a intra-node transfer, this call will invoke the
        `Unlink` subroutine.
        """
        span = trace.get_current_span()
        span.set_attribute("SidecarReceiver.mark_done.id", mark_done_req.id)
        span.set_attribute("SidecarReceiver.mark_done.chunk_id", mark_done_req.chunk_id)
        if mark_done_req.id not in self.ledger or mark_done_req.chunk_id not in self.ledger[mark_done_req.id].chunks:
            logger.error("mark_done: %s not found", mark_done_req.id)
            await context.abort(grpc.StatusCode.NOT_FOUND, "mark_done_req not found")
        req_state = self.ledger[mark_done_req.id]
        chunk_state = req_state.chunks[mark_done_req.chunk_id]

        if isinstance(chunk_state, RecvObjState):
            span.set_attribute("SidecarReceiver.mark_done.type", "Object")
            del req_state.chunks[mark_done_req.chunk_id]
            # TODO: make this counter instead of assuming sequential access
            if req_state.num_chunks == mark_done_req.chunk_id + 1:
                # last chunk
                del self.ledger[mark_done_req.id]
            return sidecar_pb2.MarkDoneResponse(status=common_pb2.Status.STATUS_OK)

        assert isinstance(chunk_state, RecvTensorState)

        if chunk_state.intra_node_rank >= 0:
            logger.info(
                "mark_done: unlink refcount for id %s in rank %d",
                mark_done_req.id,
                chunk_state.intra_node_rank,
            )
            stub = self._get_grpc_stub(chunk_state.intra_node_rank)
            unlink_req = sidecar_pb2.UnlinkRequest(id=mark_done_req.id, chunk_id=mark_done_req.chunk_id)
            res = await stub.Unlink(unlink_req)
            if res.status != common_pb2.Status.STATUS_OK:
                await context.abort(grpc.StatusCode.INTERNAL, "Failed to unlink intra node memory")
        else:
            logger.info(
                "mark_done: Freeing up %d slots from %s",
                len(chunk_state.buffer.slots),
                mark_done_req.id,
            )
            await self._free(chunk_state.buffer)

        # TODO: make this counter instead of assuming sequential access
        if req_state.num_chunks == mark_done_req.chunk_id + 1:
            # last chunk
            del self.ledger[mark_done_req.id]
        return sidecar_pb2.MarkDoneResponse(status=common_pb2.Status.STATUS_OK)
