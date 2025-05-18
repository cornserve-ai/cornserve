"""Sidecar Receiver."""

import asyncio
import contextlib
import ctypes
import os
from typing import Any

import grpc
import torch
from opentelemetry import trace

from cornserve.logging import SidcarAdapter, get_logger
from cornserve.services.pb import common_pb2, sidecar_pb2, sidecar_pb2_grpc
from cornserve.services.sidecar.schema import (
    RecvTransferRequestState,
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

        self.event_lock = asyncio.Lock()
        self.malloc_events: dict[str, asyncio.Event] = {}

        # a legder to keep the transfer status of each transfer request
        self.ledger: dict[str, RecvTransferRequestState] = {}

        # per req event, recieve will wait on this event, recv_task will try to set this event
        self.req_events: dict[str, asyncio.Event] = {}

        # we use a multiprocessing lock to protect the done flag, as this lock is used in the recv_task,
        # which is running in a separate thread to avoid blocking on recv
        self.recv_done_lock = asyncio.Lock()

        # this is used to keep track of the memory pressure events
        self.mem_pressure_count = 0

        self.peers = config.peers

        self.dst_channels: dict[int, grpc.aio.Channel] = {}
        self.dst_stubs: dict[int, sidecar_pb2_grpc.SidecarStub] = {}

        self.saved_objs: dict[str, Any] = {}

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

    async def shutdown(self):
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
        id = request.id + f"-{request.chunk_id}"
        span.set_attribute("SidecarReceiver.prepare_receive.id", id)
        obj = self.decoder.decode(request.data)
        if isinstance(obj, ForwardTensorHandle):
            span.set_attribute("SidecarReceiver.prepare_receive.type", "ForwardTensorHandle")
            # inter-node
            is_first = False
            async with self.event_lock:
                if id not in self.malloc_events:
                    # first call
                    is_first = True
                    self.malloc_events[id] = asyncio.Event()

            if is_first:
                span.add_event("allocate.start")
                buffer = await self._allocate(obj.total_numel)
                span.add_event("allocate.done")
                buffer.create_shards(obj.num_shards)

                self.ledger[id] = RecvTransferRequestState(id, buffer)
                self.malloc_events[id].set()
            else:
                # wait for the first call to finish allocating
                span.add_event("wait_for_allocate.start")
                await self.malloc_events[id].wait()
                span.add_event("wait_for_allocate.done")
                buffer = self.ledger[id].buffer

            tag = chunk_tag(
                request.id,
                request.src_rank,
                request.chunk_id,
                obj.shard_rank,
            )
            logger.debug("Tag for request id %s: %s", id, tag)

            async def recv_task():
                peer = self.peers[request.src_rank]
                logger.debug(
                    "receiving to data_ptr %s for shard %s req id %s",
                    buffer.shards[obj.shard_rank].data.data_ptr(),
                    obj.shard_rank,
                    id,
                )
                span.add_event("recv.start")
                await peer.recv(
                    buffer_from_tensor(buffer.shards[obj.shard_rank].data),
                    tag=tag,
                )
                span.add_event("recv.done")
                buffer.mark_shard_ready(obj.shard_rank)
                if buffer.is_ready():
                    async with self.recv_done_lock:
                        self.ledger[id].done = True
                        if id in self.req_events:
                            self.req_events[id].set()

            asyncio.create_task(recv_task())
            return sidecar_pb2.PrepareReceiveResponse(status=common_pb2.Status.STATUS_OK)

        elif isinstance(obj, SharedTensorHandle):
            span.set_attribute("SidecarReceiver.prepare_receive.type", "SharedTensorHandle")
            # intra-node
            logger.info("Intra node prepare receive request for request id %s", id)
            cbuf = (ctypes.c_byte * obj.numel * self.dtype.itemsize).from_address(self.config.base_ptr + obj.offset)
            tensor = torch.frombuffer(cbuf, dtype=self.dtype, count=obj.numel)
            dummy_buffer = SharedMemoryBuffer(size=obj.numel, data=tensor, slots=[])
            self.ledger[id] = RecvTransferRequestState(id, dummy_buffer, request.src_rank)
            async with self.recv_done_lock:
                self.ledger[id].done = True
                if id in self.req_events:
                    logger.info("Setting req done event for request id %s", id)
                    self.req_events[id].set()
            return sidecar_pb2.PrepareReceiveResponse(status=common_pb2.Status.STATUS_OK)
        else:
            span.set_attribute("SidecarReceiver.prepare_receive.type", "Object")
            async with self.recv_done_lock:
                self.saved_objs[id] = obj
            if id in self.req_events:
                self.req_events[id].set()
            return sidecar_pb2.PrepareReceiveResponse(status=common_pb2.Status.STATUS_OK)

    async def receive(
        self,
        recv_req: sidecar_pb2.ReceiveRequest,
        context: grpc.aio.ServicerContext,
    ) -> sidecar_pb2.ReceiveResponse:
        """Receive the tensor of a request from other ranks.

        If all shards are received, return the slot number imediately.
        Else, queues up an event for the request id and waits for all shards to be received.
        """
        span = trace.get_current_span()
        id = recv_req.id + f"-{recv_req.chunk_id}"
        logger.info("==> Receive request for request id %s", id)
        span.set_attribute("SidecarReceiver.receive.id", id)

        await self.recv_done_lock.acquire()

        if id in self.saved_objs:
            # objects
            self.recv_done_lock.release()
            return sidecar_pb2.ReceiveResponse(
                status=common_pb2.Status.STATUS_OK,
                data=self.encoder.encode(self.saved_objs[id]),
            )

        if id in self.ledger and self.ledger[id].done:
            self.recv_done_lock.release()
        else:
            # still waiting for shards/objects
            event = asyncio.Event()
            self.req_events[id] = event
            self.recv_done_lock.release()
            logger.info("Waiting for all shards to be received for request id %s", id)
            await event.wait()

        if id in self.saved_objs:
            # objects
            return sidecar_pb2.ReceiveResponse(
                status=common_pb2.Status.STATUS_OK,
                data=self.encoder.encode(self.saved_objs[id]),
            )

        logger.info("==> All shards received for request id %s", id)
        state = self.ledger[id]
        obj = state.buffer.create_handle(self.config.base_ptr)

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
        """Mark a tensor as consumed, free up the shared memory used."""
        span = trace.get_current_span()
        id = mark_done_req.id + f"-{mark_done_req.chunk_id}"
        span.set_attribute("SidecarReceiver.mark_done.id", id)
        if id in self.saved_objs:
            del self.saved_objs[id]
            return sidecar_pb2.MarkDoneResponse(status=common_pb2.Status.STATUS_OK)
        if id not in self.ledger:
            await context.abort(grpc.StatusCode.NOT_FOUND, "mark_done_req not found")
        state = self.ledger[id]
        if state.intra_node_rank >= 0:
            logger.info(
                "mark_done: unlink refcount for id %s in rank %d",
                id,
                state.intra_node_rank,
            )
            stub = self._get_grpc_stub(state.intra_node_rank)
            unlink_req = sidecar_pb2.UnlinkRequest(id=mark_done_req.id, chunk_id=mark_done_req.chunk_id)
            res = await stub.Unlink(unlink_req)
            if res.status != common_pb2.Status.STATUS_OK:
                await context.abort(grpc.StatusCode.INTERNAL, "Failed to unlink intra node memory")
        else:
            logger.info(
                "mark_done: Freeing up %d slots from %s",
                len(self.ledger[id].buffer.slots),
                id,
            )
            await self._free(state.buffer)
        del self.ledger[id]
        if id in self.req_events:
            del self.req_events[id]
        return sidecar_pb2.MarkDoneResponse(status=common_pb2.Status.STATUS_OK)
