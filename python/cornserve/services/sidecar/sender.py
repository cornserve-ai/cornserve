"""Sidecar Sender."""

import asyncio
import contextlib
import os
from typing import cast

import grpc
import torch
from opentelemetry import trace

from cornserve.logging import SidcarAdapter, get_logger
from cornserve.services.pb import common_pb2, sidecar_pb2, sidecar_pb2_grpc
from cornserve.services.sidecar.schema import SendTransferRequestState, SidecarSenderConfig
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
    device_from_rank,
    grpc_url_from_rank,
    shm_filename,
)

logger = get_logger(__name__, [SidcarAdapter])
tracer = trace.get_tracer(__name__)


class SidecarSender:
    """Sidecar sender gRPC service backend.

    Implements the gRPC service for the sender sidecar.
    """

    def __init__(  # noqa: PLR0913
        self, config: SidecarSenderConfig
    ) -> None:
        """Initialize the sender sidecar server.

        Args:
            config: A SidecarSenderConfig object
        """
        self.config = config
        self.sidecar_rank = config.sidecar_rank
        self.node_info = config.node_info
        # sidecar_rank -> local_rank
        self.local_ranks = {i: self.node_info.get_device_id(i) for i in config.group}
        # sidecar_rank -> device
        self.devices = {i: device_from_rank(self.local_ranks[i]) for i in config.group}
        self.concurrent_copy = config.concurrent_copy
        if self.concurrent_copy:
            self.streams = {i: cast(torch.cuda.Stream, torch.cuda.Stream(device=self.devices[i])) for i in config.group}
        else:
            self.streams = {
                min(config.group): cast(torch.cuda.Stream, torch.cuda.Stream(device=self.devices[min(config.group)]))
            }

        self.dst_channels: dict[int, grpc.aio.Channel] = {}
        self.dst_stubs: dict[int, sidecar_pb2_grpc.SidecarStub] = {}

        self.peers = config.peers
        self.mem_pressure_count = 0

        self.encoder = MsgpackEncoder()
        self.decoder = MsgpackDecoder()

        self.memory_freed = asyncio.Condition()

        self.shm_manager = SharedMemoryManager(
            shm=config.shared_tensor,
            slot_size=config.slot_numel,
        )

        self.saved_buffers: dict[str, SharedMemoryBuffer] = {}
        self.ref_counts: dict[str, int] = {}

        # to lock malloc & done events
        self.event_lock = asyncio.Lock()
        self.malloc_events: dict[str, asyncio.Event] = {}

        # guard the shards_sent in the ledger
        self.sent_lock = asyncio.Lock()
        self.ledger: dict[str, SendTransferRequestState] = {}

        # per copy done event for concurrent copy
        self.done_events: dict[str, asyncio.Event] = {}

        self.mem_pressure_count = 0

        logger.info("SidecarSender initialized, using slot size %s", config.slot_numel)

    async def _allocate(self, size: int) -> SharedMemoryBuffer:
        """Allocate a shared memory buffer of the given size.

        Args:
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
            buffer: The buffer to free.
        """
        async with self.memory_freed:
            self.shm_manager.free(buffer)
            self.memory_freed.notify_all()

    async def shutdown(self) -> None:
        """Cleanup routines for the sender sidecar."""
        for channel in self.dst_channels.values():
            await channel.close()
        with contextlib.suppress(Exception):
            os.unlink(shm_filename())

    def _validate_dst_groups(self, dst_ranks: list[sidecar_pb2.RankGroup]) -> bool:
        """Validate each destination rank group is valid."""
        dst_groups = [dst_group.ranks for dst_group in dst_ranks]
        ranks = [rank for group in dst_groups for rank in group]
        return all(r >= 0 and r != self.sidecar_rank for r in ranks)

    def _get_grpc_stub(self, rank: int) -> sidecar_pb2_grpc.SidecarStub:
        """Get the stub for the given rank."""
        if rank not in self.dst_stubs:
            self.dst_channels[rank] = grpc.aio.insecure_channel(grpc_url_from_rank(rank))
            self.dst_stubs[rank] = sidecar_pb2_grpc.SidecarStub(self.dst_channels[rank])
        return self.dst_stubs[rank]

    async def _send_small_objects(
        self,
        request: sidecar_pb2.SendRequest,
    ) -> sidecar_pb2.SendResponse:
        """Send small objects to the destination ranks directly.

        shard_rank, chunk_id, num_chunks are not used in this case.
        """
        if request.shard_rank != 0:
            # silently ignore the request
            return sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_OK)
        dst_ranks = [dst_group.ranks[0] for dst_group in request.dst_ranks]
        coros = []
        for rank in dst_ranks:
            req = sidecar_pb2.PrepareReceiveRequest(
                id=request.id,
                data=request.data,
                src_rank=self.sidecar_rank,
                chunk_id=request.chunk_id,
                num_chunks=request.num_chunks,
            )
            stub = self._get_grpc_stub(rank)
            coros.append(stub.PrepareReceive(req))
        responses = await asyncio.gather(*coros)
        if any(res.status != common_pb2.Status.STATUS_OK for res in responses):
            logger.error("Failed to prepare receive")
            return sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_ERROR)
        return sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_OK)

    async def unlink(
        self,
        request: sidecar_pb2.UnlinkRequest,
        context: grpc.aio.ServicerContext,
    ) -> sidecar_pb2.UnlinkResponse:
        """Mark a tensor as consumed.

        Exclusively from intra-node send. This call will decrement the ref count (initialized as the
        number of intra-node ranks when send) of the buffer. If the ref count reaches 0, the underlying
        buffer will be freed.
        """
        span = trace.get_current_span()
        id = request.id + f"-{request.chunk_id}"
        span.set_attribute("SidecarSender.unlink.id", id)
        if id not in self.saved_buffers or id not in self.ref_counts:
            logger.error("Unlinking a non-existing buffer %s", id)
            await context.abort(grpc.StatusCode.NOT_FOUND, "invalid request id")
        self.ref_counts[id] -= 1
        if self.ref_counts[id] == 0:
            del self.ref_counts[id]
            buffer = self.saved_buffers[id]
            logger.info("Freeing buffer %s", id)
            await self._free(buffer)
            del self.saved_buffers[id]
        return sidecar_pb2.UnlinkResponse(status=common_pb2.Status.STATUS_OK)

    async def send(
        self,
        request: sidecar_pb2.SendRequest,
        context: grpc.aio.ServicerContext,
    ) -> sidecar_pb2.SendResponse:
        """Forward intermedaite data to other ranks.

        1. Small objects are sent directly through gRPC PrepareReceive.
        2. GPU Tensors:
            If concurrent_copy:
                Each TP worker will call `send` on the leader sidecar,
                only one `SharedMemoryBuffer` will be allocated for that chunk,
                each TP worker will copy a shard of the tensor to the buffer.
                    a. Inter-node
                    Each shard is sent separately to the destination sidecar.
                    b. Intra-node
                    Only send after all shards are ready.
                    (Send with no sharding)
            else:
                Only the first TP worker's `send` call will have effect
                Buffer is allocated and tensor is copied to the buffer
                (Send with no sharding)
        Note whenever intra-node forward is involved, there will be a
        reference count on the buffer (initialized as the number of dst_ranks),
        and each dst sidecar need to call `mark_done` to decrease the reference
        count.
        """
        span = trace.get_current_span()
        if not self._validate_dst_groups(list(request.dst_ranks)):
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid destination rank")

        # before decoding, we need to set the device context
        torch.cuda.set_device(self.devices[self.config.group[request.shard_rank]])
        obj = self.decoder.decode(request.data)
        if isinstance(obj, torch.Tensor):
            span.set_attribute("SidecarSender.send.type", "tensor")
            return await self._send_tensor(request, obj)
        else:
            span.set_attribute("SidecarSender.send.type", "obj")
            return await self._send_small_objects(request)

    # the difference between intra-node and inter-node send when TP is enabled
    # is that the intra-node send doesn't need to send the buffer until all
    # shards are ready
    async def _send_intra_node_buffer(
        self,
        request: sidecar_pb2.SendRequest,
        buffer: SharedMemoryBuffer,
        dst_rank: int,
    ) -> sidecar_pb2.SendResponse:
        """Send a shared memory buffer to another rank in the same node.

        Args:
            request: The original gRPC send request.
            buffer: The shared memory buffer to send.
            dst_rank: The destination rank.

        Note Copy is done here.
        """
        span = trace.get_current_span()
        span.set_attribute("SidecarSender.send.intra_node.dst_rank", dst_rank)
        id = request.id + f"-{request.chunk_id}"
        if request.shard_rank != 0:
            # silently skips
            return sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_OK)
        if self.concurrent_copy:
            # wait for other shards to copy
            await self.done_events[id].wait()

        offest = buffer.data.data_ptr() - self.config.base_ptr
        numel = buffer.data.numel()
        obj = SharedTensorHandle(offset=offest, numel=numel)
        data = self.encoder.encode(obj)
        req = sidecar_pb2.PrepareReceiveRequest(
            id=request.id,
            data=data,
            src_rank=self.sidecar_rank,
            chunk_id=request.chunk_id,
            num_chunks=request.num_chunks,
        )
        stub = self._get_grpc_stub(dst_rank)
        res = await stub.PrepareReceive(req)
        if res.status != common_pb2.Status.STATUS_OK:
            logger.error("Failed to prepare receive")
            return sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_ERROR)
        return sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_OK)

    async def _send_inter_node_buffer(
        self,
        request: sidecar_pb2.SendRequest,
        buffer: SharedMemoryBuffer,
        dst_rank: int,
    ) -> sidecar_pb2.SendResponse:
        """Send a shared memory buffer to another rank in a different node through UCX.

        Args:
            request: The original gRPC send request.
            buffer: The shared memory buffer to send.
            dst_rank: The destination rank.
        """
        span = trace.get_current_span()
        span.set_attribute("SidecarSender.send.inter_node.dst_rank", dst_rank)
        if not self.concurrent_copy and request.shard_rank != 0:
            logger.error("Error: shard_rank should be 0 when concurrent_copy is disabled")
            return sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_ERROR)
        if self.concurrent_copy:
            if not buffer.is_sharded:
                raise ValueError("Buffer is not sharded")

            shard = buffer.shards[request.shard_rank]

            # locate the shard in the buffer
            obj = ForwardTensorHandle(
                total_numel=buffer.data.numel(),
                shard_rank=request.shard_rank,
                num_shards=len(self.config.group),
            )
            data = self.encoder.encode(obj)
            req = sidecar_pb2.PrepareReceiveRequest(
                id=request.id,
                src_rank=self.sidecar_rank,
                data=data,
                chunk_id=request.chunk_id,
                num_chunks=request.num_chunks,
            )
            stub = self._get_grpc_stub(dst_rank)
            res = await stub.PrepareReceive(req)
            if res.status != common_pb2.Status.STATUS_OK:
                logger.error("Failed to prepare receive")
                return sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_ERROR)
            tag = chunk_tag(request.id, self.sidecar_rank, request.chunk_id, obj.shard_rank)
            peer = self.peers[dst_rank]
            span.add_event("send.start")
            await peer.send(buffer_from_tensor(shard.data), tag=tag)
            span.add_event("send.done")
            return sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_OK)
        else:
            # when concurrent_copy disabled, ignore sharding
            obj = ForwardTensorHandle(
                total_numel=buffer.data.numel(),
                shard_rank=0,
                num_shards=1,
            )
            data = self.encoder.encode(obj)
            req = sidecar_pb2.PrepareReceiveRequest(
                id=request.id,
                src_rank=self.sidecar_rank,
                data=data,
                chunk_id=request.chunk_id,
                num_chunks=request.num_chunks,
            )
            stub = self._get_grpc_stub(dst_rank)
            res = await stub.PrepareReceive(req)
            if res.status != common_pb2.Status.STATUS_OK:
                logger.error("Failed to prepare receive")
                return sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_ERROR)
            # do send
            tag = chunk_tag(
                request.id,
                self.sidecar_rank,
                request.chunk_id,
                obj.shard_rank,
            )
            peer = self.peers[dst_rank]
            span.add_event("send.start")
            await peer.send(buffer_from_tensor(buffer.data), tag=tag)
            span.add_event("send.done")
            return sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_OK)

    async def _concurrent_copy_send(
        self,
        request: sidecar_pb2.SendRequest,
        data: torch.Tensor,
    ) -> sidecar_pb2.SendResponse:
        """Send a slice of the tensor to the destination ranks.

        Args:
            request: The original gRPC send request.
            data: The tensor to send.
        """
        span = trace.get_current_span()
        span.set_attribute("SidecarSender.copy.mode", "concurrent")
        # this is similar to prepare_recv, the first call will allocate the buffer
        # and future calls will wait if the first call blocks
        id = request.id + f"-{request.chunk_id}"
        is_first = False
        async with self.event_lock:
            if id not in self.malloc_events:
                # first call
                is_first = True
                self.malloc_events[id] = asyncio.Event()
                self.done_events[id] = asyncio.Event()

        if is_first:
            span.add_event("allocate.start")
            buffer = await self._allocate(data.numel())
            span.add_event("allocate.done")
            buffer.create_shards(len(self.config.group))
            self.ledger[id] = SendTransferRequestState(
                id,
                buffer,
                [False] * len(self.config.group),
            )
            self.malloc_events[id].set()
            self.ledger[id].buffer = buffer
        else:
            # wait for the buffer to be allocated
            span.add_event("wait_for_allocate.start")
            event = self.malloc_events[id]
            await event.wait()
            span.add_event("wait_for_allocate.done")
            buffer = self.ledger[id].buffer

        # now we launch concurrent copy & send tasks
        shard_sidecar_ranks = self.config.group[request.shard_rank]
        stream = self.streams[shard_sidecar_ranks]

        shard = buffer.shards[request.shard_rank]

        if shard.length == 0:
            # no data to copy
            buffer.mark_shard_ready(request.shard_rank)
            if buffer.ready:
                self.done_events[id].set()
            return sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_OK)

        span.add_event("copy.start")
        with torch.cuda.stream(stream):
            shard.data.copy_(data[shard.offset : shard.offset + shard.length], non_blocking=True)
        await asyncio.to_thread(stream.synchronize)
        span.add_event("copy.done")
        buffer.mark_shard_ready(request.shard_rank)
        if buffer.ready:
            # notify copy done
            self.done_events[id].set()

        # now send
        dst_ranks = [dst_group.ranks[0] for dst_group in request.dst_ranks]
        coros = []
        intra_node_indexes = []
        for i, rank in enumerate(dst_ranks):
            if self.node_info.contains(rank):
                # intra-node send
                coros.append(self._send_intra_node_buffer(request, buffer, rank))
                intra_node_indexes.append(i)
            else:
                # inter-node send
                coros.append(self._send_inter_node_buffer(request, buffer, rank))

        if is_first and intra_node_indexes:
            self.saved_buffers[id] = buffer
            self.ref_counts[id] = len(intra_node_indexes)

        responses = await asyncio.gather(*coros)

        # for inter-node
        if not intra_node_indexes:
            async with self.sent_lock:
                self.ledger[id].shards_sent[request.shard_rank] = True
                if all(self.ledger[id].shards_sent):
                    del self.ledger[id]
                    if not intra_node_indexes:
                        await self._free(buffer)

        failed = [res.status != common_pb2.Status.STATUS_OK for res in responses]
        # if all intra node failed, remove the tracker
        if intra_node_indexes and all([failed[i] for i in intra_node_indexes]) and is_first:
            del self.saved_buffers[id]
            del self.ref_counts[id]
        if any(failed):
            return sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_ERROR)
        return sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_OK)

    async def _single_copy_send(
        self,
        request: sidecar_pb2.SendRequest,
        data: torch.Tensor,
    ) -> sidecar_pb2.SendResponse:
        """Send a tensor to the destination ranks, but only use one GPU to copy.

        Args:
            request: The original gRPC send request.
            data: The tensor to send.
        """
        span = trace.get_current_span()
        span.set_attribute("SidecarSender.copy.mode", "single")
        id = request.id + f"-{request.chunk_id}"
        if request.shard_rank != 0:
            # silently ignore the request
            span.set_attribute("SidecarSender.copy.action", "ignore")
            return sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_OK)

        numel = data.numel()
        # no contention
        span.add_event("allocate.start")
        buffer = await self._allocate(numel)
        span.add_event("allocate.done")
        span.add_event("copy.start")
        # note only the leader sidecar will reach this function
        stream = self.streams[self.sidecar_rank]
        with torch.cuda.stream(stream):
            buffer.data.copy_(data, non_blocking=True)
        await asyncio.to_thread(stream.synchronize)
        span.add_event("copy.done")

        dst_ranks = [dst_group.ranks[0] for dst_group in request.dst_ranks]
        coros = []
        intra_node_indexes = []
        for i, rank in enumerate(dst_ranks):
            if self.node_info.contains(rank):
                # intra-node send
                coros.append(self._send_intra_node_buffer(request, buffer, rank))
                intra_node_indexes.append(i)
            else:
                # inter-node send
                coros.append(self._send_inter_node_buffer(request, buffer, rank))

        if intra_node_indexes:
            # save before sending done
            span.set_attribute("SidecarSender.copy.action", "save")
            self.saved_buffers[id] = buffer
            self.ref_counts[id] = len(intra_node_indexes)

        responses = await asyncio.gather(*coros)

        if not intra_node_indexes:
            # if no intra-node send, we can safely free the buffer
            await self._free(buffer)

        failed = [res.status != common_pb2.Status.STATUS_OK for res in responses]
        # if all intra node failed, remove the tracker
        if intra_node_indexes and all([failed[i] for i in intra_node_indexes]):
            logger.warning("All intra-node send failed, removing the tracker")
            del self.saved_buffers[id]
            del self.ref_counts[id]
        if any(failed):
            return sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_ERROR)
        return sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_OK)

    async def _send_tensor(
        self,
        request: sidecar_pb2.SendRequest,
        data: torch.Tensor,
    ) -> sidecar_pb2.SendResponse:
        """Send a tensor to the destination ranks.

        Args:
            request: The original gRPC send request.
            data: The tensor to send.
        """
        if self.concurrent_copy:
            return await self._concurrent_copy_send(request, data)
        return await self._single_copy_send(request, data)
