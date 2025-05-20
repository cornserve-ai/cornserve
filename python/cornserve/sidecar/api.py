"""Sidecar api to be usd by the task exucutors: Enc Server, vLLM Server, etc."""

from __future__ import annotations

import ctypes
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import grpc
import torch
from opentelemetry import trace
from opentelemetry.instrumentation.grpc import (
    GrpcAioInstrumentorClient,
    GrpcInstrumentorClient,
)
from opentelemetry.instrumentation.threading import ThreadingInstrumentor

from cornserve.logging import get_logger
from cornserve.services.pb import common_pb2, sidecar_pb2, sidecar_pb2_grpc
from cornserve.sidecar.schema import SidecarConfig
from cornserve.sidecar.serde import MsgpackDecoder, MsgpackEncoder, SharedTensorHandle
from cornserve.sidecar.utils import (
    device_from_rank,
    grpc_url_from_rank,
    init_shmem,
    shm_filename,
)

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

GrpcInstrumentorClient().instrument()
GrpcAioInstrumentorClient().instrument()
ThreadingInstrumentor().instrument()


class Sidecar:
    """The sidecar client to send or receive data to/from other sidecars."""

    supported_classes = [str, bytes, int, float, bool, torch.Tensor]

    def __init__(
        self,
        config: SidecarConfig,
    ) -> None:
        """Initialize the sidecar receiver client that receives data from a sender sidecar.

        Args:
            config: The configuration for the sidecar.
        """
        self.config = config
        self.sidecar_rank = config.sidecar_rank
        self.group = config.group
        self.dtype = config.get_dtype()

        # register the sidecar to the server, provide hint and grouping
        # note when using TP, only talks to the head sidecar
        assert self.group, "Sidecar group should not be empty"
        self.channel = grpc.insecure_channel(grpc_url_from_rank(min(self.group)))
        self.stub = sidecar_pb2_grpc.SidecarStub(self.channel)
        self.aio_channel = grpc.aio.insecure_channel(grpc_url_from_rank(min(self.group)))
        self.aio_stub = sidecar_pb2_grpc.SidecarStub(self.aio_channel)

        request = sidecar_pb2.RegisterRequest(
            rank=self.sidecar_rank,
            group=self.group,
            dtype=str(self.dtype).split(".")[-1],
            send_slot_numel=config.get_send_slot_numel(),
            recv_slot_numel=config.get_recv_slot_numel(),
            concurrent_copy=config.concurrent_copy,
        )

        response = self.stub.Register(request)
        assert response.shm_size > 0, "Failed to register sidecar"

        self.shard_rank = response.local_rank

        self.full_tensor, _ = init_shmem(
            filename=shm_filename(),
            local_ranks=[response.local_rank],
            num_local_sidecars=response.num_local_sidecars,
            partition_numel=response.shm_size * 2,
            dtype=self.dtype,
        )

        self.base_ptr = self.full_tensor.data_ptr()
        self.device = device_from_rank(self.shard_rank)

        self.msgpack_encoder = MsgpackEncoder()
        self.msgpack_decoder = MsgpackDecoder()

        # sender specific attributes
        self.worker_pool = ThreadPoolExecutor(
            max_workers=self.config.max_workers, thread_name_prefix="sidecar-send-worker"
        )

        self._finalizer = weakref.finalize(self, self.__del__)

    @tracer.start_as_current_span(name="Sidecar.send")
    def send(
        self,
        data: Any,
        id: str,
        dst_sidecar_ranks: list[list[int]],
        chunk_id: int = 0,
        num_chunks: int = 1,
    ) -> None:
        """Send some data to other sidecars.

        Args:
            data: The data to send. Can be a tensor or any other supported type.
            id: The id of the data. This is used to identify the data in the sidecar.
            dst_sidecar_ranks: The ranks of the sidecars to send the data to. This is a list of lists,
                where each list is a sidecar TP group.
            chunk_id: The chunk id of the data when only sending a chunk.
            num_chunks: The number of chunks the entire data is split into.
        """
        # need to pass a shared pytorch tensor handle
        # assume broadcast
        if any(rank == self.config.sidecar_rank for group in dst_sidecar_ranks for rank in group):
            raise ValueError("Cannot send to self")
        if any(rank < 0 for group in dst_sidecar_ranks for rank in group):
            raise ValueError("Invalid sidecar rank")
        if not any(isinstance(data, cls) for cls in self.supported_classes):
            raise ValueError(f"Unsupported data type: {type(data)}")
        if isinstance(data, torch.Tensor) and data.device != self.device:
            raise ValueError(f"Tensor must be on {self.device}, but got {data.device}")

        span = trace.get_current_span()
        span.set_attribute("sidecar.send.id", id)

        future = self.worker_pool.submit(
            self._send_worker,
            data,
            id,
            dst_sidecar_ranks,
            chunk_id,
            num_chunks,
        )
        future.add_done_callback(lambda f: f.result())

    @tracer.start_as_current_span(name="Sidecar._send_worker")
    def _send_worker(
        self,
        obj: Any,
        id: str,
        dst_sidecar_ranks: list[list[int]],
        chunk_id: int,
        num_chunks: int,
    ) -> None:
        """The worker function to send data to other sidecars.

        Args:
            obj: The data to send. Can be a tensor or any other supported type.
            id: The id of the data. This is used to identify the data in the sidecar.
            dst_sidecar_ranks: The ranks of the sidecars to send the data to.
            chunk_id: The chunk id of the data when only sending a chunk.
            num_chunks: The number of chunks the entire data is split into.
        """
        if isinstance(obj, torch.Tensor):
            if not obj.is_cuda:
                # TODO: support CPU tensors
                raise ValueError("Tensor must be on GPU")
            if not obj.is_contiguous():
                logger.warning("Tensor is not contiguous, copying to contiguous tensor will introduce overhead")
                obj = obj.contiguous()
            obj = obj.view(-1)

        data = self.msgpack_encoder.encode(obj)
        dst_ranks = [sidecar_pb2.RankGroup(ranks=group) for group in dst_sidecar_ranks]
        request = sidecar_pb2.SendRequest(
            id=id,
            dst_ranks=dst_ranks,
            data=data,
            shard_rank=self.shard_rank,
            chunk_id=chunk_id,
            num_chunks=num_chunks,
        )
        response = self.stub.Send(request)
        if response.status == common_pb2.Status.STATUS_OK:
            logger.info("Sent shard %d of chunk %d in req %s successfully", self.shard_rank, chunk_id, id)
        else:
            logger.error("Failed to send data with id %s", id)

    @tracer.start_as_current_span(name="Sidecar.recv")
    async def recv(self, id: str, chunk_id: int = 0) -> Any:
        """Receive data from the sidecar server.

        Receive (either sync or async) is idompotent.

        Args:
            id: The id of the data.
            chunk_id: The chunk id of the data to receive.
        """
        # TODO (Jeff): Async Generator
        span = trace.get_current_span()
        span.set_attribute("sidecar.recv.id", id)
        span.set_attribute("sidecar.recv.chunk_id", chunk_id)
        request = sidecar_pb2.ReceiveRequest(id=id, chunk_id=chunk_id)
        response = await self.aio_stub.Receive(request)
        if response.status != common_pb2.Status.STATUS_OK:
            raise ValueError(f"Failed to receive data with id {id}")

        obj = self.msgpack_decoder.decode(response.data)
        if isinstance(obj, SharedTensorHandle):
            cbuf = (ctypes.c_byte * obj.numel * self.dtype.itemsize).from_address(self.base_ptr + obj.offset)
            tensor = torch.frombuffer(cbuf, dtype=self.dtype, count=obj.numel)
            return tensor.view(self.config.get_recv_tensor_shape())
        else:
            return obj

    @tracer.start_as_current_span(name="Sidecar.read")
    def recv_sync(self, id: str, chunk_id: int = 0) -> Any:
        """Receive data from the sidecar server synchronously.

        Receive (either sync or async) is idompotent.
        When the data is already `recv`-ed, this function will return immediately.

        Args:
            id: The id of the data.
            chunk_id: The chunk id of the data to receive.
        """
        span = trace.get_current_span()
        span.set_attribute("sidecar.read.id", id)
        span.set_attribute("sidecar.read.chunk_id", chunk_id)
        request = sidecar_pb2.ReceiveRequest(id=id, chunk_id=chunk_id)
        response = self.stub.Receive(request)
        if response.status != common_pb2.Status.STATUS_OK:
            raise ValueError(f"Failed to receive data with id {id}")

        obj = self.msgpack_decoder.decode(response.data)
        if isinstance(obj, SharedTensorHandle):
            cbuf = (ctypes.c_byte * obj.numel * self.dtype.itemsize).from_address(self.base_ptr + obj.offset)
            tensor = torch.frombuffer(cbuf, dtype=self.dtype, count=obj.numel)
            return tensor.view(self.config.get_recv_tensor_shape())
        else:
            return obj

    @tracer.start_as_current_span(name="Sidecar.mark_done")
    async def mark_done(self, id: str, chunk_id: int = 0) -> None:
        """Mark a tensor as done in the sidecar server, which will free the shared memory buffer.

        Args:
            id: The id of the data.
            chunk_id: The chunk id of the data to mark as done.
        """
        span = trace.get_current_span()
        span.set_attribute("sidecar.mark_done.id", id)
        request = sidecar_pb2.MarkDoneRequest(id=id, chunk_id=chunk_id)
        response = await self.aio_stub.MarkDone(request)
        if response.status == common_pb2.Status.STATUS_OK:
            logger.debug("Request %s marked done", id)

    def __del__(self) -> None:
        """Unlink the shared memory buffer."""
        if not hasattr(self, "channel"):
            return
        logger.warning("Sidecar not shutdown properly, remember to call shutdown()")
        try:
            del self.channel
            del self.aio_channel
        except Exception:
            pass

    async def shutdown(self) -> None:
        """Unlink the shared memory buffer."""
        try:
            self.channel.close()
            await self.aio_channel.close()
            del self.channel
            del self.aio_channel
        except Exception:
            pass
