"""Sidecar api to be usd by the task exucutors: Enc Server, vLLM Server, etc."""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import json
import os
import weakref
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
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
from cornserve.sidecar.constants import grpc_url_from_rank, shm_filename
from cornserve.sidecar.schema import SidecarConfig
from cornserve.sidecar.serde import MsgpackDecoder, MsgpackEncoder, SharedTensorHandle
from cornserve.sidecar.utils import device_from_rank, init_shmem
from cornserve.utils import set_ulimit

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

GrpcInstrumentorClient().instrument()
GrpcAioInstrumentorClient().instrument()
ThreadingInstrumentor().instrument()


@lru_cache(maxsize=1)
def _is_mocking() -> bool:
    """Check if we are mocking the sidecar client."""
    env = os.environ.get("CORNSERVE_MOCK_SIDECAR", "0")
    return env == "1" or env.lower() == "true"


@lru_cache(maxsize=1)
def _get_mock_mapping() -> dict[str, Path]:
    """Get the mapping from sidecar id to local file path for mocking."""
    mapping = os.environ.get("CORNSERVE_MOCK_SIDECAR_MAPPING", "")
    if not mapping:
        return {}
    try:
        mapping_dict = json.loads(mapping)
        return {k: Path(v) for k, v in mapping_dict.items()}
    except json.JSONDecodeError:
        logger.exception("Failed to decode CORNSERVE_MOCK_SIDECAR_MAPPING, should be a json string")
        return {}


class Sidecar:
    """The sidecar client to send or receive data to/from other sidecars."""

    supported_classes = [str, bytes, int, float, bool, torch.Tensor, dict]

    def __init__(
        self,
        config: SidecarConfig,
    ) -> None:
        """Initialize the sidecar receiver client that receives data from a sender sidecar.

        Args:
            config: The configuration for the sidecar.
        """
        set_ulimit()

        if _is_mocking():
            logger.warning("Mocking sidecar client, no actual grpc connection will be made.")
            return

        self.config = config
        self.sidecar_rank = config.sidecar_rank
        self.group = config.group
        self.send_dtype = config.get_send_dtype()
        self.recv_dtype = config.get_recv_dtype()

        # register the sidecar to the server, provide hint and grouping
        # note when using TP, only talks to the head sidecar
        assert self.group, "Sidecar group should not be empty"

        # sidecar rank -> sync grpc channels
        self.channels = {}
        # sidecar rank -> sync grpc stubs
        self.stubs = {}
        # sidecar rank -> async grpc channels
        self.aio_channels = {}
        # sidecar rank -> async grpc stubs
        self.aio_stubs = {}

        self.stub = self._get_grpc_stub(min(self.group))
        self.aio_stub = self._get_aio_grpc_stub(min(self.group))

        request = sidecar_pb2.RegisterRequest(
            rank=self.sidecar_rank,
            group=self.group,
            send_dtype=str(self.send_dtype).split(".")[-1],
            recv_dtype=str(self.recv_dtype).split(".")[-1],
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
            partition_bytes=response.shm_size,
        )

        self.base_ptr = self.full_tensor.data_ptr()
        self.device = device_from_rank(self.shard_rank)

        self.msgpack_encoder = MsgpackEncoder()
        self.msgpack_decoder = MsgpackDecoder()

        self.worker_pool = ThreadPoolExecutor(max_workers=self.config.max_workers, thread_name_prefix="sidecar-worker")

        self._finalizer = weakref.finalize(self, self.__del__)

    def _get_grpc_stub(self, sidecar_rank: int) -> sidecar_pb2_grpc.SidecarStub:
        """Get the grpc stub for the given sidecar rank."""
        if sidecar_rank not in self.stubs:
            if sidecar_rank not in self.channels:
                channel = grpc.insecure_channel(grpc_url_from_rank(sidecar_rank))
                self.channels[sidecar_rank] = channel
            self.stubs[sidecar_rank] = sidecar_pb2_grpc.SidecarStub(self.channels[sidecar_rank])
        return self.stubs[sidecar_rank]

    def _get_aio_grpc_stub(self, sidecar_rank: int) -> sidecar_pb2_grpc.SidecarStub:
        """Get the aio grpc stub for the given sidecar rank."""
        if sidecar_rank not in self.aio_stubs:
            if sidecar_rank not in self.aio_channels:
                channel = grpc.aio.insecure_channel(grpc_url_from_rank(sidecar_rank))
                self.aio_channels[sidecar_rank] = channel
            self.aio_stubs[sidecar_rank] = sidecar_pb2_grpc.SidecarStub(self.aio_channels[sidecar_rank])
        return self.aio_stubs[sidecar_rank]

    @tracer.start_as_current_span(name="Sidecar.send")
    def send(
        self,
        data: Any,
        id: str,
        dst_sidecar_ranks: list[list[int]],
        chunk_id: int = 0,
        num_chunks: int = 1,
        stream: bool = False,
    ) -> None:
        """Send some data to other sidecars.

        Args:
            data: The data to send. Can be a tensor or any other supported type.
            id: The id of the data. This is used to identify the data in the sidecar.
            dst_sidecar_ranks: The ranks of the sidecars to send the data to. This is a list of lists,
                where each list is a sidecar TP group.
            chunk_id: The chunk id of the data when only sending a chunk.
            num_chunks: The number of chunks the entire data is split into.
            stream: Whether to stream the data. If True, `num_chunks` is ignored, and the sender needs
                to call `close_stream` to mark the end of the stream.
        """
        if _is_mocking():
            mapping = _get_mock_mapping()
            key = f"{id}-{chunk_id}"
            if key not in mapping:
                raise ValueError(f"Mocking sidecar but key {key} not in mapping")
            local_path = mapping[key]
            if isinstance(data, torch.Tensor):
                torch.save(data, local_path)
            else:
                # save as json
                str_data = json.dumps(data)
                local_path.write_text(str_data)
            return

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
        if stream:
            logger.info("Streaming chunks")
            num_chunks = 0

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
        logger.info(
            "Sending shard %d of chunk %d in req %s to ranks %s", self.shard_rank, chunk_id, id, dst_sidecar_ranks
        )
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
            if isinstance(obj, torch.Tensor):
                torch.cuda.ipc_collect()
        else:
            logger.error("Failed to send data with id %s", id)

    @tracer.start_as_current_span(name="Sidecar.close_stream")
    def close_stream(
        self,
        id: str,
        num_chunks: int,
        dst_sidecar_ranks: list[list[int]],
    ) -> None:
        """Signal the end of a stream to the sidecar server.

        Args:
            id: The id of the data. This is used to identify the data in the sidecar.
            num_chunks: The total number of chunks sent in the stream.
            dst_sidecar_ranks: The ranks of the sidecars to send the data to. This is a list of lists,
                where each list is a sidecar TP group.
        """
        if _is_mocking():
            return

        span = trace.get_current_span()
        span.set_attribute("sidecar.close_stream.id", id)

        dst_ranks = [min(group) for group in dst_sidecar_ranks]

        for sidecar_rank in dst_ranks:
            if sidecar_rank < 0:
                raise ValueError("Invalid sidecar rank")
            future = self.worker_pool.submit(
                self._close_stream_worker,
                id,
                num_chunks,
                sidecar_rank,
            )
            future.add_done_callback(lambda f: f.result())

    @tracer.start_as_current_span(name="Sidecar._close_stream_worker")
    def _close_stream_worker(
        self,
        id: str,
        num_chunks: int,
        sidecar_rank: int,
    ) -> None:
        """The worker function to close a stream in the sidecar server.

        Args:
            id: The id of the data. This is used to identify the data in the sidecar.
            num_chunks: The toatal number of chunks sent in the stream.
            sidecar_rank: The sidecar rank to notify about the stream closure.
        """
        request = sidecar_pb2.CloseStreamRequest(
            id=id,
            num_chunks=num_chunks,
        )
        grpc_stub = self._get_grpc_stub(sidecar_rank)
        response = grpc_stub.CloseStream(request)
        if response.status == common_pb2.Status.STATUS_OK:
            logger.info("Closed stream %s successfully with %d chunks", id, num_chunks)
        else:
            logger.error("Failed to close stream %s with %d chunks", id, num_chunks)

    def _mock_recv(self, id: str, chunk_id: int = 0) -> Any:
        if _is_mocking():
            mapping = _get_mock_mapping()
            key = f"{id}-{chunk_id}"
            if key not in mapping:
                raise ValueError(f"Mocking sidecar but key {key} not in mapping")
            local_path = mapping[key]
            if not local_path.exists():
                # we return None to simulate the end of stream
                return None
            suffix = local_path.suffix.lower()
            if suffix in [".pt", ".pth"]:
                tensor = torch.load(local_path, map_location="cpu")
                return tensor
            else:
                # assumes json
                str_data = local_path.read_text()
                return json.loads(str_data)

    @tracer.start_as_current_span(name="Sidecar.recv")
    async def recv(self, id: str, chunk_id: int = 0) -> Any:
        """Receive data from the sidecar server.

        Receive (either sync or async) is idompotent.

        Args:
            id: The id of the data.
            chunk_id: The chunk id of the data to receive.
        """
        if _is_mocking():
            return self._mock_recv(id, chunk_id)

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
            cbuf = (ctypes.c_byte * obj.numel * self.recv_dtype.itemsize).from_address(self.base_ptr + obj.offset)
            tensor = torch.frombuffer(cbuf, dtype=self.recv_dtype, count=obj.numel)
            tensor_view = tensor.view(self.config.get_recv_tensor_shape())
            logger.info(
                "Received shard %d of chunk %d in req %s successfully (shape %s)",
                self.shard_rank,
                chunk_id,
                id,
                list(tensor_view.shape),
            )
            return tensor_view
        else:
            return obj

    @tracer.start_as_current_span(name="Sidecar.recv_sync")
    def recv_sync(self, id: str, chunk_id: int = 0) -> Any:
        """Receive data from the sidecar server synchronously.

        Receive (either sync or async) is idompotent.
        When the data is already `recv`-ed, this function will return immediately.

        Args:
            id: The id of the data.
            chunk_id: The chunk id of the data to receive.
        """
        if _is_mocking():
            return self._mock_recv(id, chunk_id)

        span = trace.get_current_span()
        span.set_attribute("sidecar.read.id", id)
        span.set_attribute("sidecar.read.chunk_id", chunk_id)
        request = sidecar_pb2.ReceiveRequest(id=id, chunk_id=chunk_id)
        response = self.stub.Receive(request)
        if response.status != common_pb2.Status.STATUS_OK:
            raise ValueError(f"Failed to receive data with id {id}")

        obj = self.msgpack_decoder.decode(response.data)
        if isinstance(obj, SharedTensorHandle):
            cbuf = (ctypes.c_byte * obj.numel * self.recv_dtype.itemsize).from_address(self.base_ptr + obj.offset)
            tensor = torch.frombuffer(cbuf, dtype=self.recv_dtype, count=obj.numel)
            tensor_view = tensor.view(self.config.get_recv_tensor_shape())
            logger.info(
                "Sync read shard %d of chunk %d in req %s successfully (shape %s)",
                self.shard_rank,
                chunk_id,
                id,
                list(tensor_view.shape),
            )
            return tensor_view
        else:
            return obj

    @tracer.start_as_current_span(name="Sidecar.mark_done")
    async def mark_done(self, id: str, chunk_id: int = 0) -> None:
        """Mark a tensor as done in the sidecar server, which will free the shared memory buffer.

        Args:
            id: The id of the data.
            chunk_id: The chunk id of the data to mark as done.
        """
        if _is_mocking():
            return

        span = trace.get_current_span()
        span.set_attribute("sidecar.mark_done.id", id)
        request = sidecar_pb2.MarkDoneRequest(id=id, chunk_id=chunk_id)
        response = await self.aio_stub.MarkDone(request)
        if response.status == common_pb2.Status.STATUS_OK:
            logger.debug("Request %s marked done", id)

    def _mark_done_worker(self, id: str, chunk_id: int = 0) -> None:
        """Mark a tensor as done in the sidecar server to free the shared memory buffer.

        Args:
            id: The id of the data.
            chunk_id: The chunk id of the data to mark as done.
        """
        span = trace.get_current_span()
        span.set_attribute("sidecar.mark_done.id", id)
        request = sidecar_pb2.MarkDoneRequest(id=id, chunk_id=chunk_id)
        response = self.stub.MarkDone(request)
        if response.status == common_pb2.Status.STATUS_OK:
            logger.debug("Request %s marked done", id)
        else:
            logger.error("Failed to mark request %s done", id)

    @tracer.start_as_current_span(name="Sidecar.mark_done_sync")
    def mark_done_sync(self, id: str, chunk_id: int = 0) -> None:
        """Synchronously mark a tensor as done in the sidecar server to free the shared memory buffer.

        Args:
            id: The id of the data.
            chunk_id: The chunk id of the data to mark as done.
        """
        if _is_mocking():
            return

        future = self.worker_pool.submit(
            self._mark_done_worker,
            id,
            chunk_id,
        )
        future.add_done_callback(lambda f: f.result())

    def __del__(self) -> None:
        """Unlink the shared memory buffer."""
        if _is_mocking():
            return

        if not hasattr(self, "channel"):
            return
        logger.warning("Sidecar not shutdown properly, remember to call shutdown()")
        try:
            for channel in self.channels.values():
                channel.close()
                del channel
            for aio_channel in self.aio_channels.values():
                del aio_channel
        except Exception:
            pass

    async def shutdown(self) -> None:
        """Shutdown the sidecar client."""
        if _is_mocking():
            return

        try:
            for channel in self.channels.values():
                channel.close()
                del channel
            for aio_channel in self.aio_channels.values():
                await aio_channel.close()
                del aio_channel
            self.worker_pool.shutdown(wait=True)
        except Exception:
            pass

    def shutdown_sync(self) -> None:
        """Synchronously shutdown the sidecar client."""
        if _is_mocking():
            return

        with contextlib.suppress(Exception):
            asyncio.run(self.shutdown())
