from enum import Enum
import pickle
from typing import cast
from functools import reduce
from operator import mul

import grpc
import torch

from cornserve.logging import get_logger
from cornserve.services.pb import comm_sidecar_pb2, comm_sidecar_pb2_grpc, common_pb2

logger = get_logger(__name__)

"""
Sidecar api to be usd by the task exucutors: Enc Server, vLLM Server, etc.
Currently all ranks are local ranks
"""

SHM_SIZE = 2**27


def shm_fn_from_rank(rank: int) -> str:
    return f"/dev/shm/sc_shm_{rank}"


def device_from_rank(rank: int) -> torch.device:
    return torch.device(f"cuda:{rank}")


def grpc_channel_from_rank(rank: int) -> str:
    # sidecar_rank
    return f"sidecar-{rank}.torch-headless.cornserve.svc.cluster.local:{10000+rank}"


def init_shmem(shm_fn: str, size: int, dtype: torch.dtype) -> torch.Tensor:
    shared_tensor = torch.from_file(
        filename=shm_fn,
        shared=True,
        size=size,
        dtype=dtype,
    )
    return shared_tensor


class TensorLayout(Enum):
    FULL = 0


class TensorSidercar:
    def __init__(self, sidecar_rank: int, dtype: torch.dtype) -> None:
        self.sidecar_rank = sidecar_rank
        self.dtype = dtype
        self.channel = grpc_channel_from_rank(self.sidecar_rank)

    def unregister(self):
        channel = grpc.insecure_channel(self.channel)
        stub = comm_sidecar_pb2_grpc.CommSidecarStub(channel)
        request = comm_sidecar_pb2.UnregisterRequest()
        response = stub.Unregister(request)
        assert (
            response.status == common_pb2.Status.STATUS_OK
        ), "Failed to unregister sidecar"
        logger.info("Unregistered sidecar")


class TensorSidecarReceiver(TensorSidercar):
    def __init__(
        self, sidecar_rank: int, shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        super().__init__(sidecar_rank, dtype)
        self.tensor_shape = shape

        self._post_init()
        self._register()

    def _post_init(self) -> None:
        self.tensor_size = reduce(mul, self.tensor_shape)
        pass

    def _register(self) -> None:
        channel = grpc.insecure_channel(self.channel)
        stub = comm_sidecar_pb2_grpc.CommSidecarStub(channel)
        request = comm_sidecar_pb2.RegisterReceiverRequest(
            shape=self.tensor_shape,
            dtype=str(self.dtype).split(".")[-1],
        )
        response = stub.RegisterReceiver(request)
        assert response.gpu_rank >= 0, "Failed to register sidecar"

        self.gpu_rank = response.gpu_rank
        self.shm_fn = shm_fn_from_rank(self.gpu_rank)
        self.device = device_from_rank(self.gpu_rank)
        self.shared_tensor = init_shmem(self.shm_fn, SHM_SIZE, self.dtype)

    async def async_recv(self, req_id: int) -> torch.Tensor:
        async with grpc.aio.insecure_channel(self.channel) as channel:
            stub = comm_sidecar_pb2_grpc.CommSidecarStub(channel)
            request = comm_sidecar_pb2.ReceiveRequest(
                request_id=req_id,
            )
            response = await stub.Receive(request)
            assert response.slot >= 0, "Failed to receive data"
            return self.shared_tensor[
                response.slot
                * self.tensor_size : (response.slot + 1)
                * self.tensor_size
            ].view(self.tensor_shape)

    def recv(
        self,
        req_id: int,
    ) -> torch.Tensor:
        # This is not recommended, when the shared buffer is not large enough
        # or consumption speed is not high enough, there will be no more slots
        # currently this case is not handled and the system will break
        with grpc.insecure_channel(self.channel) as channel:
            stub = comm_sidecar_pb2_grpc.CommSidecarStub(channel)
            request = comm_sidecar_pb2.ReceiveRequest(
                request_id=req_id,
            )
            response = stub.Receive(request)
            assert response.slot >= 0, "Failed to receive data"
            return self.shared_tensor[
                response.slot
                * self.tensor_size : (response.slot + 1)
                * self.tensor_size
            ].view(self.tensor_shape)

    async def async_mark_done(self, req_id: int) -> None:
        async with grpc.aio.insecure_channel(self.channel) as channel:
            stub = comm_sidecar_pb2_grpc.CommSidecarStub(channel)
            request = comm_sidecar_pb2.MarkDoneRequest(
                request_id=req_id,
            )
            response = await stub.MarkDone(request)
            if response.status == common_pb2.Status.STATUS_OK:
                logger.info(f"Request {req_id} marked done")
            else:
                logger.error(f"Failed to mark request {req_id} done")

    def mark_done(self, req_id: int) -> None:
        with grpc.insecure_channel(self.channel) as channel:
            stub = comm_sidecar_pb2_grpc.CommSidecarStub(channel)
            request = comm_sidecar_pb2.MarkDoneRequest(
                request_id=req_id,
            )
            response = stub.MarkDone(request)
            if response.status == common_pb2.Status.STATUS_OK:
                logger.info(f"Request {req_id} marked done")
            else:
                logger.error(f"Failed to mark request {req_id} done")


class TensorSidecarSender(TensorSidercar):
    def __init__(
        self,
        sidecar_rank: int,
        chunk_shape: tuple[int, ...],
        dtype: torch.dtype,
        shard_rank: int = 0,
        num_shards: int = 1,
        layout: TensorLayout = TensorLayout.FULL,
    ) -> None:
        super().__init__(sidecar_rank, dtype)
        logger.info("instantiating sidecar sender")
        assert shard_rank < num_shards, "Invalid shard rank"
        self.chunk_shape = chunk_shape
        self.shard_rank = shard_rank
        self.num_shards = num_shards
        self.layout = layout
        self._post_init()
        self._register()

    def _post_init(self) -> None:
        logger.info("calling post init")
        self.chunk_size = reduce(mul, self.chunk_shape)
        self.num_slots = SHM_SIZE // self.chunk_size
        self.occupancy = [0 for _ in range(self.num_slots)]

    def _register(self) -> None:
        logger.info("calling register")
        channel = grpc.insecure_channel(self.channel)
        stub = comm_sidecar_pb2_grpc.CommSidecarStub(channel)
        request = comm_sidecar_pb2.RegisterSenderRequest(
            chunk_shape=self.chunk_shape,
            dtype=str(self.dtype).split(".")[-1],
            shard_rank=self.shard_rank,
            num_shards=self.num_shards,
            layout=self.layout.value,
        )
        response = stub.RegisterSender(request)
        assert (
            response.gpu_rank >= 0
        ), f"Failed to register sidecar with gpu{response.gpu_rank}"
        logger.info(f"Registered sidecar with gpu{response.gpu_rank}")

        self.gpu_rank = response.gpu_rank
        self.shm_fn = shm_fn_from_rank(self.gpu_rank)
        self.device = device_from_rank(self.gpu_rank)
        self.stream = cast(torch.cuda.Stream, torch.cuda.Stream(device=self.device))
        self.shared_tensor = init_shmem(self.shm_fn, SHM_SIZE, self.dtype)

    def find_slot(self) -> int:
        # no locking enforced bc this is single thread
        # possibly make this wait, and the same with the
        # find_slot on the server side
        # but this basically means lock + condition variable
        for i, occ in enumerate(self.occupancy):
            if occ == 0:
                self.occupancy[i] = 1
                return i
        return -1

    def release_slot(self, slot: int) -> None:
        # no locking enforced bc this is single thread
        self.occupancy[slot] = 0

    async def async_send(
        self,
        chunk: torch.Tensor,
        req_id: int,
        chunk_id: int,
        num_chunks: int,
        dst_sidecar_rank: int,
    ) -> None:
        assert chunk_id < num_chunks, "Chunk id out of bounds"
        assert chunk.device == self.device, "Device mismatch"
        assert (
            dst_sidecar_rank != self.sidecar_rank
        ), f"Cannot send to self {dst_sidecar_rank} and self is {self.sidecar_rank}"
        assert chunk.shape == self.chunk_shape, "Shape mismatch"
        assert chunk.dtype == self.dtype, "Dtype mismatch"

        logger.info("current slot occupancy: " + str(self.occupancy))

        slot = self.find_slot()
        assert slot >= 0, "No free slots available, increase SHM_SIZE"

        cuda_event = torch.cuda.Event(interprocess=True)
        with torch.cuda.stream(self.stream):
            self.shared_tensor[
                slot * self.chunk_size : (slot + 1) * self.chunk_size
            ].copy_(chunk.view(-1), non_blocking=True)
            cuda_event.record(self.stream)

        ipc_handle = cuda_event.ipc_handle()
        async with grpc.aio.insecure_channel(self.channel) as channel:
            stub = comm_sidecar_pb2_grpc.CommSidecarStub(channel)
            request = comm_sidecar_pb2.SendRequest(
                chunk_slot=slot,
                ipc_handle=pickle.dumps(ipc_handle),
                request_id=req_id,
                chunk_id=chunk_id,
                num_chunks=num_chunks,
                shard_rank=self.shard_rank,
                num_shards=self.num_shards,
                dst_sidecar_rank=dst_sidecar_rank,
            )
            response = await stub.Send(request)
            self.release_slot(slot)
            if response.status == common_pb2.Status.STATUS_OK:
                logger.info(
                    f"Sent chunk {chunk_id} in shard {self.shard_rank} out of shards {num_chunks} of req_id {req_id} successfully"
                )
            else:
                logger.error("Failed to send data")

    def send(
        self,
        chunk: torch.Tensor,
        req_id: int,
        chunk_id: int,
        num_chunks: int,
        dst_sidecar_rank: int,
    ) -> None:
        assert chunk_id < num_chunks, "Chunk id out of bounds"
        assert chunk.device == self.device, "Device mismatch"
        assert (
            dst_sidecar_rank != self.sidecar_rank
        ), f"Cannot send to self {dst_sidecar_rank} and self is {self.sidecar_rank}"
        assert chunk.shape == self.chunk_shape, "Shape mismatch"
        assert chunk.dtype == self.dtype, "Dtype mismatch"

        logger.info("current slot occupancy: " + str(self.occupancy))

        slot = self.find_slot()
        assert slot >= 0, "No free slots available, increase SHM_SIZE"

        cuda_event = torch.cuda.Event(interprocess=True)
        with torch.cuda.stream(self.stream):
            self.shared_tensor[
                slot * self.chunk_size : (slot + 1) * self.chunk_size
            ].copy_(chunk.view(-1), non_blocking=True)
            cuda_event.record(self.stream)

        ipc_handle = cuda_event.ipc_handle()
        with grpc.insecure_channel(self.channel) as channel:
            stub = comm_sidecar_pb2_grpc.CommSidecarStub(channel)
            request = comm_sidecar_pb2.SendRequest(
                chunk_slot=slot,
                ipc_handle=pickle.dumps(ipc_handle),
                request_id=req_id,
                chunk_id=chunk_id,
                num_chunks=num_chunks,
                shard_rank=self.shard_rank,
                num_shards=self.num_shards,
                dst_sidecar_rank=dst_sidecar_rank,
            )
            response = stub.Send(request)
            self.release_slot(slot)
            if response.status == common_pb2.Status.STATUS_OK:
                logger.info(
                    f"Sent chunk {chunk_id} in shard {self.shard_rank} out of shards {num_chunks} of req_id {req_id} successfully"
                )
            else:
                logger.error("Failed to send data")
