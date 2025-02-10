from enum import Enum
from typing import Tuple, List

import torch
import grpc
import pickle
from cornserve.services.pb import comm_sidecar_pb2, comm_sidecar_pb2_grpc, common_pb2

from cornserve.logging import get_logger

logger = get_logger(__name__)

"""
Sidecar api to be usd by the task exucutors: Enc Server, vLLM Server, etc.
Currently all ranks are local ranks
"""


class SidecarMode(Enum):
    SEND = 0
    RECV = 1


def shm_fn_from_rank(rank: int) -> str:
    return f"/dev/shm/sc_shm_{rank}"


def device_from_rank(rank: int) -> torch.device:
    return torch.device(f"cuda:{rank}")


def grpc_channel_from_rank(rank: int) -> str:
    return f"sidecar-{rank}.torch-headless.cornserve.svc.cluster.local:{10000+rank}"


def init_shmem(shm_fn: str, size: int, dtype: torch.dtype) -> torch.Tensor:
    # TODO: ring buffer
    shared_tensor = torch.from_file(
        filename=shm_fn,
        shared=True,
        size=size,
        dtype=dtype,
    )
    return shared_tensor


class TensorSidecar:
    def __init__(
        self,
        mode: SidecarMode,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        rank: int,  # local rank
    ):
        self.mode = mode
        self.shape = shape
        # for contiguous tensors, shape matters for only copy in sending and only return in receiving

        self.rank = rank  # possibly use a shared counter
        self.dtype = dtype
        self._post_init()

    def _post_init(self) -> None:
        self.shm_fn = shm_fn_from_rank(self.rank)
        self.device = device_from_rank(self.rank)
        self.channel = grpc_channel_from_rank(self.rank)

        from typing import cast

        self.stream = cast(torch.cuda.Stream, torch.cuda.Stream(device=self.device))

        from functools import reduce
        from operator import mul

        self.shm_size = reduce(mul, self.shape)
        self.shared_tensor = init_shmem(self.shm_fn, self.shm_size, self.dtype)
        self.shared_tensor = self.shared_tensor.view(*self.shape)
        self._register()

    def _register(self) -> None:
        # current implementation is blocking
        channel = grpc.insecure_channel(self.channel)
        stub = comm_sidecar_pb2_grpc.CommSidecarStub(channel)
        if self.mode == SidecarMode.SEND:
            mode = comm_sidecar_pb2.Mode.SEND
        elif self.mode == SidecarMode.RECV:
            mode = comm_sidecar_pb2.Mode.RECV
        else:
            raise ValueError("Invalid mode")
        request = comm_sidecar_pb2.RegisterRequest(
            mode=mode,
            shape=self.shape,
            dtype=str(self.dtype).split(".")[-1],
        )
        response = stub.Register(request)
        if response.status == common_pb2.Status.STATUS_OK:
            logger.info("Sidecar registered successfully")
        else:
            logger.error("Failed to register sidecar")
            exit(1)

    async def send(
        self,
        tensor: torch.Tensor,
        req_ids: List[int],
        dest_ranks: List[int],  # global ranks
        chunk_ids: List[int],
    ) -> None:
        # sanity checks
        if self.mode != SidecarMode.SEND:
            raise ValueError("Cannot send data in receive mode")
        if tensor.device != self.device:
            raise ValueError("Tensor device does not match sidecar device")
        if tensor.shape != self.shape:
            raise ValueError("Tensor shape does not match sidecar shape")
        if tensor.dtype != self.dtype:
            raise ValueError("Tensor dtype does not match sidecar dtype")

        cuda_event = torch.cuda.Event(interprocess=True)
        with torch.cuda.stream(self.stream):
            self.shared_tensor.copy_(tensor, non_blocking=True)
            cuda_event.record(self.stream)

        ipc_handle = cuda_event.ipc_handle()
        async with grpc.aio.insecure_channel(self.channel) as channel:
            stub = comm_sidecar_pb2_grpc.CommSidecarStub(channel)
            request = comm_sidecar_pb2.SendRequest(
                request_ids=req_ids,
                dest_ranks=dest_ranks,
                chunk_ids=chunk_ids,
                ipc_handle=pickle.dumps(ipc_handle),
            )
            response = await stub.Send(request)
            if response.status == common_pb2.Status.STATUS_OK:
                logger.info("Data sent successfully")
            else:
                logger.error("Failed to send data")

    async def recv(self, req_ids: List[int]) -> torch.Tensor:
        """currently no future
        Upon returning, the shared tensor contains the received data
        """
        if self.mode != SidecarMode.RECV:
            raise ValueError("Cannot receive data in send mode")
        # TODO: should check from a (global) queue
        async with grpc.aio.insecure_channel(self.channel) as channel:
            stub = comm_sidecar_pb2_grpc.CommSidecarStub(channel)
            request = comm_sidecar_pb2.ReceiveRequest(
                request_ids=req_ids,
            )
            response = await stub.Receive(request)
            if response.status == common_pb2.Status.STATUS_OK:
                logger.info(f"Request {req_ids} received successfully")
            else:
                logger.error("Failed to receive data")
        return self.shared_tensor
