from __future__ import annotations
from abc import ABC
import asyncio
from dataclasses import dataclass
from functools import reduce
from operator import mul
import os
import pickle
from typing import Dict, List, Tuple

import grpc
import kubernetes_asyncio.client as kclient
import kubernetes_asyncio.config as kconfig
import torch
import torch.distributed as dist
import tyro

from cornserve.logging import get_logger
from cornserve.services.pb import comm_sidecar_pb2, comm_sidecar_pb2_grpc, common_pb2

from .api import (
    SHM_SIZE,
    TensorLayout,
    device_from_rank,
    init_shmem,
    shm_fn_from_rank,
    grpc_channel_from_rank,
)

logger = get_logger(__name__)
cleanup_coroutines = []


def recv(tensor: torch.Tensor, src: int | None = None) -> None:
    req = dist.irecv(tensor=tensor, src=src)
    if req is not None:
        req.wait()
    else:
        print("No message")
        exit(0)


async def recv_async(
    chunk: torch.Tensor,
    src: int,
) -> None:
    return await asyncio.to_thread(recv, chunk, src)


def send(tensor: torch.Tensor, rank: int) -> None:
    req = dist.isend(tensor, dst=rank)
    if req is not None:
        req.wait()
    else:
        logger.error("Failed to send tensor to dest rank %d", rank)
        exit(0)


async def send_async(tensor: torch.Tensor, rank: int) -> None:
    return await asyncio.to_thread(send, tensor, rank)


class Sidecar(ABC):
    """Sidecar abstract class."""

    def __init__(
        self,
        gpu_rank: int,
        sidecar_rank: int,
        dtype: torch.dtype,
    ) -> None:
        self.gpu_rank = gpu_rank
        self.sidecar_rank = sidecar_rank
        self.dtype = dtype


class CommSidercarReceiver(Sidecar):
    """Sidercar receiver gRPC service backend."""

    @dataclass
    class RequestInfo:
        id: int
        slot: int
        num_shards: int
        shard_size: int
        num_chunks: int  # num_chunks_per_shard
        chunk_size: int
        chunk_availablity: List[int]
        done: bool = False

        def __post_init__(self):
            assert len(self.chunk_availablity) == self.num_chunks * self.num_shards

        def check_consistency(self, num_shards: int, num_chunks: int) -> bool:
            return self.num_shards == num_shards and self.num_chunks == num_chunks

    def __init__(
        self,
        gpu_rank: int,
        sidecar_rank: int,
        shape: Tuple[int, ...],
        dtype: str,
    ) -> None:
        super().__init__(gpu_rank, sidecar_rank, getattr(torch, dtype))

        self.shape = shape
        self.shm_fn = shm_fn_from_rank(self.gpu_rank)
        self.device = device_from_rank(self.gpu_rank)
        self.shared_tensor = init_shmem(self.shm_fn, SHM_SIZE, self.dtype)
        self.lock = asyncio.Lock()
        self._post_init()

    def _post_init(self) -> None:
        # (receiver) shape = (sender) chunk_shape * num_chunks * num_shards
        self.tensor_size = reduce(mul, self.shape)
        self.num_slots = SHM_SIZE // self.tensor_size
        self.occupancy = [0 for _ in range(self.num_slots)]

        # a legder to keep if all chunks are received of one request id
        self.rid_mapping: Dict[int, CommSidercarReceiver.RequestInfo] = {}

        # per req event, cannot be put in the RequestInfo because
        # only recieve will wait on this event, recv_task will try to set this event
        self.req_events: Dict[int, asyncio.Event] = {}

    async def find_slot(self) -> int:
        async with self.lock:
            for i, occ in enumerate(self.occupancy):
                if occ == 0:
                    self.occupancy[i] = 1
                    return i
            # possibly make this wait
            return -1

    async def release_slot(self, slot: int) -> None:
        async with self.lock:
            self.occupancy[slot] = 0

    async def recv_task(
        self,
        chunk: torch.Tensor,
        src: int,
        req_info: CommSidercarReceiver.RequestInfo,
        global_chunk_id: int,
    ) -> None:
        logger.info("Tring to receive chunk %d from src %d", global_chunk_id, src)
        await recv_async(chunk, src)
        logger.info("Received chunk %d from src %d", global_chunk_id, src)
        req_info.chunk_availablity[global_chunk_id] = 1
        if all(req_info.chunk_availablity):
            req_info.done = True
            if req_info.id in self.req_events:
                self.req_events[req_info.id].set()

    async def prepare_receive(
        self,
        request: comm_sidecar_pb2.PrepareReceiveRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.PrepareReceiveResponse:
        if self.shared_tensor is None:
            logger.error("Shared tensor not initialized")
            return comm_sidecar_pb2.PrepareReceiveResponse(
                status=common_pb2.Status.STATUS_ERROR
            )
        # every chunk must be flattened
        # the contract is to first slice by chunk then by shard
        # currently does not allow unbalanced chunk size
        logger.info(
            "Prepare receive request id %d, chunk %d, num_chunks %d, shard_rank %d, num_shards %d from src_sidecar_rank %d",
            request.request_id,
            request.chunk_id,
            request.num_chunks,
            request.shard_rank,
            request.num_shards,
            request.src_sidecar_rank,
        )
        dtype = getattr(torch, request.dtype)
        assert self.dtype == dtype, "Data type mismatch"
        async with self.lock:
            logger.info("currend slot occupancy: " + str(self.occupancy))

        if request.request_id not in self.rid_mapping:
            # first chunk
            assert (
                self.tensor_size % request.num_shards == 0
            ), f"Tensor size ({self.tensor_size}) must be divisible by shard size ({request.num_shards})."
            shard_size = int(self.tensor_size / request.num_shards)

            assert (
                shard_size % request.num_chunks == 0
            ), f"Shard tensor size ({shard_size}) must be divisible by num chunks ({request.num_chunks})."
            chunk_size = int(shard_size / request.num_chunks)
            slot = await self.find_slot()

            req_info = CommSidercarReceiver.RequestInfo(
                id=request.request_id,
                slot=slot,
                num_shards=request.num_shards,
                shard_size=shard_size,
                num_chunks=request.num_chunks,
                chunk_size=chunk_size,
                chunk_availablity=[
                    0 for _ in range(request.num_chunks * request.num_shards)
                ],
            )
            self.rid_mapping[request.request_id] = req_info
        else:
            req_info = self.rid_mapping[request.request_id]
            req_info.check_consistency(request.num_shards, request.num_chunks)

        global_chunk_id = req_info.num_chunks * request.shard_rank + request.chunk_id

        offset = global_chunk_id * req_info.chunk_size
        # Create the task but return immediately
        logger.info("Queuing recv task for chunk %d", global_chunk_id)
        asyncio.create_task(
            self.recv_task(
                self.shared_tensor[
                    req_info.slot * self.tensor_size
                    + offset : req_info.slot * self.tensor_size
                    + offset
                    + req_info.chunk_size
                ],
                request.src_sidecar_rank,
                req_info,
                global_chunk_id,
            )
        )

        return comm_sidecar_pb2.PrepareReceiveResponse(
            status=common_pb2.Status.STATUS_OK,
        )

    async def receive(
        self,
        request: comm_sidecar_pb2.ReceiveRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.ReceiveResponse:

        if (
            request.request_id in self.rid_mapping
            and self.rid_mapping[request.request_id].done
        ):
            # receiver calling receive after all chunks are received
            logger.info("All chunks received for request id %d", request.request_id)
            return comm_sidecar_pb2.ReceiveResponse(
                slot=self.rid_mapping[request.request_id].slot
            )

        # receiver calling receive before any sender sending all chunks
        event = asyncio.Event()
        self.req_events[request.request_id] = event
        logger.info(
            "Waiting for all chunks to be received for request id %d",
            request.request_id,
        )
        await event.wait()
        return comm_sidecar_pb2.ReceiveResponse(
            slot=self.rid_mapping[request.request_id].slot
        )

    async def mark_done(
        self,
        request: comm_sidecar_pb2.MarkDoneRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.MarkDoneResponse:
        if request.request_id not in self.rid_mapping:
            return comm_sidecar_pb2.MarkDoneResponse(
                status=common_pb2.Status.STATUS_ERROR
            )
        slot = self.rid_mapping[request.request_id].slot
        await self.release_slot(slot)
        del self.rid_mapping[request.request_id]
        if request.request_id in self.req_events:
            del self.req_events[request.request_id]
        return comm_sidecar_pb2.MarkDoneResponse(status=common_pb2.Status.STATUS_OK)


class CommSidercarSender(Sidecar):
    """Sidercar sender gRPC service backend."""

    def __init__(
        self,
        gpu_rank: int,
        sidecar_rank: int,
        chunk_shape: Tuple[int, ...],
        dtype: str,
        shard_rank: int = 0,
        num_shards: int = 1,
        layout: TensorLayout = TensorLayout.FULL,
    ) -> None:
        super().__init__(gpu_rank, sidecar_rank, getattr(torch, dtype))

        self.chunk_shape = chunk_shape
        self.chunk_size = reduce(mul, chunk_shape)
        self.shard_rank = shard_rank
        self.num_shards = num_shards
        self.layout = layout

        self.shm_fn = shm_fn_from_rank(self.gpu_rank)
        self.device = device_from_rank(self.gpu_rank)
        self.shared_tensor = init_shmem(self.shm_fn, SHM_SIZE, self.dtype)

    async def send(
        self,
        request: comm_sidecar_pb2.SendRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.SendResponse:
        # sanity check
        if (
            request.chunk_slot < 0
            or request.chunk_slot * self.chunk_size >= SHM_SIZE
            or request.dst_sidecar_rank < 0
            or request.dst_sidecar_rank == self.sidecar_rank
        ):
            # XXX: gRPC error propagation should be done with `context.abort`.
            logger.error("Invalid send request")
            return comm_sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_ERROR)

        # inform destination sidecar
        dst_channel = grpc_channel_from_rank(request.dst_sidecar_rank)
        # XXX: Creating a channel is an expensive overation in gRPC. Should be lazily
        #      created and reused. Stubs are cheaper to create, but still can be reused.
        async with grpc.aio.insecure_channel(dst_channel) as channel:
            stub = comm_sidecar_pb2_grpc.CommSidecarStub(channel)
            prepare_receive_request = comm_sidecar_pb2.PrepareReceiveRequest(
                request_id=request.request_id,
                chunk_id=request.chunk_id,
                num_chunks=request.num_chunks,
                dtype=str(self.dtype).split(".")[-1],
                shard_rank=self.shard_rank,
                num_shards=self.num_shards,
                src_sidecar_rank=self.sidecar_rank,
                layout=self.layout.value,
            )
            response = await stub.PrepareReceive(prepare_receive_request)
            if response.status != common_pb2.Status.STATUS_OK:
                logger.error("Failed to prepare receive")
                return comm_sidecar_pb2.SendResponse(
                    status=common_pb2.Status.STATUS_ERROR
                )

        ipc_handle = pickle.loads(request.ipc_handle)
        cuda_event = torch.cuda.Event.from_ipc_handle(self.device, ipc_handle)

        # TODO: does this need to be wrapped in another thread?
        # XXX: `synchronize` is likely to block the whole server. Should be
        #      wrapped in a task that polls `query`. Or something like:
        #      while not cuda_event.query():
        #          await asyncio.sleep(0.0)  # Allows other futures to make process.
        cuda_event.synchronize()

        # here is it making extra copy?
        await send_async(
            self.shared_tensor[
                request.chunk_slot
                * self.chunk_size : (request.chunk_slot + 1)
                * self.chunk_size
            ],
            request.dst_sidecar_rank,
        )
        return comm_sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_OK)


class CommSidecarServicer(comm_sidecar_pb2_grpc.CommSidecarServicer):
    """Comm Sidecar gRPC service implementation.
    A union wrapper for both sender and receiver sidecar services.
    """

    def __init__(self, gpu_rank: int, sidecar_rank: int) -> None:
        self.gpu_rank = gpu_rank
        self.sidecar_rank = sidecar_rank
        self.sidecar: Sidecar | None = None

    async def RegisterSender(
        self,
        request: comm_sidecar_pb2.RegisterSenderRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.RegisterResponse:
        logger.info(
            "registering sender, metadata: %s", str(context.invocation_metadata())
        )

        if self.sidecar is not None:
            logger.error("Sidecar already registered")
            return comm_sidecar_pb2.RegisterResponse(
                gpu_rank=-1,
            )

        self.sidecar = CommSidercarSender(
            gpu_rank=self.gpu_rank,
            sidecar_rank=self.sidecar_rank,
            chunk_shape=tuple(request.chunk_shape),
            dtype=request.dtype,
            shard_rank=request.shard_rank,
            num_shards=request.num_shards,
            layout=TensorLayout(request.layout),
        )

        logger.info(
            f"Registered sender of gpu_rank {self.gpu_rank}, sidecar_rank {self.sidecar_rank}"
        )

        return comm_sidecar_pb2.RegisterResponse(
            gpu_rank=self.gpu_rank,
        )

    async def RegisterReceiver(
        self,
        request: comm_sidecar_pb2.RegisterReceiverRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.RegisterResponse:
        logger.info("registering receiver")

        if self.sidecar is not None:
            logger.error("Sidecar already registered")
            return comm_sidecar_pb2.RegisterResponse(
                gpu_rank=-1,
            )
        self.sidecar = CommSidercarReceiver(
            gpu_rank=self.gpu_rank,
            sidecar_rank=self.sidecar_rank,
            shape=tuple(request.shape),
            dtype=request.dtype,
        )

        return comm_sidecar_pb2.RegisterResponse(
            gpu_rank=self.gpu_rank,
        )

    async def Send(
        self, request: comm_sidecar_pb2.SendRequest, context: grpc.aio.ServicerContext
    ) -> comm_sidecar_pb2.SendResponse:
        if self.sidecar is None:
            logger.error("Sidecar not registered")
            return comm_sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_ERROR)
        if not isinstance(self.sidecar, CommSidercarSender):
            logger.error("Invalid sidecar mode")
            return comm_sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_ERROR)

        return await self.sidecar.send(request, context)

    async def PrepareReceive(
        self,
        request: comm_sidecar_pb2.PrepareReceiveRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.PrepareReceiveResponse:
        if self.sidecar is None:
            logger.error("Sidecar not registered")
            return comm_sidecar_pb2.PrepareReceiveResponse(
                status=common_pb2.Status.STATUS_ERROR
            )
        if not isinstance(self.sidecar, CommSidercarReceiver):
            logger.error("Invalid sidecar mode")
            return comm_sidecar_pb2.PrepareReceiveResponse(
                status=common_pb2.Status.STATUS_ERROR
            )
        return await self.sidecar.prepare_receive(request, context)

    async def Receive(
        self,
        request: comm_sidecar_pb2.ReceiveRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.ReceiveResponse:
        """Initiate receiving a tensor from some other rank."""
        if self.sidecar is None:
            logger.error("Sidecar not registered")
            return comm_sidecar_pb2.ReceiveResponse(slot=-1)
        if not isinstance(self.sidecar, CommSidercarReceiver):
            logger.error("Invalid sidecar mode")
            return comm_sidecar_pb2.ReceiveResponse(slot=-1)

        return await self.sidecar.receive(request, context)

    async def MarkDone(
        self,
        request: comm_sidecar_pb2.MarkDoneRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.MarkDoneResponse:
        if self.sidecar is None:
            logger.error("Sidecar not registered")
            return comm_sidecar_pb2.MarkDoneResponse(
                status=common_pb2.Status.STATUS_ERROR
            )
        if not isinstance(self.sidecar, CommSidercarReceiver):
            logger.error("Invalid sidecar mode")
            return comm_sidecar_pb2.MarkDoneResponse(
                status=common_pb2.Status.STATUS_ERROR
            )

        return await self.sidecar.mark_done(request, context)

    async def Unregister(
        self,
        request: comm_sidecar_pb2.UnregisterRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.UnregisterResponse:
        del self.sidecar
        self.sidecar = None
        logger.info("Unregistered sidecar")
        return comm_sidecar_pb2.UnregisterResponse(status=common_pb2.Status.STATUS_OK)


NAMESPACE = "cornserve"


async def get_local_rank(pod_name: str) -> int:
    # TODO: test
    kconfig.load_incluster_config()

    async with kclient.ApiClient() as api_client:
        v1 = kclient.CoreV1Api(api_client)

        pod = await v1.read_namespaced_pod(name=pod_name, namespace=NAMESPACE)
        node_name = pod.spec.node_name
        label_selector = f"app=sidecar"
        pods = await v1.list_namespaced_pod(
            namespace=NAMESPACE, label_selector=label_selector
        )
        same_node_pods = [p for p in pods.items if p.spec.node_name == node_name]
        sorted_pods = sorted(same_node_pods, key=lambda p: p.metadata.name)

        local_rank = None
        logger.info(
            "Pods on the same node:" + str([p.metadata.name for p in sorted_pods])
        )
        for index, p in enumerate(sorted_pods):
            if p.metadata.name == pod_name:
                local_rank = index
                break

        if local_rank is None:
            logger.error(
                "Current pod not found in the list of sidecar pods on the node."
            )
            return -1

        logger.info("Local rank: %d", local_rank)
        return local_rank


async def main(
    ip: str = "[::]",
    base_port: int = 10000,
) -> None:
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    pod_name = os.environ.get("POD_NAME")
    if pod_name:
        try:
            gpu_rank = await get_local_rank(pod_name)
            sidecar_rank = int(pod_name.split("-")[-1])
        except ValueError:
            gpu_rank = -1
            sidecar_rank = -1
    else:
        gpu_rank = int(os.environ.get("RANK", -1))
        sidecar_rank = gpu_rank

    assert gpu_rank >= 0, "Invalid rank"
    assert sidecar_rank >= 0, "Invalid global rank"

    init_url = f"tcp://{master_addr}:{master_port}"
    dist.init_process_group(
        backend="gloo", init_method=init_url, rank=sidecar_rank, world_size=world_size
    )
    logger.info(
        f"Sidecar {sidecar_rank} out of {world_size}, using local rank {gpu_rank}"
    )

    server = grpc.aio.server()
    comm_sidecar_pb2_grpc.add_CommSidecarServicer_to_server(
        CommSidecarServicer(gpu_rank=gpu_rank, sidecar_rank=sidecar_rank),
        server,
    )
    port = base_port + sidecar_rank
    listen_addr = f"{ip}:{port}"
    server.add_insecure_port(listen_addr)
    logger.info(f"Sidecar server started on {listen_addr}")
    await server.start()

    async def server_graceful_shutdown():
        logger.info("Starting graceful shutdown...")
        await server.stop(5)
        logger.info("Server stopped")

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
