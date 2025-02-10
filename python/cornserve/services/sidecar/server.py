import os

import asyncio

import grpc
import tyro
import torch
import torch.distributed as dist
import pickle

from cornserve.services.pb import (
    comm_sidecar_pb2,
    comm_sidecar_pb2_grpc,
    common_pb2,
)
from .api import (
    shm_fn_from_rank,
    device_from_rank,
    init_shmem,
    SidecarMode,
)
from cornserve.logging import get_logger

import kubernetes_asyncio.config as kconfig
import kubernetes_asyncio.client as kclient

logger = get_logger(__name__)
cleanup_coroutines = []


def recv(tensor: torch.Tensor) -> None:
    req = dist.irecv(tensor=tensor)
    if req is not None:
        req.wait()
    else:
        logger.error("No message")
        exit(0)


async def recv_async(tensor: torch.Tensor) -> None:
    return await asyncio.to_thread(recv, tensor)


def send(tensor: torch.Tensor, rank: int) -> None:
    req = dist.isend(tensor, dst=rank)
    if req is not None:
        req.wait()
    else:
        logger.error("Failed to send tensor to dest rank %d", rank)
        exit(0)


async def send_async(tensor: torch.Tensor, rank: int) -> None:
    return await asyncio.to_thread(send, tensor, rank)


class CommSidecarServicer(comm_sidecar_pb2_grpc.CommSidecarServicer):
    """Comm Sidecar gRPC service implementation."""

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.shm_fn = shm_fn_from_rank(rank)
        self.device = device_from_rank(rank)
        logger.info(f"Sidecar started on device {self.device} for rank {rank}")

        self.shared_tensor: torch.Tensor | None = None
        self.dtype: torch.dtype | None = None
        self.size: int = 0
        self.mode: SidecarMode | None = None

    async def Register(
        self,
        request: comm_sidecar_pb2.RegisterRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.RegisterResponse:
        if request.mode == comm_sidecar_pb2.Mode.SEND:
            self.mode = SidecarMode.SEND
        elif request.mode == comm_sidecar_pb2.Mode.RECV:
            self.mode = SidecarMode.RECV
        self.shape = request.shape
        from functools import reduce
        from operator import mul

        self.size = reduce(mul, self.shape)

        self.dtype = getattr(torch, request.dtype)
        if self.mode is None or self.size <= 0 or self.dtype is None:
            logger.error("Invalid register request")
            return comm_sidecar_pb2.RegisterResponse(
                status=common_pb2.Status.STATUS_ERROR
            )
        logger.info(
            "Registered sidecar with mode %s, size %d, dtype %s",
            self.mode,
            self.size,
            self.dtype,
        )
        self.shared_tensor = init_shmem(self.shm_fn, self.size, self.dtype)
        return comm_sidecar_pb2.RegisterResponse(status=common_pb2.Status.STATUS_OK)

    async def Send(
        self,
        request: comm_sidecar_pb2.SendRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.SendResponse:
        """Initiate sending a tensor from current rank to another."""
        if (
            self.shared_tensor is None
            or self.mode != SidecarMode.SEND
            or self.size <= 0
            or self.dtype is None
        ):
            logger.error("Invalid send request")
            return comm_sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_ERROR)

        logger.info("received send request", extra={"request": request})

        ipc_handle = pickle.loads(request.ipc_handle)
        cuda_event = torch.cuda.Event.from_ipc_handle(self.device, ipc_handle)
        cuda_event.synchronize()
        logger.info("sending tensor: %s", torch.prod(self.shared_tensor))

        for rank in request.dest_ranks:
            await send_async(self.shared_tensor, rank)

        return comm_sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_OK)

    async def Receive(
        self,
        request: comm_sidecar_pb2.ReceiveRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.ReceiveResponse:
        """Initiate receiving a tensor from some other rank."""
        if (
            self.shared_tensor is None
            or self.mode != SidecarMode.RECV
            or self.size <= 0
            or self.dtype is None
        ):
            logger.error("Invalid send request")
            return comm_sidecar_pb2.ReceiveResponse(
                status=common_pb2.Status.STATUS_ERROR
            )
        # TODO: bind request id
        logger.info("trying to receive", extra={"request": request})
        await recv_async(self.shared_tensor)
        logger.info("sidecar received tensor: %s", torch.prod(self.shared_tensor))
        return comm_sidecar_pb2.ReceiveResponse(status=common_pb2.Status.STATUS_OK)


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
            rank = await get_local_rank(pod_name)
        except ValueError:
            rank = -1
    else:
        rank = int(os.environ.get("RANK", -1))
    if rank == -1:
        raise ValueError(
            "RANK environment variable is not set and POD_NAME is not available."
        )

    logger.info(f"Connecting to master at {master_addr}:{master_port} from rank {rank}")
    init_url = f"tcp://{master_addr}:{master_port}"
    dist.init_process_group(
        backend="gloo", init_method=init_url, rank=rank, world_size=world_size
    )
    logger.info(f"Initialized process group with rank {rank} out of {world_size}")

    server = grpc.aio.server()
    comm_sidecar_pb2_grpc.add_CommSidecarServicer_to_server(
        CommSidecarServicer(rank=rank),
        server,
    )
    port = base_port + rank
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

# Next todo:
# 1. bind request id, maybe first extend the tensor with a header, and udpate recv side to read the header
# 2. add chunk ids and total number of chunks
# 3. ring buffer, where api.py should handle the id slot
