"""Sidecar server implementniation.

The sidecar server is a gRPC service that runs on each node in the cluster. This service
is the backend for the `SidecarSender` and `SidecarReceiver` classes in the `api` module.
It has two corresponding components, `CommSidecarSender` and `CommSidecarReceiver`, which
together provide the functionality to send and receive tensors between ranks.
"""

from __future__ import annotations
import asyncio
from dataclasses import dataclass
import os
import pickle
from typing import Dict

import grpc
import kubernetes_asyncio.client as kclient
import kubernetes_asyncio.config as kconfig
import torch
import torch.distributed as dist
import tyro

from cornserve.logging import SidcarAdapter, get_logger
from cornserve.services.pb import comm_sidecar_pb2, comm_sidecar_pb2_grpc, common_pb2
from cornserve.services.utils import get_dtype_size

from .shm_manager import SharedMemoryBuffer, SharedMemoryManager
from .utils import (
    TensorLayout,
    chunk_tag,
    device_from_rank,
    grpc_channel_from_rank,
    init_shmem,
    shm_fn_from_rank,
)

logger = get_logger(__name__)
cleanup_coroutines = []


class CommSidecarReceiver:
    """
    The receiver sidecar server supports receiving tensors from other ranks using gloo backend.
    """
    @dataclass
    class TransferRequestState:
        """
        Internal data structure to keep track of a tansfer request's state

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
        gpu_rank: int,
        sidecar_rank: int,
        shm_size: int,
        slot_size: int,  # maybe slot size directly
        dtype: str,
    ) -> None:
        """
        Initialize the receiver sidecar.

        Args:
            gpu_rank: The local GPU rank.
            sidecar_rank: The sidecar rank, aka global rank.
            shm_size: The shared memory size (number of elements of given dtype).
            slot_size: The shape of the tensor to be received, currently fixed.
            dtype: The data type of the receiving tensor.
        """
        self.gpu_rank = gpu_rank
        self.sidecar_rank = sidecar_rank

        self.dtype = getattr(torch, dtype)
        self.device = device_from_rank(self.gpu_rank)
        self.shm_fn = shm_fn_from_rank(self.gpu_rank)
        self.shared_tensor = init_shmem(self.shm_fn, shm_size, self.dtype)
        logger.info("Using shm_fn %s with size %d of dtype %s", self.shm_fn, shm_size, self.dtype)
        self.shm_manager = SharedMemoryManager(
            shm=self.shared_tensor,
            shm_size=shm_size,
            slot_size=slot_size,
        )
        self.has_memory = asyncio.Condition()
        self._post_init()

    def _post_init(self) -> None:
        """Post init checks and setup."""
        # a legder to keep the transfer status of each transfer request
        self.ledger: Dict[str, CommSidecarReceiver.TransferRequestState] = {}
        # per req event, recieve will wait on this event, recv_task will try to set this event
        self.req_events: Dict[str, asyncio.Event] = {}

    async def shutdown(self):
        """Cleanup routines for the receiver."""
        # remove the shared memory file
        del self.shared_tensor
        del self.shm_manager
        os.unlink(self.shm_fn)

    async def prepare_receive(
        self,
        request: comm_sidecar_pb2.PrepareReceiveRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.PrepareReceiveResponse:
        """Prepare to receive a tensor from another rank, called by the sender sidecar server.
        This function allocates a shared memory buffer if not already allocated,
        and queues up a receive task to receive the tensor.
        """
        logger.info(
            ("Prepare receive for request id %s, shard_size %d, dtype %s, src_rank %d, shard_rank %d, "
             "num_shards %d, chunk_size %d, num_chunks %d, chunk_id %d, shard_offset %d"),
            request.id, request.shard_size, request.dtype, request.src_rank, request.shard_rank,
            request.num_shards, request.chunk_size, request.num_chunks, request.chunk_id, request.shard_offset,
        )
        dtype = getattr(torch, request.dtype)
        if self.dtype != dtype:
            logger.error("Data type mismatch")
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Data type mismatch")
        async with self.has_memory:
            # use while loop to avoid concurrent writes to the bookkeeping data
            while request.id not in self.ledger:
                logger.info("Allocating buffer for request id %s", request.id)
                buffer = self.shm_manager.allocate(request.chunk_size * request.num_chunks)
                if buffer is None:
                    logger.warning(
                        "Shared memory pressure, try increasing shm_size. Free slots %d/%d, need %d",
                        self.shm_manager.free_slots(),
                        self.shm_manager.num_slots,
                        request.chunk_size * request.num_chunks // self.shm_manager.slot_size,
                    )
                    await self.has_memory.wait()
                else:
                    buffer.create_chunks(request.num_chunks, request.num_shards)
                    state = CommSidecarReceiver.TransferRequestState(request.id, buffer)
                    self.ledger[request.id] = state
                    logger.info("buffer status %s", buffer)
            state = self.ledger[request.id]

        chunk = state.buffer.chunks[request.chunk_id]
        tag = chunk_tag(request.src_rank, request.chunk_id, request.shard_rank)

        # TODO: allow batch recv
        def recv_task():
            """The task to receive the tensor."""
            logger.info("Queuing recv task for chunk %d of request %s tag %d", request.chunk_id, request.id, tag)
            dist.recv(
                chunk.data[request.shard_offset : request.shard_offset + request.shard_size],
                src=request.src_rank,
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
            logger.info("chunk.ready %s buffer.is_ready() %s", chunk.ready, state.buffer.is_ready())
            logger.info("chunks status %s", state.buffer.chunks[request.chunk_id])
            logger.info("buffer status %s", state.buffer)
            if state.buffer.is_ready():
                state.done = True
                if request.id in self.req_events:
                    self.req_events[request.id].set()

        asyncio.create_task(asyncio.to_thread(recv_task))
        logger.info("current free slots: %d/%d", self.shm_manager.free_slots(), self.shm_manager.num_slots)
        return comm_sidecar_pb2.PrepareReceiveResponse(
            status=common_pb2.Status.STATUS_OK,
        )

    async def receive(
        self,
        recv_req: comm_sidecar_pb2.ReceiveRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.ReceiveResponse:
        """Receive the tensor of a request from other ranks, returns a slot number in the shared memory.

        If all chunks are received, return the slot number imediately.
        Else, queues up an event for the request id and waits for all chunks to be received.
        """
        logger.info("==> Receive request for request id %s", recv_req.id)
        if recv_req.id in self.ledger and self.ledger[recv_req.id].done:
            logger.info("All chunks received for request id %s", recv_req.id)
        else:
            # still waiting for chunks/shards
            event = asyncio.Event()
            self.req_events[recv_req.id] = event
            await event.wait()
            # await asyncio.sleep(3.0)
            logger.info(
                "Received event for all chunks in request id %s",
                recv_req.id,
            )

        offset = self.ledger[recv_req.id].buffer.slots[0] * self.shm_manager.slot_size
        size = self.ledger[recv_req.id].buffer.size
        return comm_sidecar_pb2.ReceiveResponse(offset=offset, size=size)

    async def mark_done(
        self,
        mark_done_req: comm_sidecar_pb2.MarkDoneRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.MarkDoneResponse:
        """Mark a tensor as consumed, free up the shared memory used."""
        if mark_done_req.id not in self.ledger:
            await context.abort(grpc.StatusCode.NOT_FOUND, "mark_done_req not found")
        logger.info("Freeing up %d slots from %s", len(self.ledger[mark_done_req.id].buffer.slots), mark_done_req.id)
        logger.info("free slots before %d/%d", self.shm_manager.free_slots(), self.shm_manager.num_slots)
        async with self.has_memory:
            self.shm_manager.free(self.ledger[mark_done_req.id].buffer)
            self.has_memory.notify_all()
        logger.info("free slots after %d/%d", self.shm_manager.free_slots(), self.shm_manager.num_slots)
        del self.ledger[mark_done_req.id]
        if mark_done_req.id in self.req_events:
            del self.req_events[mark_done_req.id]
        return comm_sidecar_pb2.MarkDoneResponse(status=common_pb2.Status.STATUS_OK)


class CommSidecarSender:
    """Sidecar sender gRPC service backend.

    Implements the gRPC service for the sender sidecar.
    """

    def __init__(  # noqa: PLR0913
        self,
        gpu_rank: int,
        sidecar_rank: int,
        shm_size: int,
        slot_size: int,
        dtype: torch.dtype,
        shard_rank: int = 0,
        num_shards: int = 1,
        layout: TensorLayout = TensorLayout.FULL,
    ) -> None:
        """Initialize the sender sidecar server.
        Args:
            gpu_rank: The local GPU rank.
            sidecar_rank: The sidecar rank, aka global rank.
            shm_size: The shared memory size (number of elements of given dtype).
            slot_size: The slot_size of the shared memory buffer.
            dtype: The data type of the sending tensor.
            shard_rank: The rank of the shard, default to 0.
            num_shards: The number of shards, default to 1.
            layout: The layout of the tensor, default to FULL.
        """
        self.gpu_rank = gpu_rank
        self.sidecar_rank = sidecar_rank
        self.shm_size = shm_size
        self.dtype = dtype

        self.slot_size = slot_size
        self.shard_rank = shard_rank
        self.num_shards = num_shards
        self.layout = layout

        self.shm_fn = shm_fn_from_rank(self.gpu_rank)
        self.device = device_from_rank(self.gpu_rank)
        logger.info("Using shm_fn %s with size %d of dtype %s", self.shm_fn, shm_size, self.dtype)
        self.shared_tensor = init_shmem(self.shm_fn, self.shm_size, self.dtype)

        self.dst_channels: Dict[int, grpc.aio.Channel] = {}
        self.dst_stubs: Dict[int, comm_sidecar_pb2_grpc.CommSidecarStub] = {}

    async def shutdown(self):
        """Cleanup routines for the sender sidecar."""
        for channel in self.dst_channels.values():
            await channel.close()
        # remove the shared memory file
        del self.shared_tensor
        os.unlink(self.shm_fn)


    async def send(
        self,
        send_request: comm_sidecar_pb2.SendRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.SendResponse:
        """Send a tensor to another rank.

        First use prepare_receive to send control signals to the destination sidecar,
        then queue up the send tasks.
        """
        # sanity check
        if send_request.slot < 0 or send_request.slot * self.slot_size + send_request.size > self.shm_size:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                f"slot out of range {send_request.slot}*{self.slot_size}+{send_request.size} {self.shm_size}",
            )

        # TODO(Jeff): only send to a head receiver when TP is enabled
        for r in send_request.dst_ranks:
            if r < 0 or r == self.sidecar_rank:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "invalid destination rank")

            # inform destination sidecar
            # lazily create channel
            if r not in self.dst_channels:
                self.dst_channels[r] = grpc.aio.insecure_channel(grpc_channel_from_rank(r))
                self.dst_stubs[r] = comm_sidecar_pb2_grpc.CommSidecarStub(self.dst_channels[r])
                logger.info("Connected to sidecar-%d", r)

        for r in send_request.dst_ranks:
            # use a second loop here to make send atomic
            stub = self.dst_stubs[r]
            logger.info(
                "Calling prepare receive on sidecar-%d for request %s chunk id %s out of %d",
                r,
                send_request.id,
                send_request.chunk_id,
                send_request.num_chunks,
            )
            prepare_receive_request = comm_sidecar_pb2.PrepareReceiveRequest(
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
                return comm_sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_ERROR)

        ipc_handle = pickle.loads(send_request.ipc_handle)
        cuda_event = torch.cuda.Event.from_ipc_handle(self.device, ipc_handle)

        while not cuda_event.query():
            await asyncio.sleep(0.0)  # Allows other futures to make process.

        logger.info("Sending chunk %d for req %s", send_request.chunk_id, send_request.id)
        ops = []
        for rank in send_request.dst_ranks:
            tag = chunk_tag(self.sidecar_rank, send_request.chunk_id, self.shard_rank)
            ops.append(
                dist.P2POp(
                    dist.isend,
                    self.shared_tensor[
                        send_request.slot * self.slot_size : send_request.slot * self.slot_size + send_request.size
                    ],
                    peer=rank,
                    tag=tag,
                )
            )
            logger.info("Queuing send task for chunk %d to sidecar-%d with tag %d", send_request.chunk_id, rank, tag)
        reqs = dist.batch_isend_irecv(ops)

        def wait_fn():
            for req in reqs:
                req.wait()

        await asyncio.to_thread(wait_fn)
        logger.info(
            "SHARD RANK %d: sent chunk %d for request %s", self.shard_rank, send_request.chunk_id, send_request.id
        )

        return comm_sidecar_pb2.SendResponse(status=common_pb2.Status.STATUS_OK)


class CommSidecarServicer(comm_sidecar_pb2_grpc.CommSidecarServicer):
    """A unified wrapper for both sender and receiver sidecar servers. Entry point for the gRPC service."""

    def __init__(self, gpu_rank: int, sidecar_rank: int, shm_size: int) -> None:
        """Initialize the sidecar service."""
        self.gpu_rank = gpu_rank
        self.sidecar_rank = sidecar_rank
        self.shm_size = shm_size
        self.sidecar: CommSidecarSender | CommSidecarReceiver | None = None

    async def RegisterSender(  # noqa: N802
        self,
        request: comm_sidecar_pb2.RegisterSenderRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.RegisterResponse:
        """Called by the sender server to register the sidecar."""
        if self.sidecar is not None:
            logger.warning("Overwriting existing sidecar")

        dtype = getattr(torch, request.dtype)
        shm_size = self.shm_size // get_dtype_size(dtype)

        self.sidecar = CommSidecarSender(
            gpu_rank=self.gpu_rank,
            sidecar_rank=self.sidecar_rank,
            shm_size=shm_size,
            slot_size=request.slot_size,
            dtype=dtype,
            shard_rank=request.shard_rank,
            num_shards=request.num_shards,
            layout=TensorLayout(request.layout),
        )

        # gpu_rank is local rank, which is the gpu_rank on the host
        logger.info("Registered sender of gpu_rank %s, sidecar_rank %s", self.gpu_rank, self.sidecar_rank)

        return comm_sidecar_pb2.RegisterResponse(
            shm_size=shm_size,
        )

    async def RegisterReceiver(  # noqa: N802
        self,
        request: comm_sidecar_pb2.RegisterReceiverRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.RegisterResponse:
        """Called by the receiver server to register the sidecar."""
        if self.sidecar is not None:
            logger.warning("Overwriting existing sidecar")

        dtype = getattr(torch, request.dtype)
        shm_size = self.shm_size // get_dtype_size(dtype)

        self.sidecar = CommSidecarReceiver(
            gpu_rank=self.gpu_rank,
            sidecar_rank=self.sidecar_rank,
            shm_size=shm_size,
            slot_size=request.slot_size,
            dtype=request.dtype,
        )
        logger.info(
            "Registered receiver of gpu_rank %s, sidecar_rank %s, slot_size %d, dtype %s",
            self.gpu_rank,
            self.sidecar_rank,
            request.slot_size,
            request.dtype,
        )

        return comm_sidecar_pb2.RegisterResponse(
            shm_size=shm_size,
        )

    async def Send(  # noqa: N802
        self, request: comm_sidecar_pb2.SendRequest, context: grpc.aio.ServicerContext
    ) -> comm_sidecar_pb2.SendResponse:
        """Called by the sender server to send a tensor to some other rank."""
        if self.sidecar is None:
            logger.error("Sidecar not registered")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not registered")
        if not isinstance(self.sidecar, CommSidecarSender):
            logger.error("Invalid sidecar mode")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Invalid sidecar mode")
        return await self.sidecar.send(request, context)

    async def PrepareReceive(  # noqa: N802
        self,
        request: comm_sidecar_pb2.PrepareReceiveRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.PrepareReceiveResponse:
        """Called by the sender sidercar to the receiver sidecar to prepare receiving a tensor."""
        if self.sidecar is None:
            logger.error("Sidecar not registered")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not registered")
        if not isinstance(self.sidecar, CommSidecarReceiver):
            logger.error("Invalid sidecar mode")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Invalid sidecar mode")
        return await self.sidecar.prepare_receive(request, context)

    async def Receive(  # noqa: N802
        self,
        request: comm_sidecar_pb2.ReceiveRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.ReceiveResponse:
        """Initiate receiving a tensor from some other rank."""
        if self.sidecar is None:
            logger.error("Sidecar not registered")
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not registered")
        if not isinstance(self.sidecar, CommSidecarReceiver):
            logger.error("Invalid sidecar mode")
            return comm_sidecar_pb2.ReceiveResponse(offset=-1, size=-1)

        return await self.sidecar.receive(request, context)

    async def MarkDone(  # noqa: N802
        self,
        request: comm_sidecar_pb2.MarkDoneRequest,
        context: grpc.aio.ServicerContext,
    ) -> comm_sidecar_pb2.MarkDoneResponse:
        """Called by the receiver server to mark a request as done."""
        if self.sidecar is None:
            logger.error("Sidecar not registered")
            return comm_sidecar_pb2.MarkDoneResponse(status=common_pb2.Status.STATUS_ERROR)
        if not isinstance(self.sidecar, CommSidecarReceiver):
            logger.error("Invalid sidecar mode")
            return comm_sidecar_pb2.MarkDoneResponse(status=common_pb2.Status.STATUS_ERROR)

        return await self.sidecar.mark_done(request, context)

    async def shutdown(self):
        """Shutdown the sidecar."""
        if self.sidecar is not None:
            await self.sidecar.shutdown()


NAMESPACE = "cornserve"

async def _get_local_rank(pod_name: str) -> int:
    """Get the local rank of the sidecar within the node."""
    # TODO: test
    kconfig.load_incluster_config()

    async with kclient.ApiClient() as api_client:
        v1 = kclient.CoreV1Api(api_client)

        pod = await v1.read_namespaced_pod(name=pod_name, namespace=NAMESPACE)
        node_name = pod.spec.node_name
        label_selector = "app=sidecar"
        pods = await v1.list_namespaced_pod(namespace=NAMESPACE, label_selector=label_selector)
        same_node_pods = [p for p in pods.items if p.spec.node_name == node_name]
        sorted_pods = sorted(same_node_pods, key=lambda p: p.metadata.name)

        local_rank = None
        logger.info("Pods on the same node: %s", str([p.metadata.name for p in sorted_pods]))
        for index, p in enumerate(sorted_pods):
            if p.metadata.name == pod_name:
                local_rank = index
                break

        if local_rank is None:
            logger.error("Current pod not found in the list of sidecar pods on the node.")
            return -1

        logger.info("Local rank: %d", local_rank)
        return local_rank


async def main(
    ip: str = "[::]",
    base_port: int = 10000,
) -> None:
    """Main entrypoint for the sidecar server.

    Environment variables:
        - WORLD_SIZE: The total number of sidecars in the cluster.
        - MASTER_ADDR: The address of the master node.
        - MASTER_PORT: The port of the master node.
        - POD_NAME: The name of the pod the sidecar is running in.
        - SHM_SIZE: The size of the shared memory buffer in bytes.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", 48105)

    pod_name = os.environ.get("POD_NAME")
    shm_size = int(os.environ.get("SHM_SIZE", 2**28))

    if pod_name:
        try:
            gpu_rank = await _get_local_rank(pod_name)
            sidecar_rank = int(pod_name.split("-")[-1])
        except ValueError:
            gpu_rank = -1
            sidecar_rank = -1
    else:
        gpu_rank = int(os.environ.get("RANK", -1))
        sidecar_rank = gpu_rank

    assert gpu_rank >= 0, "Invalid rank"
    assert sidecar_rank >= 0, "Invalid global rank"
    global logger
    logger = SidcarAdapter(logger, sidecar_rank)

    init_url = f"tcp://{master_addr}:{master_port}"
    dist.init_process_group(backend="gloo", init_method=init_url, rank=sidecar_rank, world_size=world_size)
    logger.info("Sidecar %s out of %s, using local rank %s", sidecar_rank, world_size, gpu_rank)

    server = grpc.aio.server()
    servicer = CommSidecarServicer(gpu_rank=gpu_rank, sidecar_rank=sidecar_rank, shm_size=shm_size)
    comm_sidecar_pb2_grpc.add_CommSidecarServicer_to_server(servicer, server)
    port = base_port + sidecar_rank
    listen_addr = f"{ip}:{port}"
    server.add_insecure_port(listen_addr)
    logger.info("Sidecar server started on %s", listen_addr)
    await server.start()

    async def server_graceful_shutdown():
        logger.info("Starting graceful shutdown...")
        await servicer.shutdown()
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
