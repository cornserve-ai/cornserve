"""Sidecar server implementniation.

The sidecar server is a gRPC service that runs on each node in the cluster. This service
is a wrapper for the `SidecarSender` and `SidecarReceiver` services.
"""

from __future__ import annotations

import asyncio
import os
import time
import traceback

import grpc
import kubernetes_asyncio.client as kclient
import kubernetes_asyncio.config as kconfig
import numpy as np
import torch
import tyro
import ucxx
from opentelemetry import trace
from opentelemetry.instrumentation.grpc import (
    GrpcAioInstrumentorClient,
    GrpcAioInstrumentorServer,
    GrpcInstrumentorClient,
    GrpcInstrumentorServer,
)
from pydantic import ValidationError
from ucxx._lib_async.endpoint import Endpoint

from cornserve.constants import K8S_NAMESPACE
from cornserve.logging import SidcarAdapter, get_logger
from cornserve.services.pb import common_pb2, sidecar_pb2, sidecar_pb2_grpc
from cornserve.services.sidecar.receiver import SidecarReceiver
from cornserve.services.sidecar.schema import SidecarNodeInfo, SidecarServerConfig
from cornserve.services.sidecar.sender import SidecarSender
from cornserve.sidecar.utils import (
    GRPC_BASE_PORT,
    UCX_BASE_PORT,
    grpc_url_from_rank,
    ucx_port_from_rank,
    ucx_url_from_rank,
)
from cornserve.tracing import configure_otel

logger = get_logger(__name__, [SidcarAdapter])
tracer = trace.get_tracer(__name__)
cleanup_coroutines = []

STARTUP_COOLDOWN = 20


class SidecarServicer(sidecar_pb2_grpc.SidecarServicer):
    """A unified wrapper for both sender and receiver sidecar servers. Entry point for the gRPC service."""

    def __init__(
        self,
        sidecar_rank: int,
        ucx_port: int,
        world_size: int,
        mem_pressure_threshold=500,
    ) -> None:
        """Initialize the sidecar service.

        This creates an offline sidecar server that only has the CheckHealth endpoint available.

        Args:
            sidecar_rank: The global rank of the sidecar.
            ucx_port: The UCX port to use for communication.
            world_size: The total number of sidecars in the cluster.
            mem_pressure_threshold: The threshold of memory pressure count to trigger the memory pressure status.
        """
        self.sidecar_rank = sidecar_rank
        self.sender: SidecarSender | None = None
        self.receiver: SidecarReceiver | None = None
        self.config: SidecarServerConfig | None = None
        self.live = False
        self.mem_pressure_threshold = mem_pressure_threshold
        self.ucx_port = ucx_port
        self.world_size = world_size
        self.peers = dict[int, Endpoint]()
        # The scheduler is currently disabled
        # self.scheduler = Scheduler()
        # self.scheduler.start()

        async def _ucxx_listener_callback(ep: Endpoint) -> None:
            """Callback for the UCX listener."""
            id = np.empty(1, dtype=np.int32)
            await ep.recv(id)
            if id[0] in self.peers:
                logger.warning("Overwriting endpoint %d", id[0])
            self.peers[id[0]] = ep

        while True:
            try:
                listener = ucxx.create_listener(callback_func=_ucxx_listener_callback, port=self.ucx_port)
            except ucxx.exceptions.UCXBusyError:
                logger.warning("Device busy, pause for %d seconds", STARTUP_COOLDOWN)
                time.sleep(STARTUP_COOLDOWN)
            else:
                self.ucx_listener = listener
                break

    async def _reachable(self, sidecar_rank: int) -> bool:
        """Check if the sidecar is reachable.

        Args:
            sidecar_rank: The rank of the sidecar to check.
        """
        try:
            async with grpc.aio.insecure_channel(grpc_url_from_rank(sidecar_rank)) as channel:
                req = sidecar_pb2.CheckHealthRequest()
                stub = sidecar_pb2_grpc.SidecarStub(channel)
                _ = await stub.CheckHealth(req)
                return True
        except Exception:
            return False

    async def p2p_connect(self) -> None:
        """Connect to other peers using UCX.

        Establishes a UCX connection to all other sidecars in the cluster.
        Proactively connects to peers with lower sidecar ranks,
        and waits to be connected by peers with higher sidecar ranks.
        """
        for i in range(self.sidecar_rank):
            while not await self._reachable(i):
                logger.info("Waiting for sidecar-%d to be reachable", i)
                await asyncio.sleep(10.1)
            while i not in self.peers:
                try:
                    logger.info(
                        "Connecting to sidecar-%d - url %s - port %d",
                        i,
                        ucx_url_from_rank(i),
                        ucx_port_from_rank(i),
                    )
                    ep = await ucxx.create_endpoint(ucx_url_from_rank(i), ucx_port_from_rank(i))
                    msg = np.array([self.sidecar_rank], dtype=np.int32)
                    await ep.send(msg)
                    self.peers[i] = ep
                except Exception:
                    await asyncio.sleep(0.1)

        # wait for other peers to connect to us
        while len(self.peers) < self.world_size - 1:
            await asyncio.sleep(0.5)
        logger.info("Connected to all peers")

    def online(self, node_info: SidecarNodeInfo, shm_size: int) -> None:
        """Mark the sidecar as online.

        Args:
            node_info: The sidecar information within the node.
            shm_size: The bytesize of the shared memory buffer used by each sidecar server.
        """
        self.node_info = node_info
        self.device_id = self.node_info.get_device_id(self.sidecar_rank)
        self.num_devices = self.node_info.get_sidecar_num()
        self.shm_size = shm_size
        self.live = True
        logger.info("Sidecar online")

    async def CheckHealth(  # noqa: N802
        self,
        request: sidecar_pb2.CheckHealthRequest,
        context: grpc.aio.ServicerContext,
    ) -> sidecar_pb2.CheckHealthResponse:
        """Health check for the sidecar."""
        try:
            if not self.live:
                return sidecar_pb2.CheckHealthResponse(status=sidecar_pb2.HealthStatus.HEALTH_OFFLINE)
            if (self.sender is not None and self.sender.mem_pressure_count > self.mem_pressure_threshold) or (
                self.receiver is not None and self.receiver.mem_pressure_count > self.mem_pressure_threshold
            ):
                return sidecar_pb2.CheckHealthResponse(status=sidecar_pb2.HealthStatus.HEALTH_MEMORY_PRESSURE)
            return sidecar_pb2.CheckHealthResponse(status=sidecar_pb2.HealthStatus.HEALTH_ALL_GOOD)
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.exception("Error in CheckHealth with request %s", request)
            await context.abort(grpc.StatusCode.INTERNAL, f"Error in CheckHealth: {e} \n {tb_str}")

    def _build_config(self, request: sidecar_pb2.RegisterRequest) -> SidecarServerConfig:
        """Build the sidecar config from the register request."""
        # currently the shared memory is partitioned equally between the sender and receiver
        # TODO: watermark alloc
        slab_size = self.shm_size // 2
        tensor_dtype = getattr(torch, request.dtype)
        slab_numel = slab_size // tensor_dtype.itemsize
        return SidecarServerConfig(
            sidecar_rank=request.rank,
            node_info=self.node_info,
            peers=self.peers,
            group=sorted(list(request.group)),
            tensor_dtype=tensor_dtype,
            slab_numel=slab_numel,
            send_slot_numel=request.send_slot_numel,
            recv_slot_numel=request.recv_slot_numel,
            concurrent_copy=request.concurrent_copy,
        )

    async def Register(  # noqa: N802
        self,
        request: sidecar_pb2.RegisterRequest,
        context: grpc.aio.ServicerContext,
    ) -> sidecar_pb2.RegisterResponse:
        """Register the sidecar."""
        try:
            if not self.live:
                logger.error("Sidecar not online")
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not online")

            # validate the request
            assert request.rank >= 0
            assert request.group
            assert request.dtype
            assert request.send_slot_numel
            assert request.recv_slot_numel
            assert request.concurrent_copy is not None

            if any(r not in self.node_info.sidecar_ranks for r in list(request.group)):
                logger.error("Sidecar ranks not in node")
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar ranks not in node")

            new_config = self._build_config(request)
            if self.config is None or self.config != new_config:
                logger.warning("Registering new sidecar")
                try:
                    self.config = new_config
                    self.sender = SidecarSender(self.config.sender_config())
                    self.receiver = SidecarReceiver(self.config.receiver_config())
                except ValidationError:
                    logger.exception("Invalid sidecar config")
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "Invalid sidecar config")
            else:
                if request.rank not in self.config.group:
                    logger.error("Sidecar rank not in group")
                    await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar rank not in group")
                logger.info("Sidecar %d in group %s connected", request.rank, request.group)
            return sidecar_pb2.RegisterResponse(
                status=common_pb2.Status.STATUS_OK,
                shm_size=self.shm_size,
                local_rank=self.config.group.index(request.rank),
                num_local_sidecars=self.num_devices,
            )
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.exception("Error in Register")
            await context.abort(grpc.StatusCode.INTERNAL, f"Error in Register: {e} \n {tb_str}")

    async def Send(  # noqa: N802
        self,
        request: sidecar_pb2.SendRequest,
        context: grpc.aio.ServicerContext,
    ) -> sidecar_pb2.SendResponse:
        """Called by the sender server to send a tensor to some other rank."""
        try:
            if not self.live:
                logger.error("Sidecar not online")
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not online")
            if self.sender is None:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not registered")
            return await self.sender.send(request, context)
            # return await self.scheduler.submit(self.sender.send, request, context)
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.exception("Error in Send")
            await context.abort(grpc.StatusCode.INTERNAL, f"Error in Send: {e} \n {tb_str}")

    async def PrepareReceive(  # noqa: N802
        self,
        request: sidecar_pb2.PrepareReceiveRequest,
        context: grpc.aio.ServicerContext,
    ) -> sidecar_pb2.PrepareReceiveResponse:
        """Called by the sender sidercar to the receiver sidecar to prepare receiving a tensor."""
        try:
            if not self.live:
                logger.error("Sidecar not online")
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not online")
            if self.receiver is None:
                logger.error("Sidecar not registered")
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not registered")
            return await self.receiver.prepare_receive(request, context)
            # return await self.scheduler.submit(self.receiver.prepare_receive, request, context)
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.exception("Error in PrepareReceive")
            await context.abort(grpc.StatusCode.INTERNAL, f"Error in PrepareReceive: {e} \n {tb_str}")

    async def Receive(  # noqa: N802
        self,
        request: sidecar_pb2.ReceiveRequest,
        context: grpc.aio.ServicerContext,
    ) -> sidecar_pb2.ReceiveResponse:
        """Initiate receiving a tensor from some other rank."""
        try:
            if not self.live:
                logger.error("Sidecar not online")
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not online")
            if self.receiver is None:
                logger.error("Sidecar not registered")
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not registered")
            return await self.receiver.receive(request, context)
            # return await self.scheduler.submit(self.receiver.receive, request, context)
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.exception("Error in Receive")
            await context.abort(grpc.StatusCode.INTERNAL, f"Error in Receive: {e} \n {tb_str}")

    async def MarkDone(  # noqa: N802
        self,
        request: sidecar_pb2.MarkDoneRequest,
        context: grpc.aio.ServicerContext,
    ) -> sidecar_pb2.MarkDoneResponse:
        """Called by the receiver server to mark a request as done."""
        try:
            if not self.live:
                logger.error("Sidecar not online")
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not online")
            if self.receiver is None:
                logger.error("Sidecar not registered")
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not registered")
            return await self.receiver.mark_done(request, context)
            # return await self.scheduler.submit(self.receiver.mark_done, request, context)
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.exception("Error in MarkDone")
            await context.abort(grpc.StatusCode.INTERNAL, f"Error in MarkDone: {e} \n {tb_str}")

    async def Unlink(  # noqa: N802
        self,
        request: sidecar_pb2.UnlinkRequest,
        context: grpc.aio.ServicerContext,
    ) -> sidecar_pb2.UnlinkResponse:
        """Called by the receiver server to mark a request as done."""
        try:
            if not self.live:
                logger.error("Sidecar not online")
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not online")
            if self.sender is None:
                logger.error("Sidecar not registered")
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Sidecar not registered")
            return await self.sender.unlink(request, context)
            # return await self.scheduler.submit(self.sender.unlink, request, context)
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.exception("Error in Unlink")
            await context.abort(grpc.StatusCode.INTERNAL, f"Error in Unlink: {e} \n {tb_str}")

    async def shutdown(self):
        """Shutdown the sidecar."""
        if self.sender is not None:
            await self.sender.shutdown()
        if self.receiver is not None:
            await self.receiver.shutdown()
        for peer in self.peers.values():
            await peer.close()
        # await self.scheduler.stop()
        self.ucx_listener.close()
        logger.info("Sidecar shutdown")


async def _get_node_info(pod_name: str) -> SidecarNodeInfo | None:
    """Query the Kubernetes API to get the node information."""
    kconfig.load_incluster_config()
    async with kclient.ApiClient() as api_client:
        v1 = kclient.CoreV1Api(api_client)
        pod = await v1.read_namespaced_pod(name=pod_name, namespace=K8S_NAMESPACE)  # pyright: ignore
        node_name = pod.spec.node_name  # pyright: ignore
        label_selector = "app=sidecar"
        pods = await v1.list_namespaced_pod(namespace=K8S_NAMESPACE, label_selector=label_selector)
        same_node_pod_names = [p.metadata.name for p in pods.items if p.spec.node_name == node_name]
        sorted_pod_names = sorted(same_node_pod_names)
        if pod_name not in sorted_pod_names:
            logger.error("Current pod not found in the list of sidecar pods on the node. K8s issue?")
            return None
        return SidecarNodeInfo([int(pod_name.split("-")[-1]) for pod_name in sorted_pod_names])


async def main(
    ip: str = "[::]",
) -> None:
    """Main entrypoint for the sidecar server.

    The entrypoint uses the following environment variables to configure the sidecar server.
    When launched in a Kubernetes cluster, the `SIDECAR_POD_NAME` environment variable is used
    to determine the global rank and the GPU device used for each sidecar.

    When launched outside of a Kubernetes cluster, the `SIDECAR_RANK` environment variable is used
    to determine the global rank and the GPU device used for each sidecar, which will be the same.
    Note this means that outside of k8s, only single node is supported.

    Environment variables:
        SIDECAR_WORLD_SIZE: The total number of sidecars in the cluster.
        SIDECAR_SHM_SIZE: The size of the shared memory buffer in bytes in each sidecar
            this will be divided by the dtype size so it should be a multiple of the dtype size.

        K8s only:
        SIDECAR_POD_NAME: The name of the pod the sidecar is running in.

        Outside of k8s:
        SIDECAR_RANK: The global rank of the sidecar
        SIDECAR_DEVICE_ID: The device id of the GPU used by the sidecar, will use SIDECAR_RANK if not set.
        SIDECAR_NUM_DEVICES: The number of devices on the node, will use SIDECAR_WORLD_SIZE if not set.
    """
    world_size = int(os.environ.get("SIDECAR_WORLD_SIZE", "1"))
    shm_size = int(os.environ.get("SIDECAR_SHM_SIZE", str(2**30)))

    assert world_size > 0, "Invalid SIDECAR_WORLD_SIZE"
    pod_name = os.environ.get("SIDECAR_POD_NAME")

    if pod_name:
        try:
            sidecar_rank = int(pod_name.split("-")[-1])
        except ValueError:
            sidecar_rank = -1
    else:
        sidecar_rank = int(os.environ.get("SIDECAR_RANK", "-1"))

    assert sidecar_rank >= 0, "Invalid sidecar rank"

    # OpenTelemetry setup
    configure_otel(name=f"sidecar[{sidecar_rank}]")

    GrpcInstrumentorClient().instrument()
    GrpcInstrumentorServer().instrument()
    GrpcAioInstrumentorClient().instrument()
    GrpcAioInstrumentorServer().instrument()

    # We start the server so the health check gRPC is always available
    server = grpc.aio.server()
    ucx_port = UCX_BASE_PORT + sidecar_rank
    servicer = SidecarServicer(
        sidecar_rank=sidecar_rank,
        ucx_port=ucx_port,
        world_size=world_size,
    )
    sidecar_pb2_grpc.add_SidecarServicer_to_server(servicer, server)
    port = GRPC_BASE_PORT + sidecar_rank
    listen_addr = f"{ip}:{port}"
    server.add_insecure_port(listen_addr)
    await server.start()
    logger.info("Sidecar server started on %s", listen_addr)

    async def server_graceful_shutdown():
        logger.info("Starting graceful shutdown...")
        await servicer.shutdown()
        await server.stop(5)
        logger.info("Server stopped")

    logger.info("Starting sidecar server %s", sidecar_rank)
    await servicer.p2p_connect()

    # now that every sidecar server has started, we query the cluster to retrieve
    # the device_id and num_devices within the node when using k8s
    if pod_name:
        node_info = await _get_node_info(pod_name)
    else:
        # outside of k8s, currently limited to identity mapping
        cluster_ranks_str = os.environ.get("SIDECAR_CLUSTER_RANKS")
        if cluster_ranks_str:
            try:
                cluster_ranks = list(map(int, cluster_ranks_str.split(",")))
            except ValueError as err:
                logger.exception("Invalid SIDECAR_CLUSTER_RANKS")
                raise ValueError("Invalid SIDECAR_CLUSTER_RANKS") from err
        else:
            cluster_ranks = list(range(world_size))
        node_info = SidecarNodeInfo(cluster_ranks)

    assert node_info is not None, "Failed to get node info"

    assert shm_size % torch.cdouble.itemsize == 0, (
        "shm_size should be a multiple of num_devices * max(torch.cdouble) dtype itemsize"
    )
    # sidecar group p2p connected, now we can mark the server live
    servicer.online(node_info=node_info, shm_size=shm_size)

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
