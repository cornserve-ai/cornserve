"""Resource Manager gRPC server."""

import asyncio

import grpc
import tyro
from opentelemetry.instrumentation.grpc import GrpcAioInstrumentorClient, GrpcAioInstrumentorServer

from cornserve.frontend.tasks import Task
from cornserve.logging import get_logger
from cornserve.services.pb import common_pb2, resource_manager_pb2, resource_manager_pb2_grpc
from cornserve.services.resource_manager.manager import ResourceManager
from cornserve.tracing import configure_otel

logger = get_logger(__name__)
cleanup_coroutines = []


class ResourceManagerServicer(resource_manager_pb2_grpc.ResourceManagerServicer):
    """Resource Manager gRPC service implementation."""

    def __init__(self, manager: ResourceManager) -> None:
        """Initialize the ResourceManagerServicer."""
        self.manager = manager

    async def ReconcileNewApp(
        self,
        request: resource_manager_pb2.ReconcileNewAppRequest,
        context: grpc.aio.ServicerContext,
    ) -> resource_manager_pb2.ReconcileNewAppResponse:
        """Reconcile a new app by spawning task managers if needed."""
        await self.manager.reconcile_new_app(
            app_id=request.app_id,
            tasks=[
                Task.from_json(type=common_pb2.TaskType.Name(config.type), json=config.config)
                for config in request.task_configs
            ],
        )
        return resource_manager_pb2.ReconcileNewAppResponse(status=common_pb2.Status.STATUS_OK)

    async def ReconcileRemovedApp(
        self,
        request: resource_manager_pb2.ReconcileRemovedAppRequest,
        context: grpc.aio.ServicerContext,
    ) -> resource_manager_pb2.ReconcileRemovedAppResponse:
        """Reconcile a removed app by shutting down task managers if needed."""
        await self.manager.reconcile_removed_app(app_id=request.app_id)
        return resource_manager_pb2.ReconcileRemovedAppResponse(status=common_pb2.Status.STATUS_OK)

    async def Healthcheck(
        self,
        request: resource_manager_pb2.HealthcheckRequest,
        context: grpc.aio.ServicerContext,
    ) -> resource_manager_pb2.HealthcheckResponse:
        """Recursively check and report the health of task managers."""
        try:
            overall_status, task_manager_statuses = await self.manager.healthcheck()

            status_map = {
                True: common_pb2.Status.STATUS_OK,
                False: common_pb2.Status.STATUS_ERROR,
            }

            # Convert the statuses into proto message format
            proto_statuses = [
                resource_manager_pb2.TaskManagerStatus(task_manager_id=task_manager_id, status=status_map[status])
                for task_manager_id, status in task_manager_statuses
            ]

            return resource_manager_pb2.HealthcheckResponse(
                status=status_map[overall_status], task_manager_statuses=proto_statuses
            )
        except Exception as e:
            logger.exception("Healthcheck failed: %s", e)
            return resource_manager_pb2.HealthcheckResponse(
                status=common_pb2.Status.STATUS_ERROR, task_manager_statuses=[]
            )


async def serve(ip: str = "[::]", port: int = 50051) -> None:
    """Start the gRPC server."""
    configure_otel("resource_manager")

    GrpcAioInstrumentorServer().instrument()
    GrpcAioInstrumentorClient().instrument()

    manager = await ResourceManager.init()

    server = grpc.aio.server()
    resource_manager_pb2_grpc.add_ResourceManagerServicer_to_server(ResourceManagerServicer(manager), server)
    listen_addr = f"{ip}:{port}"
    server.add_insecure_port(listen_addr)
    logger.info("Starting server on %s", listen_addr)

    await server.start()

    logger.info("Server started")

    async def server_graceful_shutdown():
        logger.info("Starting graceful shutdown...")
        # Shuts down the server with 5 seconds of grace period. During the
        # grace period, the server won't accept new connections and allow
        # existing RPCs to continue within the grace period.
        await server.stop(5)
        await manager.shutdown()
        logger.info("Server stopped")

    cleanup_coroutines.append(server_graceful_shutdown())
    await server.wait_for_termination()


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(tyro.cli(serve))
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(asyncio.gather(*cleanup_coroutines))
        loop.close()
