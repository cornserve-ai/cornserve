"""Resource Manager gRPC server."""

from __future__ import annotations

import grpc

from cornserve.logging import get_logger
from cornserve.services.pb import common_pb2, resource_manager_pb2, resource_manager_pb2_grpc
from cornserve.services.resource_manager.manager import ResourceManager
from cornserve.task.base import UnitTask

logger = get_logger(__name__)


class ResourceManagerServicer(resource_manager_pb2_grpc.ResourceManagerServicer):
    """Resource Manager gRPC service implementation."""

    def __init__(self, manager: ResourceManager) -> None:
        """Initialize the ResourceManagerServicer."""
        self.manager = manager

    async def DeployUnitTask(
        self,
        request: resource_manager_pb2.DeployUnitTaskRequest,
        context: grpc.aio.ServicerContext,
    ) -> resource_manager_pb2.DeployUnitTaskResponse:
        """Deploy a unit task in the cluster."""
        await self.manager.deploy_unit_task(UnitTask.from_pb(request.task))
        return resource_manager_pb2.DeployUnitTaskResponse(status=common_pb2.Status.STATUS_OK)

    async def TeardownUnitTask(
        self,
        request: resource_manager_pb2.TeardownUnitTaskRequest,
        context: grpc.aio.ServicerContext,
    ) -> resource_manager_pb2.TeardownUnitTaskResponse:
        """Reconcile a removed app by shutting down task managers if needed."""
        await self.manager.teardown_unit_task(UnitTask.from_pb(request.task))
        return resource_manager_pb2.TeardownUnitTaskResponse(status=common_pb2.Status.STATUS_OK)

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
                resource_manager_pb2.TaskManagerStatus(task=task.to_pb(), status=status_map[status])
                for task, status in task_manager_statuses
            ]

            return resource_manager_pb2.HealthcheckResponse(
                status=status_map[overall_status], task_manager_statuses=proto_statuses
            )
        except Exception as e:
            logger.exception("Healthcheck failed: %s", e)
            return resource_manager_pb2.HealthcheckResponse(
                status=common_pb2.Status.STATUS_ERROR, task_manager_statuses=[]
            )


def create_server(resource_manager: ResourceManager) -> grpc.aio.Server:
    """Create the gRPC server for the Resource Manager."""
    servicer = ResourceManagerServicer(resource_manager)
    server = grpc.aio.server()
    resource_manager_pb2_grpc.add_ResourceManagerServicer_to_server(servicer, server)
    listen_addr = "[::]:50051"
    server.add_insecure_port(listen_addr)
    logger.info("Starting server on %s", listen_addr)
    return server
