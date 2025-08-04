"""Task Dispatcher gRPC server."""

from __future__ import annotations

from typing import TYPE_CHECKING

import grpc

from cornserve.logging import get_logger
from cornserve.services.pb import common_pb2, task_dispatcher_pb2, task_dispatcher_pb2_grpc
from cornserve.services.cr_manager.manager import CRManager
from cornserve.task.base import UnitTask

if TYPE_CHECKING:
    from cornserve.services.task_dispatcher.dispatcher import TaskDispatcher

logger = get_logger(__name__)


class TaskDispatcherServicer(task_dispatcher_pb2_grpc.TaskDispatcherServicer):
    """Task Dispatcher gRPC service implementation."""

    def __init__(self, task_dispatcher: TaskDispatcher, cr_manager: CRManager) -> None:
        """Initializer the TaskDispatcherServicer."""
        self.task_dispatcher = task_dispatcher
        self.cr_manager = cr_manager

    async def NotifyUnitTaskDeployment(
        self,
        request: task_dispatcher_pb2.NotifyUnitTaskDeploymentRequest,
        context: grpc.aio.ServicerContext,
    ) -> task_dispatcher_pb2.NotifyUnitTaskDeploymentResponse:
        """Register new task managers with the task dispatcher."""
        # Reconstruct UnitTask from UnitTaskInstance CR instead of protobuf
        task = await self.cr_manager.get_unit_task_from_instance_cr(request.task_cr_name)
        await self.task_dispatcher.notify_task_deployment(
            task=task,
            task_manager_url=request.task_manager.url,
        )
        return task_dispatcher_pb2.NotifyUnitTaskDeploymentResponse(status=common_pb2.Status.STATUS_OK)

    async def NotifyUnitTaskTeardown(
        self,
        request: task_dispatcher_pb2.NotifyUnitTaskTeardownRequest,
        context: grpc.aio.ServicerContext,
    ) -> task_dispatcher_pb2.NotifyUnitTaskTeardownResponse:
        """Remove task managers from the task dispatcher."""
        # Reconstruct UnitTask from UnitTaskInstance CR instead of protobuf
        task = await self.cr_manager.get_unit_task_from_instance_cr(request.task_cr_name)
        await self.task_dispatcher.notify_task_teardown(task=task)
        return task_dispatcher_pb2.NotifyUnitTaskTeardownResponse(status=common_pb2.Status.STATUS_OK)


def create_server(task_dispatcher: TaskDispatcher, cr_manager: CRManager) -> grpc.aio.Server:
    """Create the gRPC server for the Task Dispatcher."""
    servicer = TaskDispatcherServicer(task_dispatcher, cr_manager)
    server = grpc.aio.server()
    task_dispatcher_pb2_grpc.add_TaskDispatcherServicer_to_server(servicer, server)
    listen_addr = "[::]:50051"
    server.add_insecure_port(listen_addr)
    logger.info("gRPC server listening on %s", listen_addr)
    return server
