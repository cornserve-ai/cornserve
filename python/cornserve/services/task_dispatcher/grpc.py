"""Task Dispatcher gRPC server."""

from __future__ import annotations

from typing import TYPE_CHECKING

import grpc

from cornserve.services.pb import task_dispatcher_pb2, task_dispatcher_pb2_grpc, common_pb2
from cornserve.logging import get_logger
from cornserve.services.task_dispatcher.models import TaskInfo

if TYPE_CHECKING:
    from cornserve.services.task_dispatcher.dispatcher import TaskDispatcher

logger = get_logger(__name__)


class TaskDispatcherServicer(task_dispatcher_pb2_grpc.TaskDispatcherServicer):
    """Task Dispatcher gRPC service implementation."""

    def __init__(self, task_dispatcher: TaskDispatcher) -> None:
        """Initializer the TaskDispatcherServicer."""
        self.task_dispatcher = task_dispatcher

    async def NotifyAppRegistration(
        self,
        request: task_dispatcher_pb2.NotifyAppRegistrationRequest,
        context: grpc.aio.ServicerContext,
    ) -> task_dispatcher_pb2.NotifyAppRegistrationResponse:
        """Register new task managers with the task dispatcher."""
        await self.task_dispatcher.notify_app_registration(
            app_id=request.app_id,
            task_info=[TaskInfo.from_pb(task_info) for task_info in request.tasks],
        )
        return task_dispatcher_pb2.NotifyAppRegistrationResponse(status=common_pb2.Status.STATUS_OK)

    async def NotifyAppUnregistration(
        self,
        request: task_dispatcher_pb2.NotifyAppUnregistrationRequest,
        context: grpc.aio.ServicerContext,
    ) -> task_dispatcher_pb2.NotifyAppUnregistrationResponse:
        """Remove task managers from the task dispatcher."""
        await self.task_dispatcher.notify_app_unregistration(app_id=request.app_id)
        return task_dispatcher_pb2.NotifyAppUnregistrationResponse(status=common_pb2.Status.STATUS_OK)


def create_server(task_dispatcher: TaskDispatcher) -> grpc.aio.Server:
    """Create the gRPC server for the Task Dispatcher."""
    server = grpc.aio.server()
    task_dispatcher_pb2_grpc.add_TaskDispatcherServicer_to_server(TaskDispatcherServicer(task_dispatcher), server)
    server.add_insecure_port("[::]:50051")
    logger.info("gRPC server listening on [::]:50051")
    return server
