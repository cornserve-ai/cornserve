"""Task Dispatcher gRPC server."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import grpc

from cornserve.constants import SYNC_WATCHERS_TIMEOUT
from cornserve.logging import get_logger
from cornserve.services.pb import common_pb2, task_dispatcher_pb2, task_dispatcher_pb2_grpc
from cornserve.services.task_registry import TaskRegistry

if TYPE_CHECKING:
    from cornserve.services.task_dispatcher.dispatcher import TaskDispatcher

logger = get_logger(__name__)


class TaskDispatcherServicer(task_dispatcher_pb2_grpc.TaskDispatcherServicer):
    """Task Dispatcher gRPC service implementation."""

    def __init__(self, task_dispatcher: TaskDispatcher, task_registry: TaskRegistry) -> None:
        """Initializer the TaskDispatcherServicer."""
        self.task_dispatcher = task_dispatcher
        self.task_registry = task_registry

    async def NotifyUnitTaskDeployment(
        self,
        request: task_dispatcher_pb2.NotifyUnitTaskDeploymentRequest,
        context: grpc.aio.ServicerContext,
    ) -> task_dispatcher_pb2.NotifyUnitTaskDeploymentResponse:
        """Register new task managers with the task dispatcher."""
        # Reconstruct unit task from named task instance
        task = await self.task_registry.get_task_instance(request.task_instance_name)
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
        # Reconstruct unit task from named task instance
        task = await self.task_registry.get_task_instance(request.task_instance_name)
        await self.task_dispatcher.notify_task_teardown(task=task)
        return task_dispatcher_pb2.NotifyUnitTaskTeardownResponse(status=common_pb2.Status.STATUS_OK)

    async def SyncTaskRegistry(
        self,
        request: task_dispatcher_pb2.SyncTaskRegistryRequest,
        context: grpc.aio.ServicerContext,
    ) -> task_dispatcher_pb2.SyncTaskRegistryResponse:
        """Sync task registry to target resource versions (fetched from LatestTasklibRV CR)."""
        try:
            await asyncio.wait_for(
                self.task_registry.sync_watchers(),
                timeout=SYNC_WATCHERS_TIMEOUT,
            )
            return task_dispatcher_pb2.SyncTaskRegistryResponse(status=common_pb2.Status.STATUS_OK)
        except TimeoutError:
            logger.error("SyncTaskRegistry timed out")
            await context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "Sync task registry timed out")
        except Exception as e:
            logger.exception("SyncTaskRegistry failed: %s", e)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))


def create_server(task_dispatcher: TaskDispatcher, task_registry: TaskRegistry) -> grpc.aio.Server:
    """Create the gRPC server for the Task Dispatcher."""
    servicer = TaskDispatcherServicer(task_dispatcher, task_registry)
    server = grpc.aio.server()
    task_dispatcher_pb2_grpc.add_TaskDispatcherServicer_to_server(servicer, server)
    listen_addr = "[::]:50051"
    server.add_insecure_port(listen_addr)
    logger.info("gRPC server listening on %s", listen_addr)
    return server
