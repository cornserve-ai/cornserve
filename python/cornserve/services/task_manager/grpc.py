"""Task Manager gRPC server."""

from __future__ import annotations

import asyncio

import grpc

from cornserve.constants import SYNC_WATCHERS_TIMEOUT
from cornserve.logging import get_logger
from cornserve.services.pb import common_pb2, task_manager_pb2, task_manager_pb2_grpc
from cornserve.services.resource import GPU
from cornserve.services.task_manager.manager import TaskManager
from cornserve.services.task_registry import TaskRegistry

logger = get_logger(__name__)
cleanup_coroutines = []


class TaskManagerServicer(task_manager_pb2_grpc.TaskManagerServicer):
    """Task Manager gRPC service implementation."""

    def __init__(self, task_registry: TaskRegistry) -> None:
        """Initialize the TaskManagerServicer."""
        self.manager: TaskManager | None = None
        self.task_registry = task_registry

    async def RegisterTask(
        self,
        request: task_manager_pb2.RegisterTaskRequest,
        context: grpc.aio.ServicerContext,
    ) -> task_manager_pb2.RegisterTaskResponse:
        """Become a task manager for a new task."""
        logger.info(
            "Registering task manager %s with task CR name %s and %d GPUs",
            request.task_manager_id,
            request.task_instance_name,
            len(request.gpus),
        )

        if not all(gpu.action == task_manager_pb2.ResourceAction.ADD for gpu in request.gpus):
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "When initializing the task manager, all resources actions must be ADD",
            )

        task = await self.task_registry.get_task_instance(request.task_instance_name)
        gpus = [GPU(node=gpu.node_id, global_rank=gpu.global_rank, local_rank=gpu.local_rank) for gpu in request.gpus]

        self.manager = await TaskManager.init(id=request.task_manager_id, task=task, gpus=gpus)

        logger.info("Successfully registered task manager %s", request.task_manager_id)

        return task_manager_pb2.RegisterTaskResponse(status=common_pb2.Status.STATUS_OK)

    async def UpdateResources(
        self,
        request: task_manager_pb2.UpdateResourcesRequest,
        context: grpc.aio.ServicerContext,
    ) -> task_manager_pb2.UpdateResourcesResponse:
        """Update the resources allocated to the task manager."""
        if self.manager is None:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "Task manager not initialized with a task")

        add_gpus, remove_gpus = [], []
        for res in request.gpus:
            gpu = GPU(node=res.node_id, global_rank=res.global_rank, local_rank=res.local_rank)
            if res.action == task_manager_pb2.ResourceAction.ADD:
                add_gpus.append(gpu)
            elif res.action == task_manager_pb2.ResourceAction.REMOVE:
                remove_gpus.append(gpu)
            else:
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Unknown resource action: {res.action}")

        await self.manager.update_resources(add_gpus, remove_gpus)

        return task_manager_pb2.UpdateResourcesResponse(status=common_pb2.Status.STATUS_OK)

    async def Shutdown(
        self,
        request: task_manager_pb2.ShutdownRequest,
        context: grpc.aio.ServicerContext,
    ) -> task_manager_pb2.ShutdownResponse:
        """Shutdown the task manager."""
        if self.manager is None:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Task manager not initialized with a task",
            )
        await self.manager.shutdown()
        return task_manager_pb2.ShutdownResponse(status=common_pb2.Status.STATUS_OK)

    async def GetRoute(
        self,
        request: task_manager_pb2.GetRouteRequest,
        context: grpc.aio.ServicerContext,
    ) -> task_manager_pb2.GetRouteResponse:
        """Select which task executor the request should be routed to."""
        if self.manager is None:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "Task manager not initialized with a task",
            )
        url, sidecar_ranks = await self.manager.get_route(request.request_id, request.routing_hint)
        return task_manager_pb2.GetRouteResponse(
            task_executor_url=url,
            sidecar_ranks=sidecar_ranks,
        )

    async def SyncTaskRegistry(
        self,
        request: common_pb2.SyncTaskRegistryRequest,
        context: grpc.aio.ServicerContext,
    ) -> common_pb2.SyncTaskRegistryResponse:
        """Sync task registry to target resource versions (fetched from LatestTasklibRV CR)."""
        try:
            await asyncio.wait_for(
                self.task_registry.sync_watchers(),
                timeout=SYNC_WATCHERS_TIMEOUT,
            )
            return common_pb2.SyncTaskRegistryResponse(status=common_pb2.Status.STATUS_OK)
        except TimeoutError:
            logger.error("SyncTaskRegistry timed out")
            await context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "Sync task registry timed out")
        except Exception as e:
            logger.exception("SyncTaskRegistry failed: %s", e)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))


def create_server(task_registry: TaskRegistry) -> tuple[grpc.aio.Server, TaskManagerServicer]:
    """Create the gRPC server for the Task Manager."""
    servicer = TaskManagerServicer(task_registry)
    server = grpc.aio.server()
    task_manager_pb2_grpc.add_TaskManagerServicer_to_server(servicer, server)
    listen_addr = "[::]:50051"
    server.add_insecure_port(listen_addr)
    logger.info("Starting server on %s", listen_addr)
    return server, servicer
