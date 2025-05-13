"""Task Manager gRPC server."""

from __future__ import annotations

import grpc

from cornserve.logging import get_logger
from cornserve.services.pb import common_pb2, task_manager_pb2, task_manager_pb2_grpc
from cornserve.services.resource_manager.resource import GPU
from cornserve.services.task_manager.manager import TaskManager
from cornserve.task.base import UnitTask

logger = get_logger(__name__)
cleanup_coroutines = []


class TaskManagerServicer(task_manager_pb2_grpc.TaskManagerServicer):
    """Task Manager gRPC service implementation."""

    def __init__(self) -> None:
        """Initialize the TaskManagerServicer."""
        self.manager: TaskManager | None = None

    async def RegisterTask(
        self,
        request: task_manager_pb2.RegisterTaskRequest,
        context: grpc.aio.ServicerContext,
    ) -> task_manager_pb2.RegisterTaskResponse:
        """Become a task manager for a new task."""
        logger.info(
            "Registering task manager %s with task class %s and config %s with %d GPUs",
            request.task_manager_id,
            request.task.task_class_name,
            request.task.task_config,
            len(request.gpus),
        )

        if not all(gpu.action == task_manager_pb2.ResourceAction.ADD for gpu in request.gpus):
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "When initializing the task manager, all resources actions must be ADD",
            )

        task = UnitTask.from_pb(request.task)
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


def create_server() -> tuple[grpc.aio.Server, TaskManagerServicer]:
    """Create the gRPC server for the Task Manager."""
    servicer = TaskManagerServicer()
    server = grpc.aio.server()
    task_manager_pb2_grpc.add_TaskManagerServicer_to_server(servicer, server)
    listen_addr = "[::]:50051"
    server.add_insecure_port(listen_addr)
    logger.info("Starting server on %s", listen_addr)
    return server, servicer
