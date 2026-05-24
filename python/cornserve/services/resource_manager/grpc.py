"""Resource Manager gRPC server."""

from __future__ import annotations

import asyncio

import grpc

from cornserve.constants import SYNC_WATCHERS_TIMEOUT
from cornserve.logging import get_logger
from cornserve.services.gateway.models import (
    GPUStatusPayload,
    NodeStatusPayload,
    ResourceSnapshot,
)
from cornserve.services.pb import common_pb2, resource_manager_pb2, resource_manager_pb2_grpc
from cornserve.services.resource_manager.manager import ResourceManager
from cornserve.services.task_registry import TaskRegistry

logger = get_logger(__name__)


class ResourceManagerServicer(resource_manager_pb2_grpc.ResourceManagerServicer):
    """Resource Manager gRPC service implementation."""

    def __init__(self, manager: ResourceManager, task_registry: TaskRegistry) -> None:
        """Initialize the ResourceManagerServicer."""
        self.manager = manager
        self.task_registry = task_registry

    async def DeployUnitTask(
        self,
        request: resource_manager_pb2.DeployUnitTaskRequest,
        context: grpc.aio.ServicerContext,
    ) -> resource_manager_pb2.DeployUnitTaskResponse:
        """Deploy a unit task in the cluster."""
        task = await self.task_registry.get_task_instance(request.task_instance_name)
        await self.manager.deploy_unit_task(task, request.task_instance_name)
        return resource_manager_pb2.DeployUnitTaskResponse(status=common_pb2.Status.STATUS_OK)

    async def TeardownUnitTask(
        self,
        request: resource_manager_pb2.TeardownUnitTaskRequest,
        context: grpc.aio.ServicerContext,
    ) -> resource_manager_pb2.TeardownUnitTaskResponse:
        """Tear down a unit task's task manager (free GPUs, kill pods).

        CR lifecycle is owned by the gateway — the RM only manages the
        task-manager deployment.  If the CR is already gone (gateway
        deleted it, or another replica cleaned up), fall back to
        looking up the task state by instance name.
        """
        try:
            task = await self.task_registry.get_task_instance(request.task_instance_name)
        except (ValueError, KeyError):
            # CR already deleted by the gateway.  Still free GPU state.
            logger.warning(
                "Task instance CR %s not found, attempting teardown by instance name",
                request.task_instance_name,
            )
            await self.manager.teardown_unit_task_by_instance_name(request.task_instance_name)
            return resource_manager_pb2.TeardownUnitTaskResponse(status=common_pb2.Status.STATUS_OK)

        await self.manager.teardown_unit_task(task, request.task_instance_name)
        return resource_manager_pb2.TeardownUnitTaskResponse(status=common_pb2.Status.STATUS_OK)

    async def ScaleUnitTask(
        self,
        request: resource_manager_pb2.ScaleUnitTaskRequest,
        context: grpc.aio.ServicerContext,
    ) -> resource_manager_pb2.ScaleUnitTaskResponse:
        """Scale a unit task up or down."""
        task = await self.task_registry.get_task_instance(request.task_instance_name)

        if request.num_gpus < 0:
            await self.manager.scale_down_unit_task(task, request.task_instance_name, -request.num_gpus)
        elif request.num_gpus > 0:
            await self.manager.scale_up_unit_task(task, request.task_instance_name, request.num_gpus)
        else:
            raise ValueError("The number of GPUs should not be zero.")
        return resource_manager_pb2.ScaleUnitTaskResponse(status=common_pb2.Status.STATUS_OK)

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

            # Convert the statuses into proto message format using actual CR names
            proto_statuses = []
            for task, status in task_manager_statuses:
                # Find the instance name for this task
                unit_task_instance_name = None
                for state_id, state in self.manager.task_states.items():
                    if state.deployment and state.deployment.task == task:
                        unit_task_instance_name = self.manager.unit_task_instance_names.get(state_id)
                        break

                if unit_task_instance_name is None:
                    logger.warning("No instance name found for task %s in healthcheck", task)
                    unit_task_instance_name = f"unknown-{task.make_name()}"

                proto_statuses.append(
                    resource_manager_pb2.TaskManagerStatus(
                        task_instance_name=unit_task_instance_name, status=status_map[status]
                    )
                )

            return resource_manager_pb2.HealthcheckResponse(
                status=status_map[overall_status], task_manager_statuses=proto_statuses
            )
        except Exception as e:
            logger.exception("Healthcheck failed: %s", e)
            return resource_manager_pb2.HealthcheckResponse(
                status=common_pb2.Status.STATUS_ERROR, task_manager_statuses=[]
            )

    async def GetResourceSnapshot(
        self,
        request: resource_manager_pb2.GetResourceSnapshotRequest,
        context: grpc.aio.ServicerContext,
    ) -> resource_manager_pb2.GetResourceSnapshotResponse:
        """Get a snapshot of resource allocation status."""
        try:
            snapshot = self.manager.get_resource_snapshot()

            # Convert Pydantic models to protobuf format
            nodes = [
                resource_manager_pb2.NodeStatus(
                    node=node.node,
                    total_gpus=node.total_gpus,
                    free_gpus=node.free_gpus,
                    gpus=[
                        resource_manager_pb2.GPUStatus(
                            node=gpu.node,
                            global_rank=gpu.global_rank,
                            local_rank=gpu.local_rank,
                            owner=gpu.owner if gpu.owner else "",
                        )
                        for gpu in node.gpus
                    ],
                )
                for node in snapshot.nodes
            ]

            return resource_manager_pb2.GetResourceSnapshotResponse(
                status=common_pb2.Status.STATUS_OK,
                nodes=nodes,
            )
        except Exception as e:
            logger.exception("GetResourceSnapshot failed: %s", e)
            return resource_manager_pb2.GetResourceSnapshotResponse(
                status=common_pb2.Status.STATUS_ERROR,
                nodes=[],
            )

    async def ApplyResourceSnapshot(
        self,
        request: resource_manager_pb2.ApplyResourceSnapshotRequest,
        context: grpc.aio.ServicerContext,
    ) -> resource_manager_pb2.ApplyResourceSnapshotResponse:
        """Apply a resource allocation snapshot."""
        try:
            # Convert protobuf to Pydantic models
            snapshot = ResourceSnapshot(
                nodes=[
                    NodeStatusPayload(
                        node=node.node,
                        total_gpus=node.total_gpus,
                        free_gpus=node.free_gpus,
                        gpus=[
                            GPUStatusPayload(
                                node=gpu.node,
                                global_rank=gpu.global_rank,
                                local_rank=gpu.local_rank,
                                owner=gpu.owner if gpu.owner else None,
                            )
                            for gpu in node.gpus
                        ],
                    )
                    for node in request.nodes
                ]
            )
            await self.manager.apply_resource_snapshot(snapshot)

            return resource_manager_pb2.ApplyResourceSnapshotResponse(
                status=common_pb2.Status.STATUS_OK,
                message="Resource snapshot applied successfully",
            )
        except ValueError as e:
            logger.warning("ApplyResourceSnapshot validation failed: %s", e)
            return resource_manager_pb2.ApplyResourceSnapshotResponse(
                status=common_pb2.Status.STATUS_ERROR,
                message=str(e),
            )
        except Exception as e:
            logger.exception("ApplyResourceSnapshot failed: %s", e)
            return resource_manager_pb2.ApplyResourceSnapshotResponse(
                status=common_pb2.Status.STATUS_ERROR,
                message=str(e),
            )

    async def GetDeployedTaskManagers(
        self,
        request: resource_manager_pb2.GetDeployedTaskManagersRequest,
        context: grpc.aio.ServicerContext,
    ) -> resource_manager_pb2.GetDeployedTaskManagersResponse:
        """Get list of deployed task managers."""
        try:
            task_managers = self.manager.get_deployed_task_managers()
            return resource_manager_pb2.GetDeployedTaskManagersResponse(
                status=common_pb2.Status.STATUS_OK,
                task_managers=[
                    resource_manager_pb2.DeployedTaskManager(
                        task_manager_id=tm_id,
                        task_instance_name=instance_name,
                        num_gpus=num_gpus,
                    )
                    for tm_id, instance_name, num_gpus in task_managers
                ],
            )
        except Exception as e:
            logger.exception("GetDeployedTaskManagers failed: %s", e)
            return resource_manager_pb2.GetDeployedTaskManagersResponse(
                status=common_pb2.Status.STATUS_ERROR,
                task_managers=[],
            )


def create_server(resource_manager: ResourceManager, task_registry: TaskRegistry) -> grpc.aio.Server:
    """Create the gRPC server for the Resource Manager."""
    servicer = ResourceManagerServicer(resource_manager, task_registry)
    server = grpc.aio.server()
    resource_manager_pb2_grpc.add_ResourceManagerServicer_to_server(servicer, server)
    listen_addr = "[::]:50051"
    server.add_insecure_port(listen_addr)
    logger.info("Starting server on %s", listen_addr)
    return server
