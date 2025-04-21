"""Task manager that manages registered and deployed tasks."""

from __future__ import annotations

import asyncio
import enum
import uuid
from collections import defaultdict
from typing import Any

import grpc
import httpx
from opentelemetry import trace
from pydantic import ValidationError

from cornserve.constants import K8S_TASK_DISPATCHER_HTTP_URL
from cornserve.logging import get_logger
from cornserve.services.gateway.models import UnitTaskSpec
from cornserve.services.pb.resource_manager_pb2 import DeployUnitTaskRequest, TeardownUnitTaskRequest
from cornserve.services.pb.resource_manager_pb2_grpc import ResourceManagerStub
from cornserve.task.base import TaskGraphDispatch, UnitTask
from cornserve.task.registry import TASK_REGISTRY

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class TaskState(enum.StrEnum):
    """Possible states of a task."""

    # Task is currently being deployed
    DEPLOYING = "not ready"

    # Task is ready to be invoked
    READY = "ready"

    # Task is currently being torn down
    TEARING_DOWN = "tearing down"


class TaskManager:
    """Manages registered and deployed tasks."""

    def __init__(self, resource_manager_grpc_url: str) -> None:
        """Initialize the task manager.

        Args:
            resource_manager_grpc_url: The gRPC URL of the resource manager.
        """
        # A big lock to protect all task states
        self.task_lock = asyncio.Lock()

        # Task-related state. Key is the task ID.
        self.tasks: dict[str, UnitTask] = {}
        self.task_states: dict[str, TaskState] = {}  # Can be read without holding lock.
        self.task_invocation_tasks: dict[str, list[asyncio.Task]] = defaultdict(list)

        # gRPC client for resource manager
        self.resource_manager_channel = grpc.aio.insecure_channel(resource_manager_grpc_url)
        self.resource_manager = ResourceManagerStub(self.resource_manager_channel)

    async def deploy_tasks(self, task_specs: list[UnitTaskSpec]) -> list[str]:
        """Deploy the given tasks.

        If a task is already deployed, it will be skipped.
        An error raised during deployment will roll back the deployment of all tasks deployed.

        Args:
            task_specs: The list of tasks to deploy.

        Returns:
            The list of task IDs.
        """
        # Check task state to find out which tasks have to be deployed
        task_ids: list[str] = []
        to_deploy: list[str] = []
        async with self.task_lock:
            for task_spec in task_specs:
                task_class = TASK_REGISTRY.get(task_spec.task_class_name)

                try:
                    task = task_class.model_validate(task_spec.task_config)
                except ValidationError as e:
                    logger.error("Validation error while constructing unit task instance (%s): %s", task_spec, e)
                    raise ValueError(
                        f"Validation error while constructing unit task instance ({task_spec}): {e}"
                    ) from e

                # Check if the task is already deployed
                for task_id, existing_task in self.tasks.items():
                    if existing_task == task:
                        logger.info("Task %s is already deployed, skipping", task_spec)
                        task_ids.append(task_id)
                        break
                else:
                    # If the task is not already deployed, deploy it
                    logger.info("Task %s should be deployed", task_spec)

                    # Generate a unique ID for the task
                    while True:
                        task_id = f"{task_spec.task_class_name.lower()}-{uuid.uuid4().hex}"
                        if task_id not in self.tasks:
                            break

                    self.tasks[task_id] = task
                    self.task_states[task_id] = TaskState.DEPLOYING
                    task_ids.append(task_id)
                    to_deploy.append(task_id)

            # Deploy tasks
            coros = []
            for task_id in to_deploy:
                task = self.tasks[task_id]
                coros.append(self.resource_manager.DeployUnitTask(DeployUnitTaskRequest(task=task.to_pb())))
            responses = await asyncio.gather(*coros, return_exceptions=True)

            # Check for errors
            errors: list[BaseException] = []
            deployed_tasks: list[str] = []
            for resp, deployed_task in zip(responses, to_deploy, strict=True):
                if isinstance(resp, BaseException):
                    logger.error("Error while deploying task: %s", resp)
                    errors.append(resp)
                else:
                    deployed_tasks.append(deployed_task)

            # Roll back successful deployments if something went wrong.
            # We're treating the whole list of deployments as a single transaction.
            if errors:
                cleanup_coros = []
                for task_id in deployed_tasks:
                    task = self.tasks[task_id]
                    cleanup_coros.append(
                        self.resource_manager.TeardownUnitTask(TeardownUnitTaskRequest(task=task.to_pb()))
                    )
                    del self.tasks[task_id]
                    del self.task_states[task_id]
                await asyncio.gather(*cleanup_coros)
                logger.info("Rolled back deployment of all deployed tasks")
                raise RuntimeError(f"Error while deploying tasks: {errors}")

            # Update task states
            for task_id in task_ids:
                if task_id not in self.tasks:
                    raise ValueError(f"Task with ID {task_id} does not exist")
                self.task_states[task_id] = TaskState.READY

        return task_ids

    async def teardown_tasks(self, task_specs: list[UnitTaskSpec]) -> None:
        """Teardown the given tasks.

        If the specific task is not deployed, it will be skipped.
        An error raised during tear down will *not* roll back the tear down of other tasks.

        Args:
            task_specs: The list of tasks to teardown.
        """
        async with self.task_lock:
            to_teardown: list[str] = []
            for task_spec in task_specs:
                task_class = TASK_REGISTRY.get(task_spec.task_class_name)

                try:
                    task = task_class.model_validate(task_spec.task_config)
                except ValidationError as e:
                    logger.error("Validation error while constructing unit task instance (%s): %s", task_spec, e)
                    raise ValueError(
                        f"Validation error while constructing unit task instance ({task_spec}): {e}"
                    ) from e

                # Check if the task is deployed
                for task_id, existing_task in self.tasks.items():
                    if existing_task == task:
                        logger.info("Task %s is deployed, tearing down", task_spec)
                        to_teardown.append(task_id)
                        self.task_states[task_id] = TaskState.TEARING_DOWN
                        break
                else:
                    logger.info("Task %s is not deployed, skipping", task_spec)
                    continue

            # Cancel running invocation of tasks
            for task_id in to_teardown:
                if task_id not in self.task_invocation_tasks:
                    continue
                for invocation_task in self.task_invocation_tasks[task_id]:
                    invocation_task.cancel()
                del self.task_invocation_tasks[task_id]

            # Teardown tasks
            coros = []
            for task_id in to_teardown:
                task = self.tasks[task_id]
                coros.append(self.resource_manager.TeardownUnitTask(TeardownUnitTaskRequest(task=task.to_pb())))
            responses = await asyncio.gather(*coros, return_exceptions=True)

            # Check for errors and update task states
            errors: list[BaseException] = []
            for resp, task_id in zip(responses, to_teardown, strict=True):
                if isinstance(resp, BaseException):
                    logger.error("Error while tearing down task: %s", resp)
                    errors.append(resp)
                else:
                    del self.tasks[task_id]
                    del self.task_states[task_id]
                    logger.info("Task %s has been torn down", task_id)

            if errors:
                logger.error("Errors occured while tearing down tasks")
                raise RuntimeError(f"Error while tearing down tasks: {errors}")

    async def invoke_tasks(self, dispatch: TaskGraphDispatch) -> list[dict[str, Any]]:
        """Invoke the given tasks.

        Before invocation, this method ensures that all tasks part of the invocation
        are deployed and ready to be invoked. It is ensured that the number of outputs
        returned by the task dispatcher matches the number of invocations.

        Args:
            dispatch: The dispatch object containing the tasks to invoke.

        Returns:
            The outputs of all tasks.
        """
        # Check if all tasks are deployed
        async with self.task_lock:
            for invocation in dispatch.invocations:
                for task_id, task in self.tasks.items():
                    if task == invocation.task:
                        match self.task_states[task_id]:
                            case TaskState.READY:
                                break
                            case TaskState.DEPLOYING:
                                raise ValueError(f"Task {invocation.task} is being deployed")
                            case TaskState.TEARING_DOWN:
                                raise ValueError(f"Task {invocation.task} is being torn down")
                else:
                    raise KeyError(f"Task {invocation.task} is not deployed")

        # Dispatch to the Task Dispatcher
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                response = await client.post(K8S_TASK_DISPATCHER_HTTP_URL + "/task", json=dispatch.model_dump())
                response.raise_for_status()
            except httpx.RequestError as e:
                logger.error("Error while invoking tasks: %s", e)
                raise RuntimeError(f"Error while invoking tasks: {e}") from e

        output = response.json()
        if not isinstance(output, list):
            raise RuntimeError(f"Invalid response from task dispatcher: {output}")
        if len(output) != len(dispatch.invocations):
            raise RuntimeError(
                f"Invalid response from task dispatcher: {output} (expected {len(dispatch.invocations)} outputs)"
            )
        return output
