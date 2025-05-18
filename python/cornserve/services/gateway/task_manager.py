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

from cornserve.constants import K8S_TASK_DISPATCHER_HTTP_URL
from cornserve.logging import get_logger
from cornserve.services.pb.resource_manager_pb2 import DeployUnitTaskRequest, TeardownUnitTaskRequest
from cornserve.services.pb.resource_manager_pb2_grpc import ResourceManagerStub
from cornserve.task.base import TaskGraphDispatch, UnitTask

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
        self.task_usage_counter: dict[str, int] = defaultdict(int)

        # gRPC client for resource manager
        self.resource_manager_channel = grpc.aio.insecure_channel(resource_manager_grpc_url)
        self.resource_manager = ResourceManagerStub(self.resource_manager_channel)

    async def declare_used(self, tasks: list[UnitTask]) -> None:
        """Deploy the given tasks.

        If a task is already deployed, it will be skipped.
        An error raised during deployment will roll back the deployment of all tasks deployed.
        """
        # Check task state to find out which tasks have to be deployed
        task_ids: list[str] = []
        to_deploy: list[str] = []
        async with self.task_lock:
            for task in tasks:
                # Check if the task is already deployed
                for task_id, existing_task in self.tasks.items():
                    if existing_task.is_equivalent_to(task):
                        logger.info("Skipping already deployed task: %r", task)
                        task_ids.append(task_id)
                        break
                else:
                    # If the task is not already deployed, deploy it
                    logger.info("Should deploy task: %r", task)

                    # Generate a unique ID for the task
                    while True:
                        task_id = f"{task.__class__.__name__.lower()}-{uuid.uuid4().hex}"
                        if task_id not in self.tasks:
                            break

                    self.tasks[task_id] = task
                    self.task_states[task_id] = TaskState.DEPLOYING
                    task_ids.append(task_id)
                    to_deploy.append(task_id)

                # Whether or not it was already deployed, increment the usage counter
                self.task_usage_counter[task_id] += 1

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
                for task_id in to_deploy:
                    task = self.tasks[task_id]
                    cleanup_coros.append(
                        self.resource_manager.TeardownUnitTask(TeardownUnitTaskRequest(task=task.to_pb()))
                    )
                    del self.tasks[task_id]
                    del self.task_states[task_id]
                    self.task_usage_counter[task_id] -= 1
                    if self.task_usage_counter[task_id] == 0:
                        del self.task_usage_counter[task_id]
                await asyncio.gather(*cleanup_coros)
                logger.info("Rolled back deployment of all deployed tasks")
                raise RuntimeError(f"Error while deploying tasks: {errors}")

            # Update task states
            for task_id in task_ids:
                if task_id not in self.tasks:
                    raise ValueError(f"Task with ID {task_id} does not exist")
                self.task_states[task_id] = TaskState.READY

    async def declare_not_used(self, tasks: list[UnitTask]) -> None:
        """Declare that the given tasks are not used anymore.

        This will decrease the usage counter of the tasks and tear them down if the usage
        counter reaches 0. If the specific task is not deployed, it will be skipped.
        An error raised during tear down will *not* roll back the tear down of other tasks.
        """
        async with self.task_lock:
            to_teardown: list[str] = []
            for task in tasks:
                # Check if the task is deployed
                for task_id, existing_task in self.tasks.items():
                    if existing_task.is_equivalent_to(task):
                        usage_counter = self.task_usage_counter[task_id]
                        assert usage_counter > 0, f"{task!r} has usage counter of 0"
                        usage_counter -= 1
                        self.task_usage_counter[task_id] = usage_counter
                        # This task should be torn down
                        if usage_counter == 0:
                            logger.info("Last usage of task was removed, tearing down: %r", task)
                            to_teardown.append(task_id)
                            self.task_states[task_id] = TaskState.TEARING_DOWN
                            # Cancel running invocation of tasks
                            for invocation_task in self.task_invocation_tasks.pop(task_id, []):
                                invocation_task.cancel()
                        else:
                            logger.info("Usage count is %d, skipping teardown: %r", usage_counter, task)
                        break
                # Task is not deployed
                else:
                    logger.warning("Cannot find task, skipping teardown: %r", task)
                    continue

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
                    logger.error("Error while tearing down: %r", resp)
                    errors.append(resp)
                else:
                    del self.tasks[task_id]
                    del self.task_states[task_id]
                    del self.task_usage_counter[task_id]
                    logger.info("Teardown complete: %r", task_id)

            if errors:
                logger.error("Errors occured while tearing down tasks")
                raise RuntimeError(f"Error while tearing down tasks: {errors}")

    def list_tasks(self) -> list[tuple[UnitTask, TaskState]]:
        """List all deployed tasks.

        Returns:
            A list of tuples containing the task and its state.
        """
        return [(task, self.task_states[task_id]) for task_id, task in self.tasks.items()]

    async def invoke_tasks(self, dispatch: TaskGraphDispatch) -> list[Any]:
        """Invoke the given tasks.

        Before invocation, this method ensures that all tasks part of the invocation
        are deployed and ready to be invoked. It is ensured that the number of outputs
        returned by the task dispatcher matches the number of invocations.

        Args:
            dispatch: The dispatch object containing the tasks to invoke.

        Returns:
            The outputs of all tasks.
        """
        logger.info("Invoking tasks: %s", dispatch)

        # Check if all tasks are deployed
        running_task_ids: list[str] = []
        async with self.task_lock:
            for invocation in dispatch.invocations:
                for task_id, task in self.tasks.items():
                    if task.is_equivalent_to(invocation.task):
                        match self.task_states[task_id]:
                            case TaskState.READY:
                                running_task_ids.append(task_id)
                                break
                            case TaskState.DEPLOYING:
                                raise ValueError(f"Task {invocation.task} is being deployed")
                            case TaskState.TEARING_DOWN:
                                raise ValueError(f"Task {invocation.task} is being torn down")
                else:
                    raise KeyError(f"Task {invocation.task} is not deployed")
            assert len(running_task_ids) == len(dispatch.invocations)

        # Dispatch to the Task Dispatcher
        async with httpx.AsyncClient(timeout=60.0) as client:
            invocation_task = asyncio.create_task(
                client.post(K8S_TASK_DISPATCHER_HTTP_URL + "/task", json=dispatch.model_dump())
            )
            # Store the invocation task under the task IDs of all running tasks.
            # If any of the unit tasks are unregistered, the whole thing will be cancelled.
            for task_id in running_task_ids:
                self.task_invocation_tasks[task_id].append(invocation_task)
            try:
                response = await invocation_task
                response.raise_for_status()
            except asyncio.CancelledError:
                logger.info("Invocation task was cancelled: %s", dispatch)
                raise RuntimeError(
                    "Invocation task was cancelled. This is likely because one or more "
                    "constituent unit tasks were unregistered.",
                ) from None
            except httpx.RequestError as e:
                logger.error("Error while invoking tasks: %s", e)
                raise RuntimeError(f"Error while invoking tasks: {e}") from e
            finally:
                # Remove the invocation task from all task IDs
                for task_id in running_task_ids:
                    self.task_invocation_tasks[task_id].remove(invocation_task)

        output = response.json()
        if not isinstance(output, list):
            raise RuntimeError(f"Invalid response from task dispatcher: {output}")
        if len(output) != len(dispatch.invocations):
            raise RuntimeError(
                f"Invalid response from task dispatcher: {output} (expected {len(dispatch.invocations)} outputs)"
            )
        return output

    async def shutdown(self) -> None:
        """Shutdown the task manager."""
        logger.info("Shutting down the Gateway task manager")

        # Close the gRPC channel to the resource manager
        await self.resource_manager_channel.close()

        logger.info("Gateway task manager has been shut down")
