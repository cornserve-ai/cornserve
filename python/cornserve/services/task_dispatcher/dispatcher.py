"""Task Dispatcher."""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import grpc
import httpx
from opentelemetry import trace
from pydantic import BaseModel

from cornserve.logging import get_logger
from cornserve.services.pb import task_manager_pb2, task_manager_pb2_grpc
from cornserve.task.base import TaskInvocation, TaskOutput, UnitTask
from cornserve.task.forward import DataForward

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class TaskInfo:
    """Stores all task-related information.

    Attributes:
        task: The unit task object.
        task_manager_url: The URL to the task manager.
        task_manager_channel: The gRPC channel to the task manager.
        task_manager_stub: The gRPC stub to the task manager.
    """

    def __init__(self, task: UnitTask, task_manager_url: str) -> None:
        """Initialize the TaskInfo object."""
        self.task = task
        self.task_manager_url = task_manager_url
        self.task_manager_channel = grpc.aio.insecure_channel(task_manager_url)
        self.task_manager_stub = task_manager_pb2_grpc.TaskManagerStub(self.task_manager_channel)


@dataclass
class UnitTaskExecution:
    """Execution information for a single unit task invocation.

    Attributes:
        invocation: The task invocation object.
        executor_url: The URL to the task executor.
        executor_sidecar_ranks: The sidecar ranks for the task executor.
    """

    invocation: TaskInvocation
    executor_url: str
    executor_sidecar_ranks: list[int]


def iter_data_forwards(obj: object) -> Iterator[DataForward]:
    """Recursively find and iterate through all DataForward objects.

    This method knows how to flatten list, tuple, dict, and nested BaseModel objects.

    Args:
        obj: The object to search for DataForward objects.

    Yields:
        All DataForward objects found in the model's field values
    """
    if isinstance(obj, DataForward):
        yield obj
    elif isinstance(obj, list | tuple):
        for item in obj:
            yield from iter_data_forwards(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            yield from iter_data_forwards(item)
    elif isinstance(obj, BaseModel):
        # Recursively search through nested BaseModels. Make sure references to the original
        # `DataForward` objects are yielded, so that external mutations are reflected.
        for name in obj.__class__.model_fields:
            yield from iter_data_forwards(getattr(obj, name))


class TaskDispatcher:
    """Task Dispatcher."""

    def __init__(self) -> None:
        """Initialize the Task Dispatcher."""
        self.task_lock = asyncio.Lock()
        self.task_infos: dict[str, TaskInfo] = {}

        self.ongoing_task_lock = asyncio.Lock()
        self.ongoing_invokes: dict[str, list[asyncio.Task]] = defaultdict(list)

    async def notify_task_deployment(self, task: UnitTask, task_manager_url: str) -> None:
        """Register a newly deployed task and its task manager with the dispatcher."""
        async with self.task_lock:
            self.task_infos[task.id] = TaskInfo(task, task_manager_url)

        logger.info(
            "Registered new task %s(%s) with task manager URL %s",
            task.__class__.__name__,
            task,
            task_manager_url,
        )

    async def notify_task_teardown(self, task: UnitTask) -> None:
        """Remove a task that has been torn down.

        This will cancel all ongoing invokes for the task.
        """
        async with self.task_lock:
            if task.id not in self.task_infos:
                raise ValueError(f"Task {task} not found in task dispatcher.")

            task_info = self.task_infos.pop(task.id)

        # Cancel all ongoing invokes for the task
        async with self.ongoing_task_lock:
            for invoke_task in self.ongoing_invokes.pop(task.id, []):
                invoke_task.cancel()

        # Close the gRPC channel to the task manager
        await task_info.task_manager_channel.close()

        logger.info("Removed task %s from task dispatcher", task)

    async def shutdown(self) -> None:
        """Shutdown the Task Dispatcher."""
        coros = []
        for task_info in self.task_infos.values():
            coros.append(task_info.task_manager_channel.close())

        results = await asyncio.gather(*coros, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                logger.error("Error occured while shutting down task dispatcher: %s", result)

        logger.info("Task dispatcher shutdown complete")

    @tracer.start_as_current_span("TaskDispatcher.invoke")
    async def invoke(self, invocations: list[TaskInvocation]) -> list[Any]:
        """Dispatch a graph of task invocations to task managers.

        This method:
        1. Gets routes for all tasks from their task managers
        2. Collects and connects DataForward objects from inputs/outputs
        3. Executes tasks concurrently and collects responses
        4. Transforms executor responses back to task outputs
        """
        span = trace.get_current_span()
        span.set_attributes(
            {
                f"task_dispatcher.invoke.invocations.{i}": invocation.model_dump_json()
                for i, invocation in enumerate(invocations)
            }
        )

        # Check if all tasks are registered with the dispatcher
        task_infos: list[TaskInfo] = []
        async with self.task_lock:
            for invocation in invocations:
                for task_info in self.task_infos.values():
                    if task_info.task.is_equivalent_to(invocation.task):
                        task_infos.append(task_info)
                        break
                else:
                    logger.error("Task not found for invocation %s", invocation)
                    raise ValueError(f"Task {invocation.task} not found in task dispatcher.")
        assert len(task_infos) == len(invocations), "Task info count mismatch"

        # Get task executor routes and sidecar ranks
        task_executions: list[UnitTaskExecution] = []
        get_route_coros: list[asyncio.Future[task_manager_pb2.GetRouteResponse]] = []
        request_id = uuid.uuid4().hex
        for task_info in task_infos:
            get_route_coros.append(
                task_info.task_manager_stub.GetRoute(task_manager_pb2.GetRouteRequest(request_id=request_id))
            )
        for invocation, route_response in zip(invocations, await asyncio.gather(*get_route_coros), strict=True):
            task_executions.append(
                UnitTaskExecution(
                    invocation=invocation,
                    executor_url=route_response.task_executor_url,
                    executor_sidecar_ranks=list(route_response.sidecar_ranks),
                )
            )

        # TODO: Build an actual graph, submit to a centralized asyncio.Task that does scheduling.

        # Dig up `DataForward` objects and connect producers and consumers.
        #
        # Example: Two encoders embed two images, and both are sent to two separate LLMs.
        #
        #                   task_input           task_output
        #    Encoder1                         DataForward(id=1)
        #    Encoder2                         DataForward(id=2)
        #        LLM1  DataForward(id=1,2)
        #        LLM2  DataForward(id=1,2)
        #
        # We iterate through `DataForward` objects in the order of invocations, and within each invocation,
        # those in the input and then those in the output. Ultimately, the source and destination sidecar
        # ranks of connected (i.e., same ID) `DataForward` objects should be identical.
        # When we see a `DataForward` object in the output, it is a producer, and from the invocation's
        # task executor routing result, we can figure out its source sidecar ranks.
        # When we see a `DataForward` object in the input, it is a consumer, and it *must* have been
        # previously encountered in the output of a previous invocation. From the invocation's task executor
        # routing result, we can figure out its destination sidecar ranks.
        # Note that when we inplace set the sidecar ranks in `DataForward` objects, we are doing so on
        # references to the original `DataForward` objects in the task input and output.
        producer_forwards: dict[str, DataForward] = {}
        for execution in task_executions:
            # Iterate recursively over all `DataForward` objects in the task input.
            # Encountered `DataForward` objects are consumers, which should have been encountered
            # previously in earlier task invocations. If not, it's an error.
            for consumer_forward in iter_data_forwards(execution.invocation.task_input):
                try:
                    producer_forward = producer_forwards[consumer_forward.id]
                except KeyError as e:
                    raise ValueError(
                        f"Consumer `DataForward[{consumer_forward.data_type}](id={consumer_forward.id})` in the "
                        f"input of invocation {execution.invocation} not found in previous task invocations."
                    ) from e
                assert producer_forward.src_sidecar_ranks is not None
                consumer_forward.src_sidecar_ranks = producer_forward.src_sidecar_ranks
                if producer_forward.dst_sidecar_ranks is None:
                    producer_forward.dst_sidecar_ranks = []
                producer_forward.dst_sidecar_ranks.append(execution.executor_sidecar_ranks)
                consumer_forward.dst_sidecar_ranks = producer_forward.dst_sidecar_ranks

            # Iterate recursively over all `DataForward` objects in the task output.
            # Encountered `DataForward` objects are producers, which we save in `data_forwards`.
            for producer_forward in iter_data_forwards(execution.invocation.task_output):
                producer_forward.src_sidecar_ranks = execution.executor_sidecar_ranks
                producer_forwards[producer_forward.id] = producer_forward

        logger.info("Connected all DataForward objects in task invocations: %s", invocations)

        # Verify whether all `DataForward` objects are properly connected
        for data_forward in producer_forwards.values():
            assert data_forward.src_sidecar_ranks is not None
            assert data_forward.dst_sidecar_ranks is not None

        # Dispatch all task invocations to task executors
        dispatch_coros: list[asyncio.Task[Any]] = []
        client = httpx.AsyncClient(timeout=180.0)
        try:
            async with asyncio.TaskGroup() as tg:
                for execution in task_executions:
                    # `TaskInput` -> JSON request to task executor
                    request = execution.invocation.task.execution_descriptor.to_request(
                        task_input=execution.invocation.task_input,
                        task_output=execution.invocation.task_output,
                    )
                    dispatch_coros.append(tg.create_task(self._execute_unit_task(client, execution, request)))
        except* Exception as e:
            logger.exception("Error while invoking task: %s", e.exceptions)
            raise RuntimeError(f"Task invocation failed: {e.exceptions}") from e
        finally:
            await client.aclose()

        # Collect responses from task executors
        return [task.result() for task in dispatch_coros]

    async def _execute_unit_task(
        self, client: httpx.AsyncClient, execution: UnitTaskExecution, request: dict[str, Any]
    ) -> Any:
        """Execute a single task by sending request to executor and processing response."""
        url = execution.invocation.task.execution_descriptor.get_api_url(execution.executor_url)
        logger.info(
            "Invoking task %s with request: %s",
            execution.invocation.task.__class__.__name__,
            request,
        )
        try:
            response = await client.post(url=url, json=request)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.exception("Error while invoking task")
            raise RuntimeError(
                f"HTTP request failed with code {e.response.status_code}: {e.response}",
            ) from e
        except Exception as e:
            logger.exception("Error while invoking task")
            raise RuntimeError(f"HTTP request failed: {e}") from e

        logger.info(
            "Task %s response: %s",
            execution.invocation.task.__class__.__name__,
            response.content.decode(),
        )

        # JSON response from task executor -> `TaskOutput` -> dump
        task_output: TaskOutput = execution.invocation.task.execution_descriptor.from_response(
            task_output=execution.invocation.task_output,
            response=response.json(),
        )
        return task_output.model_dump()
