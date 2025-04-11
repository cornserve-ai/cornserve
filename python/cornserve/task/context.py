from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Generator, Generic

import httpx
from opentelemetry import trace
from pydantic import BaseModel

from cornserve.constants import K8S_TASK_DISPATCHER_HTTP_URL
from cornserve.logging import get_logger
from cornserve.services.task_dispatcher.models import TaskGraphDispatch, TaskGraphResponse, TaskInvocation
from cornserve.task.base import InputT, OutputT, Task, TaskInput, TaskOutput

# This context variable is set inside the top-level task's `__call__` method
# just before creating an `asyncio.Task` (`_call_impl`) to run the task.
# All internal task invocations done by the top-level task will be recorded
# in a single task context object.
task_context: ContextVar[TaskContext] = ContextVar("task_context")

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


class TaskInvocation(BaseModel, Generic[InputT, OutputT]):
    """An invocation of a task.

    Attributes:
        task: The task that was invoked.
        task_input: The input to the task.
        task_output: The output of the task.
    """

    task: Task[InputT, OutputT]
    task_input: InputT
    task_output: OutputT


class TaskContext:
    """Task execution context.

    Attributes:
        is_recording: Whether the context is in recording mode.
        is_replaying: Whether the context is in replay mode.
    """

    def __init__(self, task_id: str) -> None:
        """Initialize the task context.

        Args:
            task_id: The ID of the top-level task.
        """
        self.task_id = task_id

        # Task invocations during recording mode.
        self.invocations: list[TaskInvocation] = []

        # Output of each task invocation. Task ID -> task output.
        # Values are lists because the same task could be invoked multiple times
        # within the same context.
        self.task_outputs: dict[str, list[TaskOutput]] = defaultdict(list)

    @contextmanager
    def record(self) -> Generator[None, None, None]:
        """Set the context mode to record task invocations."""
        if self.is_recording:
            raise RuntimeError("Task context is already in recording mode.")

        if self.is_replaying:
            raise RuntimeError("Cannot enter record mode while replaying.")

        self.is_recording = True

        try:
            yield
        finally:
            self.is_recording = False

    @contextmanager
    def replay(self) -> Generator[None, None, None]:
        """Set the context mode to replay task invocations."""
        if self.is_replaying:
            raise RuntimeError("Task context is already in replay mode.")

        if self.is_recording:
            raise RuntimeError("Cannot enter replay mode while recording.")

        self.is_replaying = True

        try:
            yield
        finally:
            self.is_replaying = False

    def record_invocation(self, task: Task[InputT, OutputT], task_input: InputT, task_output: OutputT) -> None:
        """Record a task invocation."""
        if not self.is_recording:
            raise RuntimeError("Task invocation can only be recorded in recording mode.")

        if self.is_replaying:
            raise RuntimeError("Cannot record task invocation while replaying.")

        invocation = TaskInvocation(task=task, task_input=task_input, task_output=task_output)
        self.invocations.append(invocation)

    @tracer.start_as_current_span("TaskContext.dispatch_tasks_and_wait")
    async def dispatch_tasks_and_wait(self) -> None:
        """Dispatch all recorded tasks and wait for their completion."""
        if self.is_recording:
            raise RuntimeError("Cannot dispatch tasks while recording.")

        if self.is_replaying:
            raise RuntimeError("Cannot dispatch tasks while replaying.")

        if not self.invocations:
            logger.warning("No task invocations were recorded. Finishing dispatch immediately.")
            return

        if self.task_outputs:
            raise RuntimeError("Task outputs already exist. Task contexts are not supposed to be reused.")

        span = trace.get_current_span()
        span.set_attribute("task_context.task_id", self.task_id)
        span.set_attributes(
            {
                f"task_context.task.{i}.invocation": invocation.model_dump_json()
                for i, invocation in enumerate(self.invocations)
            }
        )

        # Dispatch the entire task graph to the Task Dispatcher and wait for results.
        request = TaskGraphDispatch(task_id=self.task_id, invocations=self.invocations)
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    K8S_TASK_DISPATCHER_HTTP_URL + "/task",
                    json=request.model_dump(),
                )
            response.raise_for_status()
        except httpx.RequestError as e:
            logger.exception("Failed to send dispatch request to the Task Dispatcher: %s", e)
            raise RuntimeError("Failed to send dispatch request to the Task Dispatcher.") from e
        except httpx.HTTPStatusError as e:
            logger.exception("Task Dispatcher returned an error: %s", e)
            raise RuntimeError("Task Dispatcher returned an error") from e

        task_outputs = TaskGraphResponse.model_validate(response.json()).task_outputs
        for i, (invocation, output) in enumerate(zip(self.invocations, task_outputs, strict=True)):
            span.set_attribute(f"task_context.task.{i}.output", output.model_dump_json())
            self.task_outputs[invocation.task.id].append(output)

    def replay_invocation(self, task: Task) -> TaskOutput:
        """Replay a task invocation.

        Special handling is done because the same task may be invoked multiple times
        within the same context. Still, during record and replay, those will be
        invoked in the same order.
        """
        if not self.is_replaying:
            raise RuntimeError("Task context is not in replay mode.")

        if self.is_recording:
            raise RuntimeError("Cannot replay task invocation while recording.")

        if not self.task_outputs:
            raise RuntimeError("No task outputs exist.")

        try:
            task_outputs = self.task_outputs[task.id]
        except KeyError as e:
            raise RuntimeError(f"Task {task.id} not found in task outputs.") from e

        try:
            return task_outputs.pop(0)
        except IndexError as e:
            raise RuntimeError(f"Task {task.id} has no more outputs to replay.") from e
