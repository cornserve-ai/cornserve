"""Base class for tasks."""

from __future__ import annotations

import asyncio
import inspect
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Generator, Generic, TypeVar

import httpx
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field

# from cornserve.task.context import TaskContext, task_context
from cornserve.constants import K8S_TASK_DISPATCHER_HTTP_URL
from cornserve.logging import get_logger

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


# This context variable is set inside the top-level task's `__call__` method
# just before creating an `asyncio.Task` (`_call_impl`) to run the task.
# All internal task invocations done by the top-level task will be recorded
# in a single task context object.
task_context: ContextVar[TaskContext] = ContextVar("task_context")


class TaskInput(BaseModel):
    """Base class for task input."""


class TaskOutput(BaseModel):
    """Base class for task output."""


InputT = TypeVar("InputT", bound=TaskInput)
OutputT = TypeVar("OutputT", bound=TaskOutput)


class Task(BaseModel, ABC, Generic[InputT, OutputT]):
    """Base class for tasks.

    Attributes:
        id: The ID of the task.
        subtasks: A list of subtasks that are assigned as instance attributes
            to this task (e.g., `self.image_encoder`). This list is automatically
            populated whenever users assign anything that is an instance of `Task`
            as an instance attribute of this task (e.g., `self.image_encoder = ...`).
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)

    # Automatically populated whenever users assign tasks as instance attributes.
    subtasks: list[Task] = Field(default_factory=list)

    # Allow extra fields so that users can set subtasks as instance attributes.
    model_config = ConfigDict(extra="allow")

    @abstractmethod
    def invoke(self, task_input: InputT) -> OutputT:
        """Invoke the task."""

    def __init_subclass__(cls, **kwargs):
        """Check the invoke method of the subclass."""
        super().__init_subclass__(**kwargs)

        # `invoke` should a sync function.
        if not inspect.isfunction(cls.invoke):
            raise TypeError(f"{cls.__name__}.invoke should be a function")

        if inspect.iscoroutinefunction(cls.invoke):
            raise TypeError(f"{cls.__name__}.invoke should not be an async function")

    def __setattr__(self, name: str, value: Any, /) -> None:
        """Same old setattr but puts tasks in the subtasks list."""
        if isinstance(value, Task):
            self.subtasks.append(value)
        return super().__setattr__(name, value)

    async def __call__(self, task_input: InputT) -> OutputT:
        """Invoke the task.

        Args:
            task_input: The input to the task.
        """
        # Initialize a new task context for the top-level task invocation.
        task_context.set(TaskContext(task_id=self.id))

        return await asyncio.create_task(self._call_impl(task_input))

    async def _call_impl(self, task_input: InputT) -> OutputT:
        """Invoke the task implementation.

        This function is called by the `__call__` method. It is expected to be
        overridden by subclasses to provide the actual task implementation.

        Args:
            task_input: The input to the task.
        """
        # Fetch the task context.
        ctx = task_context.get()

        # Run the invoke method to trace and record task invocations.
        # The `record` context manager will have all task invocations
        # record their invocations within the context.
        with ctx.record():
            _ = self.invoke(task_input)

        # Dispatch all tasks to the Task Dispatcher and wait for their completion.
        await ctx.dispatch_tasks_and_wait()

        # Re-run the invoke method to construct the final result of the task.
        # The `replay` context manager will have all tasks directly use actual task outputs.
        with ctx.replay():
            return self.invoke(task_input)


class UnitTask(Task, Generic[InputT, OutputT]):
    """A task that does not invoke other tasks.

    This class provides a default implementation of the `invoke` method that
    does the following:
    1. If we're executing in recording mode, it calls `make_record_output` to
        construct a task output object whose structure should be the same as what
        the actual task output would be. Task invocation is recoreded in the task
        context object.
    2. Otherwise, if we're executing in replay mode, it directly returns the task
        output saved within the task context object.
    3. Otherwise, it's an error; it raises an `AssertionError`.
    """

    @abstractmethod
    def make_record_output(self, task_input: InputT) -> OutputT:
        """Construct a task output object for recording task invocations.

        Concrete task invocation results are not available during recording mode,
        but semantic information in the task output object is still needed to execute
        the `invoke` method of composite tasks. For instance, an encoder task will
        return a list of embeddings given a list of multimodal data URLs, and the
        length of the embeddings list should match the length of the data URLs list.
        Behaviors like this are expected to be implemented by this method.
        """

    def invoke(self, task_input: InputT) -> OutputT:
        """Invoke the task."""
        ctx = task_context.get()

        if ctx.is_recording:
            task_output = self.make_record_output(task_input)
            ctx.record_invocation(
                task=self,
                task_input=task_input,
                task_output=task_output,
            )
            return task_output

        if ctx.is_replaying:
            task_output = ctx.replay_invocation(self)
            return task_output  # type: ignore

        raise AssertionError("Task context is neither in recording nor replay mode.")


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


class TaskGraphDispatch(BaseModel):
    """Payload used for dispatching recorded task invocations.

    Attributes:
        task_id: The ID of the top-level task.
        invocations: The recorded task invocations.
    """

    task_id: str
    invocations: list[TaskInvocation]


class TaskGraphResponse(BaseModel):
    """Outputs of dispatched and completed tasks.

    Attributes:
        task_outputs: The outputs of the dispatched tasks.
    """

    task_outputs: list[TaskOutput]


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

    def replay_invocation(self, task: Task[InputT, OutputT]) -> OutputT:
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
            # Ensure output type.
            task_output = task_outputs.pop(0)
            expected_output_type = task.__pydantic_generic_metadata__["args"][1]
            if not isinstance(task_output, expected_output_type):
                raise TypeError(
                    f"Task output type mismatch: {type(task_output)} != {expected_output_type}",
                )

            # We just manually checked the type, so we can bypass the type checker here.
            return task_output  # type: ignore
        except IndexError as e:
            raise RuntimeError(f"Task {task.id} has no more outputs to replay.") from e
