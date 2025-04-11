"""Base class for tasks."""

from __future__ import annotations

import asyncio
import inspect
import uuid
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, override

from pydantic import BaseModel, ConfigDict, Field

from cornserve.task.context import TaskContext, task_context


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
        """Ensure the child classes enforce the Task contract.

        1. `def invoke(self, task_input: TaskInput) -> TaskOutput` should be defined.
        2. Class variable `input_cls` should be a subclass of `TaskInput`.
        3. Class variable `output_cls` should be a subclass of `TaskOutput`.

        TODO: Unit tests.
        """
        super().__init_subclass__(**kwargs)

        # `invoke` should a sync function.
        if not inspect.isfunction(cls.invoke):
            raise TypeError(f"{cls.__name__}.invoke should be a function")

        if inspect.iscoroutinefunction(cls.invoke):
            raise TypeError(f"{cls.__name__}.invoke should not be an async function")

        # `invoke` should take exactly two parameters: self and the task input.
        sig = inspect.signature(cls.invoke)
        parameters = list(sig.parameters.values())

        if len(parameters) != 2:
            raise TypeError(f"{cls.__name__}.invoke should take `self` and one parameter")

        if parameters[0].name != "self":
            raise TypeError(f"{cls.__name__}.invoke should take self as the first parameter")

        if not issubclass(parameters[1].annotation, TaskInput):
            raise TypeError(f"{cls.__name__}.invoke should take a subclass of TaskInput as the second parameter")

        cls.input_cls = parameters[1].annotation

        # `invoke` should return a subclass of `TaskOutput`.
        if not issubclass(sig.return_annotation, TaskOutput):
            raise TypeError(f"{cls.__name__}.invoke should return a subclass of TaskOutput")

        cls.output_cls = sig.return_annotation

        # Check `input_cls` and `output_cls` class variables.
        try:
            if not issubclass(cls.input_cls, TaskInput):
                raise TypeError(f"{cls.__name__}.input_cls should be a subclass of TaskInput")
        except AttributeError:
            raise TypeError(f"{cls.__name__}.input_cls is not defined") from None

        try:
            if not issubclass(cls.output_cls, TaskOutput):
                raise TypeError(f"{cls.__name__}.output_cls should be a subclass of TaskOutput")
        except AttributeError:
            raise TypeError(f"{cls.__name__}.output_cls is not defined") from None

    def __setattr__(self, name: str, value: Any, /) -> None:
        if isinstance(value, Task):
            self.subtasks.append(value)
        return super().__setattr__(name, value)

    async def __call__(self, task_input: InputT) -> OutputT:
        """Invoke the task.

        Args:
            task_input: The input to the task.
        """
        if not isinstance(task_input, self.input_cls):
            raise TypeError(
                f"{self.__class__.__name__}.__call__ should take a subclass of {self.input_cls}",
            )

        # Initialize a new task context for the top-level task invocation.
        token = task_context.set(TaskContext(task_id=self.id))

        try:
            output = await asyncio.create_task(self._call_impl(task_input))

            if not isinstance(output, self.output_cls):
                raise TypeError(
                    f"{self.__class__.__name__}.__call__ should return a subclass of {self.output_cls}",
                )

            return output
        finally:
            # Clear the task context.
            task_context.reset(token)

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

    @override
    def invoke(self, task_input: InputT) -> OutputT:
        """Invoke the task."""
        # Ensure that the task input is of the correct type.
        if not isinstance(task_input, self.input_cls):
            raise TypeError(
                f"{self.__class__.__name__}.invoke should take a subclass of {self.input_cls}",
            )

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

            # Ensure that the task output is of the correct type.
            if not isinstance(task_output, self.output_cls):
                raise RuntimeError(f"Task {self.id} replayed output is not of type {self.output_cls}.")
            return task_output  # type: ignore

        raise AssertionError("Task context is neither in recording nor replay mode.")
