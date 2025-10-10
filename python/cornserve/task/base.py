"""Base class for tasks."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Generator, Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Self, TypeVar, final

import httpx
from opentelemetry import trace
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator, ValidationError

from cornserve.constants import K8S_GATEWAY_SERVICE_HTTP_URL
from cornserve.logging import get_logger
from cornserve.services.pb.common_pb2 import UnitTask as UnitTaskProto
from cornserve.services.task_registry.task_class_registry import TASK_CLASS_REGISTRY
from cornserve.services.task_registry.descriptor_registry import DESCRIPTOR_REGISTRY

if TYPE_CHECKING:
    from cornserve.services.gateway.task_manager import TaskManager
    from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

TASK_TIMEOUT = 300

# Global shared HTTP client used when we are outside of the Gateway service.
CLIENT = httpx.AsyncClient(timeout=TASK_TIMEOUT)

# This context variable is set inside the top-level task's `__call__` method
# just before creating an `asyncio.Task` (`_call_impl`) to run the task.
# All internal task invocations done by the top-level task will be recorded
# in a single task context object.
task_context: ContextVar[TaskContext] = ContextVar("task_context")

# This context variable is set by the Gateway service when it starts up.
# `TaskContext` requires a reference to the `TaskManager` instance in order to
# dispatch task invocations to the Task Dispatcher.
task_manager_context: ContextVar[TaskManager | None] = ContextVar("task_manager_context", default=None)

# This context variable is used to keep track of the current task being executed.
# THe current task is retrieved to properly register subtasks as they are constructed.
current_task_context: ContextVar[Task | None] = ContextVar("current_task_context", default=None)


class TaskInput(BaseModel):
    """Base class for task input."""


class TaskOutput(BaseModel):
    """Base class for task output."""


InputT = TypeVar("InputT", bound=TaskInput)
OutputT = TypeVar("OutputT", bound=TaskOutput)
TransformT = TypeVar("TransformT", bound=TaskOutput)


class Stream(TaskOutput, Generic[OutputT]):
    """An asynchronous handle through which a task's output can be streamed from.

    This class represents a stream of data (of type `OutputT`).
    Consumers of this stream can treat it as an asynchronous generator.
    """

    # Each line in the stream should be a JSON string that can be parsed into an `OutputT` object.
    async_iterator: AsyncIterator[str] | None = Field(default=None, exclude=True)
    response: httpx.Response | None = Field(default=None, exclude=True)

    _prev_type: type[TaskOutput] | None = None
    _transform_func: Callable[[TaskOutput], OutputT] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @property
    def item_type(self) -> type[OutputT]:
        """The type of items in the stream, parsed from the generic type argument."""
        if not hasattr(self, "_item_type"):
            raise ValueError("Item type is not set. Ensure to call `_item_type` after initialization.")

        metadata = self.__class__.__pydantic_generic_metadata__
        if metadata["origin"] is None:
            raise ValueError("Generic type argument is missing.")

        args = metadata["args"]
        assert len(args) == 1, "There should be exactly one generic type argument."
        item_type = args[0]

        if not issubclass(item_type, TaskOutput):
            raise ValueError(f"Generic type argument {item_type} is not a subclass of TaskOutput.")

        return item_type  # type: ignore

    def transform(self, transform_func: Callable[[OutputT], TransformT]) -> Stream[TransformT]:
        """Transform the stream's output items using a transformation function.

        Args:
            transform_func: A function that takes an `OutputT` item and returns a `TransformT` item.
        """
        new_stream = Stream[TransformT](
            async_iterator=self.async_iterator,
            response=self.response,
        )
        new_stream._prev_type = self.item_type
        new_stream._transform_func = transform_func  # type: ignore
        self.async_iterator = None
        self.response = None
        return new_stream

    @model_validator(mode="after")
    def _item_type(self) -> Self:
        """Parse out the generic output type from the generic type argument."""
        _ = self.item_type  # Access the property to trigger validation
        return self

    async def get_next(self) -> str | None:
        """Get the next line from the stream.

        This is a convenience method to get the next line from the stream.
        It raises `StopAsyncIteration` when there are no more lines in the stream.
        """
        if self.async_iterator is None:
            raise ValueError("Stream generator is not initialized.")

        try:
            return await anext(self.async_iterator)
        except StopAsyncIteration:
            # Close the connection at exit.
            if self.response is not None:
                try:
                    await self.response.aclose()
                except Exception as e:
                    logger.warning("Failed to close stream response: %s", e)

                self.response = None
                self.async_iterator = None
            return None

    def __aiter__(self) -> Self:
        """Return the asynchronous iterator for the stream."""
        return self

    async def __anext__(self) -> OutputT:
        """Get the next item from the stream."""
        item_type = self.item_type
        assert item_type is not None
        if self.async_iterator is None:
            raise ValueError("Stream generator is not initialized.")

        line = await self.get_next()
        if line is None:
            raise StopAsyncIteration

        # If a transformation function is set, first parse the line into the previous type,
        # then apply the transformation function to it.
        if self._transform_func is not None:
            assert self._prev_type is not None, "Previous type must be set for transformation."
            item_prev_type = self._prev_type.model_validate_json(line.strip())
            return self._transform_func(item_prev_type)

        return self.item_type.model_validate_json(line.strip())

    async def aiter_raw(self) -> AsyncGenerator[str]:
        """Asynchronously iterate over the raw output of the stream without parsing."""
        if self.async_iterator is None:
            raise ValueError("Stream generator is not initialized.")

        while True:
            line = await self.get_next()
            if line is None:
                break

            line = line.strip()
            if not line:
                continue

            yield line


class Task(BaseModel, ABC, Generic[InputT, OutputT]):
    """Base class for tasks.

    Attributes:
        id: The ID of the task.
        subtask_attr_names: A list of instance attribute names that hold a `Task`
            instance (e.g., `self.image_encoder` may be an `EncoderTask` and this
            list will contain `"image_encoder"`). This list is automatically
            populated whenever users assign anything that is an instance of `Task`
            as an instance attribute of this task (e.g., `self.image_encoder = ...`).
    """

    id: str = Field(init=False, default_factory=lambda: uuid.uuid4().hex)

    # Automatically populated whenever users assign tasks as instance attributes.
    subtask_attr_names: list[str] = Field(init=False, default_factory=list)

    # Allow extra fields so that users can set subtasks as instance attributes.
    model_config = ConfigDict(extra="allow")

    def post_init(self) -> None:
        """Initialize subtasks.

        Any `Task` instance that is constructed inside this method will be
        automatically registered as a subtask of this task.
        """

    def model_post_init(self, context: Any, /) -> None:
        """Called after the model is initialized."""
        # See if the current task context is set.
        # If so, `self` has been constructed as a subtask of that task, so add it as a subtask.
        # Otherwise, `self` is the top-level task and it's not a subtask of any other task.
        current_task = current_task_context.get()
        if current_task is not None:
            attr_name = f"__subtask_{len(current_task.subtask_attr_names)}__"
            setattr(current_task, attr_name, self)
            current_task.subtask_attr_names.append(attr_name)

        # Now we call `post_init` to initialize our own subtasks.
        # We want our subtasks to be added as subtasks to ourself.
        token = current_task_context.set(self)
        try:
            self.post_init()
        finally:
            current_task_context.reset(token)

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

    async def __call__(self, task_input: InputT) -> OutputT:
        """Invoke the task.

        Args:
            task_input: The input to the task.
        """
        # Initialize a new task context for the top-level task invocation.
        task_context.set(TaskContext())

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


def discover_unit_tasks(tasks: Iterable[Task]) -> list[UnitTask]:
    """Discover unit tasks from an iterable of tasks.

    A task may itself be a unit task, or a composite task that contains unit tasks
    as subtasks inside it.

    Args:
        tasks: An iterable over task objects
    """
    unit_tasks: list[UnitTask] = []
    for task in tasks:
        if isinstance(task, UnitTask):
            unit_tasks.append(task)
        else:
            unit_tasks.extend(discover_unit_tasks(getattr(task, attr) for attr in task.subtask_attr_names))

    return unit_tasks


class UnitTask(Task, Generic[InputT, OutputT]):
    """A task that does not invoke other tasks.

    The unit task is the unit of Task Manager deployment and scaling.
    A unit task is associated with one or more compatible task execution descriptors,
    and one is chosen at init-time based on the `executor_descriptor_name` attribute.
    The same Task Manager is shared by unit tasks when its `root_unit_task_cls`
    attributes are the same, all fields defined by `root_unit_task_cls` are the same,
    and their execution descriptor are the same. Note that child classes of the
    `root_unit_task_cls` are *upcasted* to the `root_unit_task_cls` type and then
    checked for equivalence.

    This class provides a default implementation of the `invoke` method that
    does the following:
    1. If we're executing in recording mode, it calls `make_record_output` to
        construct a task output object whose structure should be the same as what
        the actual task output would be. Task invocation is recoreded in the task
        context object.
    2. Otherwise, if we're executing in replay mode, it directly returns the task
        output saved within the task context object.
    3. Otherwise, it's an error; it raises an `AssertionError`.

    If you want to create a completely new unit task, you should subclass `UnitTask`
    directly. On the other hand, if you want to slightly customize the behavior,
    input/output models, `make_record_output`, etc. of an existing unit task, you
    should subclass that specific unit task subclass, and your subclass's
    `root_unit_task_cls` class attribute will be set to the concrete unit task you
    subclassed.

    For instance, let's say `LLMTask` is a direct subclass of `UnitTask`. `LLMTask`
    will have `root_unit_task_cls` set to `LLMTask` -- itself. All children classes
    of `LLMTask` will have `root_unit_task_cls` set to `LLMTask` as well.

    Attributes:
        root_unit_task_cls: The root unit task class for this task used to (1) find
            task execution descriptors compatible with this task, and (2) determine
            the equivalence of task objects (only the fields in the root unit task
            class are used to determine the equivalence).
        execution_descriptor_name: The name of the task execution descriptor.
            If `None`, the default descriptor registered for the task will be used.
        execution_descriptor: The `TaskExecutionDescriptor` instance for this task.
    """

    root_unit_task_cls: ClassVar[type[UnitTask]]

    execution_descriptor_name: str | None = None

    def __init_subclass__(cls, **kwargs):
        """A hook that runs when a subclass is created.

        This sets the root unit task class for this task.

        For instance, for `LLMTask` that inherits from `UnitTask[LLMInput, LLMOutput]`,
        the inheritence order (`__bases__`) is:
            `LLMTask` -> `UnitTask[LLMInput, LLMOutput]` -> `UnitTask` -> `Task`

        So, we need to look at least two hops to find `UnitTask`.
        """
        super().__init_subclass__(**kwargs)

        def is_proxy_for_unit(base: type) -> bool:
            """True if *base* appears to be the auto‑generated proxy."""
            return UnitTask in getattr(base, "__bases__", ())

        # If any immediate base (or proxies) is `UnitTask`, cls is the root.
        if any(is_proxy_for_unit(b) for b in cls.__bases__):
            cls.root_unit_task_cls = cls
            return

        # Otherwise climb the MRO until you meet that condition
        for anc in cls.mro()[1:]:
            if any(is_proxy_for_unit(b) for b in anc.__bases__):
                cls.root_unit_task_cls = anc
                break
        # Fallback for the intemediate class `UnitTask[SpecificInput, SpecificOutput]`
        # that appears due to generic inheritance.
        else:
            cls.root_unit_task_cls = UnitTask

    @property
    def execution_descriptor(self) -> TaskExecutionDescriptor[Self, InputT, OutputT]:
        """Get the task execution descriptor for this task."""
        descriptor_cls = DESCRIPTOR_REGISTRY.get(self.root_unit_task_cls, self.execution_descriptor_name)
        
        # CRITICAL: Use model_construct instead of standard Pydantic validation.
        # Reason: descriptor types are registered from CR-loaded task classes, while apps
        # load their own identically named task classes. Strict validation compares class
        # identity, so passing an app instance is rejected.
        #
        # EXAMPLE:
        # - Descriptor expects: EncoderTask (id: 104678592029536) from TASK_CLASS_REGISTRY  
        # - App provides:       EncoderTask (id: 104678589216976) from app source
        # - Same name, same structure, but different memory addresses → Validation fails
        #
        # So, we use model_construct to bypass identity checks but preserve data integrity.
        
        return descriptor_cls.model_construct(task=self)

    def is_equivalent_to(self, other: object) -> bool:
        """Check if two unit tasks are equivalent.

        Equivalent unit tasks share the same Task Manager.

        Two unit tasks are equivalent if they have the same root unit task class, same execution descriptor,
        and for all fields defined by the root unit task class, the same values (except for the ID field).
        """
        if not isinstance(other, UnitTask):
            return False

        if self.root_unit_task_cls != other.root_unit_task_cls:
            return False

        if self.execution_descriptor.__class__ != other.execution_descriptor.__class__:
            return False

        # Check if all fields defined by the root unit task class are the same.
        for field_name in self.root_unit_task_cls.model_fields:
            if field_name == "id":
                # Skip the ID field; it can be different for different instances.
                continue
            try:
                if getattr(self, field_name) != getattr(other, field_name):
                    return False
            except AttributeError:
                return False

        return True

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

    def validate_input(self, task_input: InputT) -> None:
        """Validate the task input.

        This hook is called before invoking the task during recording mode.
        """

    @final
    def invoke(self, task_input: InputT) -> OutputT:
        """Invoke the task."""
        ctx = task_context.get()

        if ctx.is_recording:
            self.validate_input(task_input)
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


    @classmethod

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        return f"{self.__class__.__name__.lower()}"


class TaskInvocation(BaseModel, Generic[InputT, OutputT]):
    """An invocation of a task.

    Attributes:
        task: The task that was invoked.
        task_input: The input to the task.
        task_output: The output of the task.
    """

    task: UnitTask[InputT, OutputT]
    task_input: InputT
    task_output: OutputT

    @model_serializer()
    def _serialize(self):
        """Serialize the task invocation."""
        return {
            "class_name": self.task.__class__.__name__,
            "body": {
                "task": self.task.model_dump_json(),
                "task_input": self.task_input.model_dump_json(),
                "task_output": self.task_output.model_dump_json(),
            },
        }

    @model_validator(mode="before")
    @classmethod
    def _deserialize(cls, data: dict[str, Any]):
        """Deserialize the task invocation."""
        # This is likely when we're constructing the object normally by calling the constructor.
        if "class_name" not in data:
            return data

        # Now this is likely when we're deserializing the object from the serialized data.
        task_cls, task_input_cls, task_output_cls = TASK_CLASS_REGISTRY.get_unit_task(data["class_name"])
        task = task_cls.model_validate_json(data["body"]["task"])
        task_input = task_input_cls.model_validate_json(data["body"]["task_input"])
        task_output = task_output_cls.model_validate_json(data["body"]["task_output"])
        return {"task": task, "task_input": task_input, "task_output": task_output}


class TaskGraphDispatch(BaseModel):
    """Payload used for dispatching recorded task invocations.

    Attributes:
        invocations: The recorded task invocations.
    """

    invocations: list[TaskInvocation]
    is_streaming: bool = False

    def model_post_init(self, context: Any, /) -> None:
        """Validate the task graph."""
        if not self.invocations:
            raise ValueError("Task graph must have at least one invocation.")

        # `Stream` may only be used as the output of the last task invocation.
        if any(isinstance(inv.task_output, Stream) for inv in self.invocations[:-1]):
            raise ValueError("Only the last task invocation can have a stream output.")

        # If the last invocation is a stream, the whole task graph is streaming.
        self.is_streaming = isinstance(self.invocations[-1].task_output, Stream)

    @tracer.start_as_current_span("TaskGraphDispatch.dispatch")
    async def dispatch(self, url: str, client: httpx.AsyncClient) -> list[Any]:
        """Dispatch the task graph and wait for the response.

        Returns a list of task outputs. Each task output is deserialized from JSON,
        i.e., a dict that can be used to construct the task output object.
        """
        span = trace.get_current_span()
        span.set_attributes(
            {f"dispatch.invocation.{i}": invocation.model_dump_json() for i, invocation in enumerate(self.invocations)}
        )

        try:
            if self.is_streaming:
                request_obj = client.build_request(method="POST", url=url, json=self.model_dump())
                response = await client.send(request_obj, stream=True)
                response.raise_for_status()
                ait = response.aiter_lines()

                # First line is the task outputs of all invocations.
                try:
                    buffer = await anext(ait)
                except StopAsyncIteration:
                    raise RuntimeError("Received incomplete response from the server.") from None
                dispatch_outputs = json.loads(buffer.strip())

                # Following lines are all streamed outputs of the last invocation.
                dispatch_outputs[-1]["async_iterator"] = ait
                dispatch_outputs[-1]["response"] = response
                stream_cls = self.invocations[-1].task_output.__class__
                assert issubclass(stream_cls, Stream), "Last invocation output must be a Stream."
                _ = stream_cls.model_validate(dispatch_outputs[-1])  # validation
            else:
                response = await client.post(url, json=self.model_dump())
                response.raise_for_status()
                dispatch_outputs = response.json()
        except httpx.RequestError as e:
            logger.exception("Failed to send dispatch request to %s: %s", url, e)
            raise RuntimeError(f"Failed to send dispatch request to {url}.") from e
        except httpx.HTTPStatusError as e:
            logger.exception("Received error response from %s: %s", url, e)
            raise RuntimeError(f"Received error response from {url}.") from e

        if not isinstance(dispatch_outputs, list):
            raise RuntimeError(f"Expected a list of task outputs, but got {type(dispatch_outputs)}.")
        return dispatch_outputs


class TaskContext:
    """Task execution context.

    Attributes:
        is_recording: Whether the context is in recording mode.
        is_replaying: Whether the context is in replay mode.
    """

    def __init__(self) -> None:
        """Initialize the task context."""
        # Task invocations during recording mode.
        self.invocations: list[TaskInvocation] = []

        # Output of each task invocation. Task ID -> task output.
        # Values are lists because the same task could be invoked multiple times
        # within the same context.
        self.task_outputs: dict[str, list[TaskOutput]] = defaultdict(list)

        self.is_recording = False
        self.is_replaying = False

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

    def record_invocation(self, task: UnitTask[InputT, OutputT], task_input: InputT, task_output: OutputT) -> None:
        """Record a task invocation."""
        if not self.is_recording:
            raise RuntimeError("Task invocation can only be recorded in recording mode.")

        if self.is_replaying:
            raise RuntimeError("Cannot record task invocation while replaying.")

        invocation = TaskInvocation(
            task=task.model_copy(deep=True),
            task_input=task_input.model_copy(deep=True),
            task_output=task_output.model_copy(deep=True),
        )
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
        span.set_attributes(
            {
                f"task_context.task.{i}.invocation": invocation.model_dump_json()
                for i, invocation in enumerate(self.invocations)
            }
        )

        # Build the task graph dispatch payload.
        graph_dispatch = TaskGraphDispatch(invocations=self.invocations)

        # Get the Task Manager from the context variable.
        task_manager = task_manager_context.get()

        # This means we're outside of the Gateway service.
        if task_manager is None:
            # Figure out where to dispatch the tasks.
            gateway_url = os.getenv("CORNSERVE_GATEWAY_URL", K8S_GATEWAY_SERVICE_HTTP_URL)

            logger.info("Dispatching tasks to %s/tasks/invoke", gateway_url)

            dispatch_outputs = await graph_dispatch.dispatch(gateway_url + "/tasks/invoke", client=CLIENT)

        # We're inside the Gateway service.
        else:
            logger.info("Dispatching tasks via the Task Manager")

            try:
                dispatch_outputs = await task_manager.invoke_tasks(graph_dispatch)
            except Exception as e:
                logger.exception("Failed to invoke tasks: %s", e)
                raise RuntimeError("Failed to invoke tasks.") from e

        # Parse the response to the right task output type.
        for i, (invocation, output) in enumerate(zip(self.invocations, dispatch_outputs, strict=True)):
            task_output = invocation.task_output.__class__.model_validate(output)
            span.set_attribute(f"task_context.task.{i}.output", task_output.model_dump_json())
            self.task_outputs[invocation.task.id].append(task_output)

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
        except IndexError as e:
            raise RuntimeError(f"Task {task.id} has no more outputs to replay.") from e

        # This should be the right type because it's just being deserialized from the
        # task's actual output.
        return task_output  # type: ignore


class UnitTaskList(BaseModel):
    """A wrapper class for a list of unit tasks.

    This class is added to avoid self-recursion on serialization the `UnitTask` class.
    """

    tasks: list[UnitTask]

    @model_serializer()
    def _serialize(self):
        """Serialize the unittask list."""
        return {
            "_task_list": [
                {
                    "class_name": task.__class__.__name__,
                    "task": task.model_dump_json(),
                }
                for task in self.tasks
            ],
        }

    @model_validator(mode="before")
    @classmethod
    def _deserialize(cls, data: dict[str, Any]):
        """Deserialize the unittask list."""
        if "_task_list" not in data:
            return data
        tasks = []
        for item in data["_task_list"]:
            task_data = item["task"]
            task_class, _, _ = TASK_CLASS_REGISTRY.get_unit_task(item["class_name"])
            task = task_class.model_validate_json(task_data)
            tasks.append(task)
        return {"tasks": tasks}
