"""Task Dispatcher."""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from typing import Any

import grpc
import httpx
from opentelemetry import trace

from cornserve.logging import get_logger
from cornserve.services.pb import task_manager_pb2, task_manager_pb2_grpc
from cornserve.services.task_manager.models import TaskManagerType
from cornserve.task.base import TaskInvocation, UnitTask

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


class TaskDispatcher:
    """Task Dispatcher."""

    def __init__(self) -> None:
        """Initialize the Task Dispatcher."""
        self.task_lock = asyncio.Lock()
        self.tasks: dict[str, TaskInfo] = {}

        self.ongoing_task_lock = asyncio.Lock()
        self.ongoing_invokes: dict[str, list[asyncio.Task]] = defaultdict(list)

    async def notify_task_deployment(self, task: UnitTask, task_manager_url: str) -> None:
        """Register a newly deployed task and its task manager with the dispatcher."""
        async with self.task_lock:
            self.tasks[task.id] = TaskInfo(task, task_manager_url)

        logger.info("Registered new task %s with task manager URL %s", task, task_manager_url)

    async def notify_task_teardown(self, task: UnitTask) -> None:
        """Remove a task that has been torn down.

        This will cancel all ongoing invokes for the task.
        """
        async with self.task_lock:
            if task.id not in self.tasks:
                raise ValueError(f"Task {task} not found in task dispatcher.")

            task_info = self.tasks.pop(task.id)

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
        for task_info in self.tasks.values():
            coros.append(task_info.task_manager_channel.close())

        results = await asyncio.gather(*coros, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                logger.error("Error occured while shutting down task dispatcher: %s", result)

        logger.info("Task dispatcher shutdown complete")

    @tracer.start_as_current_span("TaskDispatcher.invoke")
    async def invoke(self, invocations: list[TaskInvocation]) -> list[Any]:
        """Diaptch a graph of task invocations to task managers."""

    async def old_invoke(self, app_id: str, task_id: str, request_id: str, data: str) -> Any:
        """Invoke a task with the given request data."""
        span = trace.get_current_span()
        span.set_attribute("task_dispatcher.invoke.app_id", app_id)
        span.set_attribute("task_dispatcher.invoke.task_id", task_id)
        async with self.app_lock:
            try:
                task_infos = self.app_task_info[app_id]
            except KeyError as e:
                raise ValueError(f"App ID {app_id} not found in task dispatcher.") from e

            try:
                task_info = task_infos[task_id]
            except KeyError as e:
                raise ValueError(f"Task ID {task_id} not found for app {app_id}.") from e

        # Spawn invoke task and add it to the list of ongoing invokes for the app.
        invoke_task = asyncio.create_task(self._do_invoke(app_id, task_info, request_id, data))
        async with self.ongoing_task_lock:
            self.ongoing_invokes[app_id].append(invoke_task)

        try:
            return await invoke_task
        except asyncio.CancelledError as e:
            raise ValueError("Task invoke cancelled. The app is likely shutting down.") from e
        finally:
            async with self.ongoing_task_lock:
                self.ongoing_invokes[app_id].remove(invoke_task)

    async def _do_invoke(
        self,
        app_id: str,
        task_info: TaskInfo,
        request_id: str,
        data: str,
    ) -> Any:
        """Do the actual invocation of the task."""
        # Reformat the request data depending on the type of the task.
        # For the LLM task, if the reqeust has multimodal items in it,
        # break the reqeust into an Eric request and a vLLM request.
        # For multimodal data items, a unique data ID is generated.
        # Data IDs are passed to Eric (as part of the embedding request)
        # and to vLLM (as a key-value pair in the data URI).
        span = trace.get_current_span()
        if isinstance(task_info.task, LLMTask):
            invoke_input = LLMTask._InvokeInput.model_validate_json(data)
            if not invoke_input.multimodal_data:
                invoke_input.multimodal_data = None
            encoder_request: dict | None = None
            embedding_data: list[tuple[str, str, str]] = []

            # LLM text generation request.
            for task_manager in task_info.task_managers:
                if task_manager.type == TaskManagerType.LLM:
                    llm_stub = task_manager.stub
                    break
            else:
                raise RuntimeError(
                    f"LLM task manager not found for app {app_id} and task {task_info.id}.",
                )
            span.add_event("llm_get_route.start")
            llm_route: task_manager_pb2.GetRouteResponse = await llm_stub.GetRoute(
                task_manager_pb2.GetRouteRequest(
                    app_id=app_id,
                    request_id=request_id,
                    routing_hint=None,
                ),
            )
            span.add_event("llm_get_route.done")

            async with httpx.AsyncClient(timeout=60.0) as http_client:
                http_tasks: list[asyncio.Task[httpx.Response]] = []

                # Multimodal embedding request, if the task has multimodal data.
                # Construct and send out.
                if invoke_input.multimodal_data is not None:
                    # Send routing request to the encoder task manager.
                    for task_manager in task_info.task_managers:
                        if task_manager.type == TaskManagerType.ENCODER:
                            encoder_stub = task_manager.stub
                            break
                    else:
                        raise RuntimeError(
                            f"Encoder task manager not found for app {app_id} and task {task_info.id}.",
                        )
                    span.add_event("encoder_get_route.start")
                    encoder_route: task_manager_pb2.GetRouteResponse = await encoder_stub.GetRoute(
                        task_manager_pb2.GetRouteRequest(
                            app_id=app_id,
                            request_id=request_id,
                            routing_hint=None,
                        ),
                    )
                    span.add_event("encoder_get_route.done")

                    for modality, url in invoke_input.multimodal_data:
                        embedding_data.append((uuid.uuid4().hex, modality, url))

                    encoder_request = dict(
                        id=request_id,
                        receiver_sidecar_ranks=list(llm_route.sidecar_ranks),
                        data=[
                            dict(id=data_id, modality=modality, url=url) for data_id, modality, url in embedding_data
                        ],
                    )

                    http_tasks.append(
                        asyncio.create_task(
                            http_client.post(url=f"{encoder_route.task_executor_url}/embeddings", json=encoder_request),
                        )
                    )

                # Construct and send out LLM text generation request.
                # The LLM request is in the form of OpenAI Chat Completions API.
                multimodal_messages = []
                for multimodal_item in embedding_data:
                    data_id, modality, url = multimodal_item
                    data_uri = f"data:{modality}/uuid;data_id={data_id};url={url},"
                    multimodal_messages.append({"type": f"{modality}_url", f"{modality}_url": {"url": data_uri}})

                llm_request = dict(
                    model=task_info.task.model_id,
                    messages=[
                        dict(
                            role="user",
                            content=[dict(type="text", text=invoke_input.prompt), *multimodal_messages],
                        ),
                    ],
                    max_completion_tokens=512,
                    request_id=request_id,
                )

                http_tasks.append(
                    asyncio.create_task(
                        http_client.post(url=f"{llm_route.task_executor_url}/v1/chat/completions", json=llm_request),
                    )
                )

                # Wait for all HTTP tasks to complete.
                try:
                    responses = await asyncio.gather(*http_tasks)
                    for response in responses:
                        response.raise_for_status()
                except httpx.HTTPStatusError as e:
                    logger.exception("Error while invoking task")
                    raise RuntimeError(
                        f"HTTP request failed with code {e.response.status_code}: {e.response}",
                    ) from e
                except Exception as e:
                    logger.exception("Error while invoking task")
                    raise RuntimeError(f"HTTP request failed: {e}") from e

                # Last one is the LLM response, which is returned.
                response = responses[-1].json()
                logger.info("LLM response: %s", response)

                final_response = response["choices"][0]["message"]["content"]
                if not isinstance(final_response, task_info.task._InvokeOutput):
                    raise RuntimeError(
                        f"LLM response is not of type {task_info.task._InvokeOutput}: {final_response}",
                    )
                return final_response

        else:
            raise ValueError(f"Unknown task type: {type(task_info.task)}")
