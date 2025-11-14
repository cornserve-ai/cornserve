"""Gateway FastAPI app definition."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncGenerator, AsyncIterator

from fastapi import (
    APIRouter,
    FastAPI,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from opentelemetry import trace
from pydantic import ValidationError

from cornserve.constants import K8S_RESOURCE_MANAGER_GRPC_URL
from cornserve.logging import get_logger
from cornserve.services.gateway.app.manager import AppManager
from cornserve.services.gateway.models import (
    AppInvocationRequest,
    AppRegistrationRequest,
    RegistrationErrorResponse,
    RegistrationFinalResponse,
    RegistrationInitialResponse,
    RegistrationStatusEvent,
    ScaleTaskRequest,
    TasksDeploymentRequest,
)
from cornserve.services.gateway.session import SessionManager
from cornserve.services.gateway.task_manager import TaskManager
from cornserve.services.task_registry import TaskRegistry
from cornserve.task.base import Stream, TaskGraphDispatch, TaskOutput, UnitTaskList, task_manager_context

router = APIRouter()
logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


@router.post("/app/register")
async def register_app(request: AppRegistrationRequest, raw_request: Request):
    """Register a new application with streaming response for deployment progress."""
    app_manager: AppManager = raw_request.app.state.app_manager

    async def generate_registration_stream() -> AsyncGenerator[str]:
        """Generate Server-Sent Events (SSE) for the registration process.

        For SSE standards (e.g. the "data: " prefix),
        see https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events
        """
        app_id: str | None = None
        try:
            # Check: if no task class observed, very likely the tasklib is not deployed yet
            task_registry: TaskRegistry = raw_request.app.state.task_registry
            is_empty = await task_registry.check_emptiness()
            if is_empty:
                raise ValueError(
                    "No task definitions found in the cluster. Did you forget to deploy the tasklib? "
                    "See https://cornserve.ai/getting_started/ for more instructions."
                )

            # Parse and validate the app
            app_id, task_names = await app_manager.validate_and_create_app(request.source_code)

            # Send initial response
            initial_event = RegistrationStatusEvent(
                event=RegistrationInitialResponse(app_id=app_id, task_names=task_names)
            )
            yield f"data: {initial_event.model_dump_json()}\n\n"

            # Now deploy tasks and send final result
            await app_manager.deploy_app_tasks(app_id)
            final_event = RegistrationStatusEvent(
                event=RegistrationFinalResponse(message=f"Successfully deployed {len(task_names)} unit tasks")
            )
            yield f"data: {final_event.model_dump_json()}\n\n"

        except Exception as e:
            logger.exception("Error during app registration for app_id '%s'", app_id)
            error_event = RegistrationStatusEvent(event=RegistrationErrorResponse(message=str(e)))
            yield f"data: {error_event.model_dump_json()}\n\n"

    return StreamingResponse(
        generate_registration_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.post("/app/unregister/{app_id}")
async def unregister_app(app_id: str, raw_request: Request):
    """Unregister the application with the given ID."""
    app_manager: AppManager = raw_request.app.state.app_manager
    span = trace.get_current_span()
    span.set_attribute("gateway.unregister_app.app_id", app_id)

    try:
        await app_manager.unregister_app(app_id)
        return Response(status_code=status.HTTP_200_OK)
    except KeyError as e:
        return Response(status_code=status.HTTP_404_NOT_FOUND, content=str(e))
    except Exception as e:
        logger.exception("Unexpected error while unregistering app")
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=str(e),
        )


@router.post("/app/invoke/{app_id}")
async def invoke_app(app_id: str, request: AppInvocationRequest, raw_request: Request):
    """Invoke a registered application."""
    app_manager: AppManager = raw_request.app.state.app_manager

    span = trace.get_current_span()
    span.set_attribute("gateway.invoke_app.app_id", app_id)
    span.set_attributes(
        {f"gateway.invoke_app.request.{key}": str(value) for key, value in request.request_data.items()},
    )

    async def stream_app_response(
        app_response_iter: AsyncIterator[TaskOutput],
    ) -> AsyncGenerator[str]:
        """Stream the response for a streaming app."""
        async for response_item in app_response_iter:
            response_json = response_item.model_dump_json()
            yield response_json + "\n"

    try:
        response = await app_manager.invoke_app(app_id, request.request_data)

        if isinstance(response, AsyncIterator):
            return StreamingResponse(
                stream_app_response(response),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache"},
            )
        else:
            return response

    except ValidationError as e:
        raise RequestValidationError(errors=e.errors()) from e
    except KeyError as e:
        return Response(status_code=status.HTTP_404_NOT_FOUND, content=str(e))
    except ValueError as e:
        logger.info("Error while running app %s: %s", app_id, e)
        return Response(status_code=status.HTTP_400_BAD_REQUEST, content=str(e))
    except Exception as e:
        logger.exception("Unexpected error while running app %s", app_id)
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=str(e),
        )


@router.get("/app/list")
async def list_apps(raw_request: Request):
    """List all registered applications."""
    app_manager: AppManager = raw_request.app.state.app_manager

    try:
        return await app_manager.list_apps()
    except Exception as e:
        logger.exception("Unexpected error while listing apps")
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=str(e),
        )


@router.websocket("/session")
async def session(socket: WebSocket):
    """WebSocket endpoint for developers to interact with the gateway.

    Within the session, a CornserveClient can deploy tasks, and they
    will be removed when the session ends.
    """
    await socket.accept()
    session_manager: SessionManager = socket.app.state.session_manager
    session_id = await session_manager.create_session()
    try:
        while True:
            request = await socket.receive_json()
            response = await session_manager.handle_request(session_id, request)
            await socket.send_text(response.model_dump_json())
    except WebSocketDisconnect:
        logger.info("Websocket disconnected")
        pass
    except Exception:
        logger.exception("Error handling websocket")
    finally:
        await session_manager.destroy_session(session_id)


@router.post("/task/register")
async def register_task(raw_request: Request):
    """Register a new task and its execution descriptor with the given its source code."""
    raise NotImplementedError("Task registration is not implemented yet.")


@router.post("/task/scale")
async def scale_task(request: ScaleTaskRequest, raw_request: Request):
    """Scale the number of GPUs for a unit task.

    Positive values will scale up, negative values will scale down.
    """
    if request.num_gpus == 0:
        return Response(
            status_code=status.HTTP_400_BAD_REQUEST,
            content="Scaling with 0 GPUs has no effect.",
        )
    task_manager: TaskManager = raw_request.app.state.task_manager

    try:
        await task_manager.scale_unit_task(request.task_id, request.num_gpus)
        return Response(status_code=status.HTTP_200_OK)
    except KeyError as e:
        return Response(status_code=status.HTTP_404_NOT_FOUND, content=str(e))
    except Exception as e:
        logger.exception("Unexpected error while scaling up task")
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=str(e),
        )


@router.get("/tasks/list")
async def list_tasks(raw_request: Request):
    """List all deployed unit tasks."""
    task_manager: TaskManager = raw_request.app.state.task_manager

    try:
        return task_manager.list_tasks()
    except Exception as e:
        logger.exception("Unexpected error while listing tasks")
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=str(e),
        )


@router.post("/tasks/usage")
async def declare_task_usage(request: UnitTaskList, raw_request: Request):
    """Ensure that one or more unit tasks are deployed.

    If a task is already deployed, it will be skipped without error.
    """
    task_manager: TaskManager = raw_request.app.state.task_manager

    try:
        await task_manager.declare_used(request.tasks)
        return Response(status_code=status.HTTP_200_OK)
    except Exception as e:
        logger.exception("Unexpected error while deploying tasks")
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=str(e),
        )


@router.delete("/tasks/usage")
async def declare_unused_tasks(request: UnitTaskList, raw_request: Request):
    """Notify the gateway that one or more unit tasks are no longer in use.

    If a task is not found, it will be skipped without error.
    """
    task_manager: TaskManager = raw_request.app.state.task_manager

    try:
        await task_manager.declare_not_used(request.tasks)
        return Response(status_code=status.HTTP_200_OK)
    except Exception as e:
        logger.exception("Unexpected error while tearing down tasks")
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=str(e),
        )


@router.post("/tasks/invoke")
async def invoke_tasks(request: TaskGraphDispatch, raw_request: Request):
    """Invoke a unit task graph."""
    task_manager: TaskManager = raw_request.app.state.task_manager

    async def stream_response(results: list) -> AsyncGenerator[str | bytes, None]:
        """Stream the response for a streaming task results."""
        *results_with_empty_stream, stream = results
        results_with_empty_stream.append({})
        all_outputs = json.dumps(results_with_empty_stream)
        yield all_outputs + "\n"

        stream_obj = request.invocations[-1].task_output.__class__.model_validate(stream)
        assert isinstance(stream_obj, Stream), "Last result must be a Stream"
        async for chunk in stream_obj.aiter_raw():
            chunk = chunk.strip()
            if not chunk:
                continue
            if isinstance(chunk, bytes):
                yield chunk + b"\n"
            else:
                yield chunk + "\n"

    try:
        results = await task_manager.invoke_tasks(request)
    except KeyError as e:
        return Response(status_code=status.HTTP_404_NOT_FOUND, content=str(e))
    except ValueError as e:
        logger.info("Error while invoking tasks: %s", e)
        return Response(status_code=status.HTTP_400_BAD_REQUEST, content=str(e))
    except Exception as e:
        logger.exception("Unexpected error while invoking tasks")
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=str(e),
        )

    if request.is_streaming:
        return StreamingResponse(
            stream_response(results),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache"},
        )
    return results


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return Response(status_code=status.HTTP_200_OK)


def init_app_state(app: FastAPI) -> None:
    """Initialize the app state with required components."""
    # Create registry for handling unit task instance names
    app.state.task_registry = TaskRegistry()

    # Pass registry to TaskManager for task-based deployment
    app.state.task_manager = TaskManager(K8S_RESOURCE_MANAGER_GRPC_URL, app.state.task_registry)
    app.state.app_manager = AppManager(app.state.task_manager)
    app.state.session_manager = SessionManager(app.state.task_manager)

    # Make the Task Manager available to `cornserve.task.base.TaskContext`
    task_manager_context.set(app.state.task_manager)


def create_app() -> FastAPI:
    """Create a FastAPI app for the Gateway service."""
    app = FastAPI(title="Cornserve Gateway")
    app.include_router(router)
    init_app_state(app)
    return app


@router.post("/deploy-tasks")
async def deploy_tasks(request: TasksDeploymentRequest, raw_request: Request):
    """Deploy tasks (unit or composite) and their descriptors from provided sources."""
    task_registry: TaskRegistry = raw_request.app.state.task_registry
    try:
        # Create task definitions and descriptors concurrently
        async def create_task_definition(spec):
            source = base64.b64decode(spec.source_b64).decode("utf-8")
            try:
                return await task_registry.create_task_definition(
                    name=spec.task_definition_name,
                    task_class_name=spec.task_class_name,
                    module_name=spec.module_name,
                    source_code=source,
                    is_unit_task=spec.is_unit_task,
                )
            except ValueError as e:
                # Ignore duplicate definition errors (idempotent for unit and composite tasks)
                if "already exists" in str(e):
                    return None
                raise

        async def create_execution_descriptor(spec):
            source = base64.b64decode(spec.source_b64).decode("utf-8")
            try:
                return await task_registry.create_execution_descriptor(
                    name=spec.descriptor_definition_name,
                    task_class_name=spec.task_class_name,
                    descriptor_class_name=spec.descriptor_class_name,
                    module_name=spec.module_name,
                    source_code=source,
                    is_default=True,
                )
            except ValueError as e:
                # Ignore duplicate descriptor errors (idempotent)
                if "already exists" in str(e):
                    return None
                raise

        coroutines = [
            *(create_task_definition(spec) for spec in request.task_definitions),
            *(create_execution_descriptor(spec) for spec in request.descriptor_definitions),
        ]

        if coroutines:
            results = await asyncio.gather(*coroutines, return_exceptions=True)
            errors = [e for e in results if isinstance(e, Exception)]
            if errors:
                # Raise if any creation failed
                raise errors[0]

        return {"status": "ok"}
    except Exception as e:
        logger.exception("Failed to deploy tasks")
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=str(e))


@router.post("/purge-tasklib")
async def purge_tasklib(raw_request: Request):
    """Purge tasklib by deleting all TaskDefinition/ExecutionDescriptor CRs.

    Requires the cluster to be idle (no UnitTaskInstance CRs). Each service's TaskRegistry
    watcher will responds to CR DELETED events by purging its local runtime.
    """
    task_registry: TaskRegistry = raw_request.app.state.task_registry
    try:
        idle = await task_registry.check_no_activate_task_instance()
        if not idle:
            return Response(
                status_code=status.HTTP_409_CONFLICT,
                content="Cluster is not idle, cannot purge tasklib.",
            )

        await task_registry.delete_all_task_definitions_and_descriptors()
        return {"status": "ok"}
    except Exception as e:
        logger.exception("Failed to purge tasklib")
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=str(e))
