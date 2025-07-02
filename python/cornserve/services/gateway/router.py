"""Gateway FastAPI app definition."""

from __future__ import annotations

from collections.abc import AsyncGenerator

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
from cornserve.services.gateway.app.models import (
    RegistrationErrorResponse,
    RegistrationFinalResponse,
    RegistrationInitialResponse,
    RegistrationStatusEvent,
)
from cornserve.services.gateway.models import (
    AppInvocationRequest,
    AppRegistrationRequest,
    ScaleTaskRequest,
)
from cornserve.services.gateway.session import SessionManager
from cornserve.services.gateway.task_manager import TaskManager
from cornserve.task.base import TaskGraphDispatch, UnitTaskList, task_manager_context

router = APIRouter()
logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


@router.post("/app/register")
async def register_app(request: AppRegistrationRequest, raw_request: Request):
    """Register a new application with streaming response for deployment progress."""
    app_manager: AppManager = raw_request.app.state.app_manager

    async def generate_registration_stream() -> AsyncGenerator[str, None]:
        """Generate Server-Sent Events (SSE) for the registration process.

        For SSE standards (e.g. the "data: " prefix),
        see https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events
        """
        app_id: str | None = None
        try:
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
            logger.info("Error during app registration for app_id '%s': %s", app_id, e)
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
    try:
        return await app_manager.invoke_app(app_id, request.request_data)
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

    try:
        return await task_manager.invoke_tasks(request)
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


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return Response(status_code=status.HTTP_200_OK)


def init_app_state(app: FastAPI) -> None:
    """Initialize the app state with required components."""
    app.state.task_manager = TaskManager(K8S_RESOURCE_MANAGER_GRPC_URL)
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
