"""Gateway FastAPI app definition."""

from __future__ import annotations

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
from opentelemetry import trace
from pydantic import ValidationError

from cornserve.constants import K8S_RESOURCE_MANAGER_GRPC_URL
from cornserve.logging import get_logger
from cornserve.services.gateway.app.manager import AppManager
from cornserve.services.gateway.app.models import AppState
from cornserve.services.gateway.models import (
    AppInvocationRequest,
    AppRegistrationRequest,
    AppRegistrationResponse,
)
from cornserve.services.gateway.session import SessionManager
from cornserve.services.gateway.task_manager import TaskManager
from cornserve.task.base import TaskGraphDispatch, UnitTaskList, task_manager_context

router = APIRouter()
logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


@router.post("/app/register", response_model=AppRegistrationResponse)
async def register_app(request: AppRegistrationRequest, raw_request: Request):
    """Register a new application with the given ID and source code."""
    app_manager: AppManager = raw_request.app.state.app_manager

    try:
        app_id, task_names = await app_manager.register_app(request.source_code)
        return AppRegistrationResponse(app_id=app_id, task_names=task_names)
    except ValueError as e:
        logger.info("Error while initiating app registration: %s", e)
        return Response(status_code=status.HTTP_400_BAD_REQUEST, content=str(e))
    except Exception as e:
        logger.exception("Unexpected error while registering app")
        return Response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=str(e),
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


@router.get("/app/status/{app_id}")
async def get_app_status(app_id: str, raw_request: Request):
    """Get the registration status of an application."""
    app_manager: AppManager = raw_request.app.state.app_manager
    span = trace.get_current_span()
    span.set_attribute("gateway.get_app_status.app_id", app_id)

    try:
        app_state = await app_manager.get_app_status(app_id)
        if app_state is None:
            return Response(status_code=status.HTTP_404_NOT_FOUND, content=f"App ID '{app_id}' not found")
        # Return the app_id and the string value of the AppState enum
        return {"app_id": app_id, "status": app_state.value}
    except Exception as e:
        logger.exception("Unexpected error while getting app status for %s", app_id)
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
