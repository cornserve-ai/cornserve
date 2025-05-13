"""Task Dispatcher REST API server."""

from __future__ import annotations

from fastapi import APIRouter, FastAPI, Request, Response, status
from opentelemetry import trace

from cornserve.logging import get_logger
from cornserve.services.task_dispatcher.dispatcher import TaskDispatcher
from cornserve.task.base import TaskGraphDispatch

router = APIRouter()
logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


@router.post("/task")
async def invoke_task(request: TaskGraphDispatch, raw_request: Request):
    """Invoke a task with the given request data."""
    logger.info("Task dispatch received: %s", request)

    dispatcher: TaskDispatcher = raw_request.app.state.dispatcher
    try:
        response = await dispatcher.invoke(request.invocations)
        return response
    except Exception as e:
        logger.exception("Error while invoking task")
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return Response(status_code=status.HTTP_200_OK)


def init_app_state(app: FastAPI) -> None:
    """Initialize the app state for the Task Dispatcher."""
    app.state.dispatcher = TaskDispatcher()


def create_app() -> FastAPI:
    """Build the FastAPI app for the Task Dispatcher."""
    app = FastAPI(title="CornServe Task Dispatcher")
    app.include_router(router)
    init_app_state(app)
    return app
