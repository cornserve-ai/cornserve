"""Task Dispatcher REST API server."""

from __future__ import annotations

from fastapi import FastAPI, APIRouter, Request, Response, status

from cornserve.logging import get_logger
from cornserve.services.task_dispatcher.dispatcher import TaskDispatcher
from cornserve.services.task_dispatcher.models import TaskDispatchRequest

router = APIRouter()
logger = get_logger(__name__)


@router.post("/task")
async def invoke_task(request: TaskDispatchRequest, raw_request: Request):
    """Invoke a task with the given request data."""
    dispatcher: TaskDispatcher = raw_request.app.state.dispatcher
    logger.info("Received invoke request for app %s: %s", request.app_id, request.request_data)
    try:
        response = await dispatcher.invoke(
            request.app_id,
            request.task_id,
            request.request_id,
            request.request_data,
        )
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
