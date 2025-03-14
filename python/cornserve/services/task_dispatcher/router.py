"""Task Dispatcher REST API server."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from fastapi import FastAPI, APIRouter, Request, Response, status

from cornserve.logging import get_logger
from cornserve.services.task_dispatcher.dispatcher import TaskDispatcher

router = APIRouter()
logger = get_logger(__name__)


class TaskDispatchRequest(BaseModel):
    """Request for invoking a task.

    Attributes:
        app_id: The unique identifier for the application.
        task_id: The unique identifier for the task.
        request_data: The input data for the task.
    """
    
    app_id: str
    task_id: str
    request_data: dict[str, Any]


@router.post("/task")
async def invoke_task(request: TaskDispatchRequest, raw_request: Request):
    """Invoke a task with the given request data."""
    dispatcher: TaskDispatcher = raw_request.app.state.dispatcher
    try:
        response = await dispatcher.invoke(request.app_id, request.task_id, request.request_data)
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
