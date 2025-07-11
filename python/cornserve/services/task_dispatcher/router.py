"""Task Dispatcher REST API server."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, FastAPI, Request, Response, status
from fastapi.responses import StreamingResponse
from opentelemetry import trace

from cornserve.logging import get_logger
from cornserve.services.task_dispatcher.dispatcher import TaskDispatcher
from cornserve.task.base import Stream, TaskGraphDispatch

router = APIRouter()
logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


@router.post("/task")
async def invoke_task(request: TaskGraphDispatch, raw_request: Request):
    """Invoke a task with the given request data."""
    dispatcher: TaskDispatcher = raw_request.app.state.dispatcher

    logger.info("Task dispatch received: %s", request)

    async def stream_response(results: list) -> AsyncGenerator[str]:
        """Stream the response for a streaming task results."""
        all_outputs = json.dumps(results)
        yield all_outputs + "\n"

        stream = results[-1]
        assert isinstance(stream, Stream), "Last result must be a Stream"
        async for chunk_ in stream.aiter_raw():
            chunk = chunk_.strip()
            if not chunk:
                continue
            yield chunk + "\n"

    try:
        results = await dispatcher.invoke(request.invocations)
    except Exception as e:
        logger.exception("Error while invoking task")
        return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=str(e))

    if request.is_streaming:
        return StreamingResponse(stream_response(results), media_type="text/plain")
    return results


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return Response(status_code=status.HTTP_200_OK)


def init_app_state(app: FastAPI) -> None:
    """Initialize the app state for the Task Dispatcher."""
    app.state.dispatcher = TaskDispatcher()


def create_app() -> FastAPI:
    """Build the FastAPI app for the Task Dispatcher."""
    app = FastAPI(title="Cornserve Task Dispatcher")
    app.include_router(router)
    init_app_state(app)
    return app
