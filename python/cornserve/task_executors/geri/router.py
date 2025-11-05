"""Geri FastAPI app definition."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from opentelemetry import trace

from cornserve.logging import get_logger
from cornserve.task_executors.geri.api import (
    AudioGeriRequest,
    BatchGeriResponse,
    ImageGeriRequest,
    Status,
)
from cornserve.task_executors.geri.config import GeriConfig
from cornserve.task_executors.geri.engine.client import EngineClient

router = APIRouter()
logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)


@router.get("/health")
async def health_check(request: Request) -> Response:
    """Check whether the router and the engine are alive."""
    return Response(status_code=status.HTTP_200_OK)


@router.get("/info")
async def info(raw_request: Request) -> GeriConfig:
    """Return Geri's configuration information."""
    return raw_request.app.state.config


@router.post("/image/generate")
async def generate_image(
    request: ImageGeriRequest,
    raw_request: Request,
    raw_response: Response,
) -> BatchGeriResponse:
    """Handler for generation requests."""
    engine_client: EngineClient = raw_request.app.state.engine_client

    logger.info("Received image generation request: %s", request)

    try:
        request_id = uuid.uuid4().hex
        trace.get_current_span().set_attribute("request_id", request_id)
        response = await engine_client.generate_batch(request_id, request)

        # Set appropriate HTTP status code
        match response.status:
            case Status.SUCCESS:
                raw_response.status_code = status.HTTP_200_OK
            case Status.ERROR:
                raw_response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            case _:
                logger.error("Unexpected status: %s", response.status)
                raw_response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

        return response

    except Exception as e:
        logger.exception("Image generation request failed: %s", str(e))
        raw_response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return BatchGeriResponse(status=Status.ERROR, error_message=f"Generation failed: {str(e)}")


@router.post("/audio/generate")
async def generate_audio(
    request: AudioGeriRequest,
    raw_request: Request,
    raw_response: Response,
) -> StreamingResponse:
    """Handler for audio generation requests, where outputs are streamed."""
    engine_client: EngineClient = raw_request.app.state.engine_client

    logger.info("Received streaming audio generation request: %s", request)

    try:
        request_id = uuid.uuid4().hex
        trace.get_current_span().set_attribute("request_id", request_id)

        # Gets an async generator that returns wav byte chunks as they become ready
        stream_consumer = engine_client.generate_streaming(request_id, request)
        return StreamingResponse(stream_consumer, media_type="text/event-stream")

    except Exception as e:
        logger.exception("Audio generation request failed: %s", str(e))
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Generation failed: {str(e)}") from e


def init_app_state(app: FastAPI, config: GeriConfig) -> None:
    """Initialize the app state with the configuration and engine client."""
    app.state.config = config
    app.state.engine_client = EngineClient(config)


def create_app(config: GeriConfig) -> FastAPI:
    """Create a FastAPI app with the given configuration."""
    app = FastAPI()
    app.include_router(router)
    init_app_state(app, config)
    return app
