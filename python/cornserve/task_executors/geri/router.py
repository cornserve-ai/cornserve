"""Geri FastAPI app definition."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, FastAPI, Request, Response, status
from opentelemetry import trace

from cornserve.logging import get_logger
from cornserve.task_executors.geri.api import GenerationRequest, GenerationResponse, Status
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


@router.post("/generate")
async def generate(
    request: GenerationRequest,
    raw_request: Request,
    raw_response: Response,
) -> GenerationResponse:
    """Handler for generation requests."""
    engine_client: EngineClient = raw_request.app.state.engine_client

    logger.info("Received generation request: %s", request)

    try:
        request_id = uuid.uuid4().hex
        trace.get_current_span().set_attribute("request_id", request_id)
        response = await engine_client.generate(request_id, request)

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
        logger.exception("Generation request failed: %s", str(e))
        raw_response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return GenerationResponse(status=Status.ERROR, error_message=f"Generation failed: {str(e)}")


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
