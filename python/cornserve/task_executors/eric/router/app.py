"""Eric FastAPI app definition."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, FastAPI, Request, Response, status
from opentelemetry import trace

from cornserve.logging import get_logger
from cornserve.task_executors.eric.api import EmbeddingRequest, EmbeddingResponse, Modality, Status
from cornserve.task_executors.eric.config import EricConfig
from cornserve.task_executors.eric.engine.client import EngineClient
from cornserve.task_executors.eric.models.registry import MODEL_REGISTRY
from cornserve.task_executors.eric.router.processor import Processor

import asyncio

router = APIRouter()
logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)

class QueueCounter:
    """A simple queue counter to keep track of the number of items in the queue."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.count = 0

    async def increment(self) -> None:
        """Increment the count by 1."""
        async with self.lock:
            self.count += 1

    async def decrement(self) -> None:
        """Decrement the count by 1."""
        async with self.lock:
            if self.count > 0:
                self.count -= 1

    async def get_count(self) -> int:
        """Get the current count."""
        async with self.lock:
            return self.count


@router.get("/health")
async def health_check(request: Request) -> Response:
    """Checks whether the router and the engine are alive."""
    engine_client: EngineClient = request.app.state.engine_client
    match engine_client.health_check():
        case True:
            return Response(status_code=status.HTTP_200_OK)
        case False:
            return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


@router.get("/info")
async def info(raw_request: Request) -> EricConfig:
    """Returns Eric's configuration information."""
    return raw_request.app.state.config


@router.get("/modalities")
async def modalities(raw_request: Request) -> list[Modality]:
    """Return the list of modalities supported by this model."""
    config: EricConfig = raw_request.app.state.config
    return list(MODEL_REGISTRY[config.model.hf_config.model_type].modality.keys())


@router.get("/queue_length")
async def get_queue_length(raw_request: Request) -> int:
    """Return the list of modalities supported by this model."""
    queue_counter: QueueCounter = raw_request.app.state.queue_counter
    return await queue_counter.get_count()


@router.post("/embeddings")
async def embeddings(
    request: EmbeddingRequest,
    raw_request: Request,
    raw_response: Response,
) -> EmbeddingResponse:
    """Handler for embedding requests."""
    span = trace.get_current_span()
    for data_item in request.data:
        span.set_attribute(
            f"eric.embeddings.data.{data_item.id}.url",
            data_item.url,
        )

    # Request validation: model id and modality
    config: EricConfig = raw_request.app.state.config
    queue_counter: QueueCounter = raw_request.app.state.queue_counter
    supported_model_ids = [config.model.id, *config.model.adapter_model_ids]
    for data_item in request.data:
        if data_item.model_id not in supported_model_ids:
            logger.info("Data item %s has unknown model ID %s", data_item.id, data_item.model_id)
            raw_response.status_code = status.HTTP_400_BAD_REQUEST
            return EmbeddingResponse(status=Status.ERROR, error_message="Unsupported model ID")
        if data_item.modality != config.model.modality:
            logger.info("Data item %s has unsupported modality %s", data_item.id, data_item.modality)
            raw_response.status_code = status.HTTP_400_BAD_REQUEST
            return EmbeddingResponse(status=Status.ERROR, error_message="Unsupported modality")

    processor: Processor = raw_request.app.state.processor
    engine_client: EngineClient = raw_request.app.state.engine_client

    await queue_counter.increment()
    # Load data from URLs and apply processing
    processed = await processor.process(request.data)

    # Send to engine process (embedding + transmission via Tensor Sidecar)
    response = await engine_client.embed(uuid.uuid4().hex, processed)

    match response.status:
        case Status.SUCCESS:
            raw_response.status_code = status.HTTP_200_OK
        case Status.ERROR:
            raw_response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        case _:
            logger.error("Unexpected status: %s", response.status)
            raw_response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    await queue_counter.decrement()
    return response


def init_app_state(app: FastAPI, config: EricConfig) -> None:
    """Initialize the app state with the configuration and engine client."""
    app.state.config = config
    app.state.processor = Processor(config.model.id, config.modality)
    app.state.engine_client = EngineClient(config)
    app.state.queue_counter = QueueCounter()


def create_app(config: EricConfig) -> FastAPI:
    """Create a FastAPI app with the given configuration."""
    app = FastAPI()
    app.include_router(router)
    init_app_state(app, config)
    return app
