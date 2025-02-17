import asyncio

from fastapi import FastAPI, APIRouter, Request, Response, status

from cornserve.logging import get_logger
from cornserve.task_executors.eric.config import EricConfig
from cornserve.task_executors.eric.engine.client import EngineClient
from cornserve.task_executors.eric.router.processor import Processor
from cornserve.task_executors.eric.schema import EmbeddingRequest, EmbeddingResponse, EmbeddingStatus

router = APIRouter()
logger = get_logger(__name__)


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


@router.post("/embeddings")
async def embeddings(request: EmbeddingRequest, raw_request: Request) -> Response:
    processor: Processor = raw_request.app.state.processor
    engine_client: EngineClient = raw_request.app.state.engine_client

    # Load data from URLs and apply processing
    processed = await processor.process(request.urls)

    # Send to engine process (embedding + transmission via Tensor Sidecar)
    response = await engine_client.embed(request.request_id, processed)

    match response.status:
        case EmbeddingStatus.SUCCESS:
            return Response(status_code=status.HTTP_200_OK)
        case EmbeddingStatus.ERROR:
            return Response(status_code=status.HTTP_200_OK, content=response.error_message)


def init_app_state(app: FastAPI, config: EricConfig) -> None:
    """Initialize the app state with the configuration and engine client."""
    app.state.config = config
    app.state.engine_client = EngineClient(config)
    app.state.processor = Processor(config.model.id, config.modality.ty, config.modality.num_workers)


def create_app(config: EricConfig) -> FastAPI:
    """Create a FastAPI app with the given configuration."""
    app = FastAPI()
    app.include_router(router)
    init_app_state(app, config)
    return app
