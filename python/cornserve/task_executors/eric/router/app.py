"""Eric FastAPI app definition."""

from fastapi import FastAPI, APIRouter, Request, Response, status

from cornserve.logging import get_logger
from cornserve.task_executors.eric.config import EricConfig
from cornserve.task_executors.eric.engine.client import EngineClient
from cornserve.task_executors.eric.router.processor import Processor
from cornserve.task_executors.eric.models.registry import MODEL_REGISTRY
from cornserve.task_executors.eric.schema import EmbeddingRequest, Status, Modality


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


@router.get("/modalities")
async def modalities(raw_request: Request) -> list[Modality]:
    """Return the list of modalities supported by this model."""
    config: EricConfig = raw_request.app.state.config
    return list(MODEL_REGISTRY[config.model.hf_config.model_type].modality.keys())


@router.post("/embeddings")
async def embeddings(request: EmbeddingRequest, raw_request: Request) -> Response:
    """Handler for embedding requests."""
    processor: Processor = raw_request.app.state.processor
    engine_client: EngineClient = raw_request.app.state.engine_client

    # Load data from URLs and apply processing
    processed = await processor.process(request.data)

    # Send to engine process (embedding + transmission via Tensor Sidecar)
    response = await engine_client.embed(
        request.id,
        request.receiver_sidecar_ranks,
        processed,
    )

    match response.status:
        case Status.SUCCESS:
            return Response(status_code=status.HTTP_200_OK)
        case Status.ERROR:
            return Response(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=response.error_message,
            )


def init_app_state(app: FastAPI, config: EricConfig) -> None:
    """Initialize the app state with the configuration and engine client."""
    app.state.config = config
    app.state.processor = Processor(config.model.id, config.modality)
    app.state.engine_client = EngineClient(config)


def create_app(config: EricConfig) -> FastAPI:
    """Create a FastAPI app with the given configuration."""
    app = FastAPI()
    app.include_router(router)
    init_app_state(app, config)
    return app
