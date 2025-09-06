"""FastAPI router for HuggingFace task executor."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter, FastAPI, Request

from cornserve.logging import get_logger
from cornserve.task_executors.huggingface.api import HuggingFaceRequest, HuggingFaceResponse
from cornserve.task_executors.huggingface.config import HuggingFaceConfig
from cornserve.task_executors.huggingface.engine import HuggingFaceEngine

router = APIRouter()
logger = get_logger(__name__)


@router.post("/generate")
async def generate(request: HuggingFaceRequest, raw_request: Request) -> HuggingFaceResponse:
    """Generate response for a request."""
    engine: HuggingFaceEngine = raw_request.app.state.engine

    logger.info("Received generation request: %s", request)

    return await engine.generate(request)


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Health status.
    """
    return {"status": "healthy"}


@router.get("/info")
async def info(raw_request: Request) -> dict[str, Any]:
    """Information about the task executor.

    Returns:
        Task executor information.
    """
    config: HuggingFaceConfig = raw_request.app.state.config
    return {
        "model_type": config.model.model_type,
        "model_id": config.model.id,
        "max_batch_size": config.model.max_batch_size,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to handle startup and shutdown events."""
    engine: HuggingFaceEngine = app.state.engine
    await engine.start()
    yield
    await engine.shutdown()


def create_app(config: HuggingFaceConfig) -> FastAPI:
    """Create FastAPI application.

    Args:
        config: Configuration for the HuggingFace task executor.

    Returns:
        FastAPI application instance.
    """
    app = FastAPI(
        title="HuggingFace Task Executor",
        description="Task executor for Qwen-Image and Qwen 2.5 Omni models",
        lifespan=lifespan,
    )
    app.include_router(router)

    # Initialize app state
    engine = HuggingFaceEngine(
        max_batch_size=config.model.max_batch_size,
        model_type=config.model.model_type,
        model_id=config.model.id,
    )
    app.state.engine = engine
    app.state.config = config

    return app
