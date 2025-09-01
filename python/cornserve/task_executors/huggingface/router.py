"""FastAPI router for HuggingFace task executor."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from cornserve.logging import get_logger
from cornserve.task_executors.huggingface.api import HuggingFaceRequest, HuggingFaceResponse, Status, TaskType
from cornserve.task_executors.huggingface.config import HuggingFaceConfig
from cornserve.task_executors.huggingface.engine import HuggingFaceEngine

logger = get_logger(__name__)


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
        version="1.0.0",
    )

    # Initialize engine
    engine = HuggingFaceEngine(max_batch_size=config.model.max_batch_size)
    app.state.engine = engine
    app.state.config = config

    @app.on_event("startup")
    async def startup_event():
        """Start the engine when the application starts."""
        logger.info("Starting HuggingFace task executor")
        await engine.start()

    @app.on_event("shutdown")
    async def shutdown_event():
        """Shutdown the engine when the application stops."""
        logger.info("Shutting down HuggingFace task executor")
        await engine.shutdown()

    @app.post("/generate")
    async def generate(request: HuggingFaceRequest) -> Any:
        """Generate response for a request.

        Args:
            request: The generation request.

        Returns:
            Response or streaming response.
        """
        try:
            # Validate request matches configured task type
            if request.task_type != config.task_type:
                raise HTTPException(
                    status_code=400, detail=f"Task type mismatch. Expected {config.task_type}, got {request.task_type}"
                )

            # Validate model ID
            if request.model_id != config.model.id:
                raise HTTPException(
                    status_code=400, detail=f"Model ID mismatch. Expected {config.model.id}, got {request.model_id}"
                )

            # Generate response
            result = await engine.generate(request)

            if request.task_type == TaskType.QWEN_IMAGE:
                # Non-streaming response for image generation
                if not isinstance(result, HuggingFaceResponse):
                    raise HTTPException(status_code=500, detail="Invalid response type for image generation")
                return result

            elif request.task_type == TaskType.QWEN_OMNI:
                # Streaming response for omni generation
                if not hasattr(result, "__aiter__"):
                    raise HTTPException(status_code=500, detail="Invalid response type for omni generation")

                async def stream_generator():
                    """Generate streaming response in SSE format."""
                    try:
                        async for chunk in result:
                            if chunk.status != Status.SUCCESS:
                                yield f"data: {chunk.model_dump_json()}\n\n"
                                return

                            yield f"data: {chunk.model_dump_json()}\n\n"

                        # Send done signal
                        yield "data: [DONE]\n\n"

                    except Exception as e:
                        logger.exception("Error in streaming response: %s", e)
                        error_response = HuggingFaceResponse(status=Status.ERROR, error_message=f"Streaming error: {e}")
                        yield f"data: {error_response.model_dump_json()}\n\n"

                return StreamingResponse(
                    stream_generator(),
                    media_type="text/plain",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    },
                )

            else:
                raise HTTPException(status_code=400, detail=f"Unknown task type: {request.task_type}")

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Error processing request: %s", e)
            raise HTTPException(status_code=500, detail=f"Internal server error: {e}") from e

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint.

        Returns:
            Health status.
        """
        return {"status": "healthy"}

    @app.get("/info")
    async def info() -> dict[str, Any]:
        """Information about the task executor.

        Returns:
            Task executor information.
        """
        return {
            "task_type": config.task_type,
            "model_id": config.model.id,
            "max_batch_size": config.model.max_batch_size,
        }

    return app
