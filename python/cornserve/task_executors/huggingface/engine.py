"""Colocated engine for HuggingFace task executor."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator

from cornserve.logging import get_logger
from cornserve.task_executors.huggingface.api import HuggingFaceRequest, HuggingFaceResponse, Status, TaskType
from cornserve.task_executors.huggingface.models.qwen_image import QwenImageModel
from cornserve.task_executors.huggingface.models.qwen_omni import QwenOmniModel

logger = get_logger(__name__)


class HuggingFaceEngine:
    """Colocated engine for HuggingFace task execution.

    Handles request queuing, batching, and model inference without
    requiring a separate process.
    """

    def __init__(self, max_batch_size: int = 1):
        """Initialize the engine.

        Args:
            max_batch_size: Maximum batch size for processing requests.
        """
        self.max_batch_size = max_batch_size
        self.request_queue: asyncio.Queue[tuple[HuggingFaceRequest, asyncio.Future]] = asyncio.Queue()

        # Models are loaded on-demand
        self.qwen_image_model: QwenImageModel | None = None
        self.qwen_omni_model: QwenOmniModel | None = None

        # Task for processing requests
        self._processing_task: asyncio.Task | None = None
        self._shutdown = False

    async def start(self) -> None:
        """Start the engine processing loop."""
        if self._processing_task is not None:
            logger.warning("Engine is already running")
            return

        logger.info("Starting HuggingFace engine")
        self._processing_task = asyncio.create_task(self._process_requests())

    async def shutdown(self) -> None:
        """Shutdown the engine."""
        logger.info("Shutting down HuggingFace engine")
        self._shutdown = True

        if self._processing_task:
            self._processing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._processing_task

    async def generate(
        self, request: HuggingFaceRequest
    ) -> HuggingFaceResponse | AsyncGenerator[HuggingFaceResponse, None]:
        """Generate response for a request.

        Args:
            request: The request to process.

        Returns:
            Response or async generator of responses for streaming.
        """
        future: asyncio.Future[HuggingFaceResponse | AsyncGenerator[HuggingFaceResponse, None]] = asyncio.Future()

        await self.request_queue.put((request, future))

        try:
            result = await future
            return result
        except Exception as e:
            logger.exception("Error processing request: %s", e)
            return HuggingFaceResponse(status=Status.ERROR, error_message=str(e))

    async def _process_requests(self) -> None:
        """Process requests from the queue."""
        logger.info("Started request processing loop")

        while not self._shutdown:
            try:
                # Get request from queue with timeout
                try:
                    request, future = await asyncio.wait_for(self.request_queue.get(), timeout=1.0)
                except TimeoutError:
                    continue

                # Process the request
                try:
                    if request.task_type == TaskType.QWEN_IMAGE:
                        result = await self._process_qwen_image_request(request)
                    elif request.task_type == TaskType.QWEN_OMNI:
                        result = await self._process_qwen_omni_request(request)
                    else:
                        result = HuggingFaceResponse(
                            status=Status.ERROR, error_message=f"Unknown task type: {request.task_type}"
                        )

                    future.set_result(result)

                except Exception as e:
                    logger.exception("Error processing request: %s", e)
                    future.set_exception(e)

            except Exception as e:
                logger.exception("Error in processing loop: %s", e)

        logger.info("Request processing loop ended")

    async def _process_qwen_image_request(self, request: HuggingFaceRequest) -> HuggingFaceResponse:
        """Process a Qwen-Image request.

        Args:
            request: The image generation request.

        Returns:
            Response with generated image.
        """
        # Load model on demand
        if self.qwen_image_model is None:
            logger.info("Loading Qwen-Image model: %s", request.model_id)
            try:
                self.qwen_image_model = QwenImageModel(request.model_id)
            except Exception as e:
                logger.exception("Failed to load Qwen-Image model: %s", e)
                return HuggingFaceResponse(status=Status.ERROR, error_message=f"Failed to load model: {e}")

        # Generate image
        try:
            image_b64 = await self.qwen_image_model.generate(
                prompt=request.prompt,
                height=request.height,
                width=request.width,
                num_inference_steps=request.num_inference_steps,
            )

            return HuggingFaceResponse(status=Status.SUCCESS, image=image_b64)

        except Exception as e:
            logger.exception("Error generating image: %s", e)
            return HuggingFaceResponse(status=Status.ERROR, error_message=f"Error generating image: {e}")

    async def _process_qwen_omni_request(
        self, request: HuggingFaceRequest
    ) -> AsyncGenerator[HuggingFaceResponse, None]:
        """Process a Qwen 2.5 Omni request.

        Args:
            request: The multimodal generation request.

        Returns:
            Async generator of response chunks.
        """
        # Load model on demand
        if self.qwen_omni_model is None:
            logger.info("Loading Qwen 2.5 Omni model: %s", request.model_id)
            try:
                self.qwen_omni_model = QwenOmniModel(request.model_id)
            except Exception as e:
                logger.exception("Failed to load Qwen 2.5 Omni model: %s", e)
                yield HuggingFaceResponse(status=Status.ERROR, error_message=f"Failed to load model: {e}")
                return

        # Generate streaming response
        try:
            async for chunk in self.qwen_omni_model.generate_stream(request):
                yield chunk

        except Exception as e:
            logger.exception("Error generating Qwen 2.5 Omni response: %s", e)
            yield HuggingFaceResponse(status=Status.ERROR, error_message=f"Error generating response: {e}")
