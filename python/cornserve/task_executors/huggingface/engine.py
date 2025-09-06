"""Colocated engine for HuggingFace task executor."""

from __future__ import annotations

import asyncio
import contextlib

from cornserve.logging import get_logger
from cornserve.task_executors.huggingface.api import HuggingFaceRequest, HuggingFaceResponse, ModelType
from cornserve.task_executors.huggingface.models.base import HFModel
from cornserve.task_executors.huggingface.models.qwen_image import QwenImageModel
from cornserve.task_executors.huggingface.models.qwen_omni import QwenOmniModel

logger = get_logger(__name__)


class HuggingFaceEngine:
    """Colocated engine for HuggingFace model execution."""

    def __init__(self, max_batch_size: int, model_type: ModelType, model_id: str) -> None:
        """Initialize the engine."""
        self.max_batch_size = max_batch_size
        self.request_queue: asyncio.Queue[tuple[HuggingFaceRequest, asyncio.Future]] = asyncio.Queue()

        # Load the model based on type
        match model_type:
            case ModelType.QWEN_IMAGE:
                model = QwenImageModel(model_id)
            case ModelType.QWEN_OMNI:
                model = QwenOmniModel(model_id)
            case _:
                raise ValueError(f"Unsupported model type: {model_type}. Supported types: {list(ModelType)}")
        self.model: HFModel = model

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
            with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                await asyncio.wait_for(self._processing_task, timeout=5.0)

    async def generate(self, request: HuggingFaceRequest) -> HuggingFaceResponse:
        """Generate response for a request."""
        future: asyncio.Future[HuggingFaceResponse] = asyncio.Future()

        await self.request_queue.put((request, future))

        try:
            return await future
        except Exception as e:
            logger.exception("Error processing request: %s", e)
            raise

    async def _process_requests(self) -> None:
        """Process requests from the queue."""
        logger.info("Started request processing loop")

        while not self._shutdown:
            try:
                # Get request from queue (FCFS)
                request, future = await self.request_queue.get()

                # Process the request
                try:
                    response = await asyncio.to_thread(self.model.generate, request)
                    future.set_result(response)
                except Exception as e:
                    future.set_exception(e)

            except Exception as e:
                logger.exception("Error in processing loop: %s", e)

        logger.info("Request processing loop ended")
