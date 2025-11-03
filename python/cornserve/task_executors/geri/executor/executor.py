"""The model executor manages generation operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any

import torch

from cornserve.logging import get_logger
from cornserve.task_executors.geri.models.base import (
    BatchGeriModel,
    GeriModel,
    StreamGeriModel,
)
from cornserve.task_executors.geri.schema import (
    BatchGenerationResult,
    Status,
    StreamGenerationResult,
)

logger = get_logger(__name__)


class ModelExecutor(ABC):
    """A class to execute generation with a model.

    This is a simplified version compared to Eric's ModelExecutor.
    Since we're not using tensor parallelism initially, this directly
    manages a single model instance and executes generation requests.
    """

    model: GeriModel

    def shutdown(self) -> None:
        """Shutdown the executor and clean up resources."""
        logger.info("Shutting down ModelExecutor")

        if hasattr(self, "model"):
            del self.model

    @abstractmethod
    def generate(self, *args, **kwargs) -> Any:
        """Execute generation with the model.

        The ModelExecutor base class requires that subclasses implement this method, but
        parameters and return type will vary and therefore left to each subclass to decide.
        """


class BatchExecutor(ModelExecutor):
    """Executor for batched (i.e., non-streaming) generation requests."""

    def __init__(self, model: BatchGeriModel) -> None:
        """Initialize the batch executor."""
        self.model = model

    def generate(
        self,
        prompt_embeds: list[torch.Tensor],
        height: int,
        width: int,
        num_inference_steps: int,
    ) -> BatchGenerationResult:
        """Execute batched generation with the model.

        Currently, the primary use case for this class is image generation.

        Args:
            prompt_embeds: List of text embeddings from the LLM encoder, one per batch item.
            height: Height of the generated image in pixels.
            width: Width of the generated image in pixels.
            num_inference_steps: Number of denoising steps to perform.

        Returns:
            Generation result containing images or error information.
        """
        try:
            logger.info("Generating content with size %dx%d, %d inference steps", height, width, num_inference_steps)

            # Generate images using the model (returns PNG bytes directly)
            assert isinstance(self.model, BatchGeriModel)
            generated_bytes = self.model.generate(
                prompt_embeds=prompt_embeds,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
            )

            logger.info("Generation completed successfully, got %d images as PNG bytes", len(generated_bytes))
            return BatchGenerationResult(status=Status.SUCCESS, generated=generated_bytes)

        except Exception as e:
            logger.exception("Generation failed: %s", str(e))
            return BatchGenerationResult(status=Status.ERROR, error_message=f"Generation failed: {str(e)}")


class StreamExecutor(ModelExecutor):
    """Executor for streamed generation requests."""

    def __init__(self, model: StreamGeriModel):
        """Initialize the batch executor."""
        self.model = model

    def generate(
        self,
        prompt_embeds: list[torch.Tensor],
        chunk_size: int | None,
        left_context_size: int | None,
    ) -> StreamGenerationResult:
        """Execute streamed generation with the model.

        Currently, the primary use case for this class is audio generation.

        Args:
            prompt_embeds: List of text embeddings from the LLM encoder, one per batch item.
            chunk_size: number of codes to be processed at a time
            left_context_size: number of codes immediately prior to each chunk to be processed as context

        Returns:
            Result holding a generator that will iteratively yield results as they become ready.
        """
        try:
            logger.info("Beginning streamed generation")

            gen_kwargs = {
                "prompt_embeds": prompt_embeds,
                "chunk_size": chunk_size,
                "left_context_size": left_context_size,
            }
            # only pass in non-None arguments
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
            assert isinstance(self.model, StreamGeriModel)
            streamed_generator: Generator[list[torch.Tensor | None], None, None] = self.model.generate(**gen_kwargs)

            logger.info("Obtained generator object")
            return StreamGenerationResult(status=Status.SUCCESS, generator=streamed_generator)

        except Exception as e:
            logger.exception("Generation failed: %s", str(e))
            return StreamGenerationResult(status=Status.ERROR, error_message=f"Generation failed: {str(e)}")
