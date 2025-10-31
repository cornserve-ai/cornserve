"""The model executor manages generation operations."""

from __future__ import annotations

from collections.abc import Generator

import torch

from cornserve.logging import get_logger
from cornserve.task_executors.geri.api import Status
from cornserve.task_executors.geri.models.base import BatchGeriModel, GeriModel, StreamGeriModel
from cornserve.task_executors.geri.schema import GenerationResult

logger = get_logger(__name__)


class ModelExecutor:
    """A class to execute generation with a model.

    This is a simplified version compared to Eric's ModelExecutor.
    Since we're not using tensor parallelism initially, this directly
    manages a single model instance and executes generation requests.
    """

    def __init__(self, model: GeriModel) -> None:
        """Initialize the executor."""
        self.model = model

    def generate(
        self,
        prompt_embeds: list[torch.Tensor],
        height: int,
        width: int,
        num_inference_steps: int,
    ) -> GenerationResult:
        """Execute generation with the model.

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

            if not isinstance(self.model, BatchGeriModel):
                raise TypeError(
                    f"Expected self.model to be a BatchGeriModel, but got {type(self.model).__name__} instead."
                )

            # Generate images using the model (returns PNG bytes directly)
            generated_bytes = self.model.generate(
                prompt_embeds=prompt_embeds,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
            )

            logger.info("Generation completed successfully, got %d images as PNG bytes", len(generated_bytes))
            return GenerationResult(status=Status.SUCCESS, generated=generated_bytes)

        except Exception as e:
            logger.exception("Generation failed: %s", str(e))
            return GenerationResult(status=Status.ERROR, error_message=f"Generation failed: {str(e)}")

    def generate_streaming(
        self,
        prompt_embeds: list[torch.Tensor],
        chunk_size: int | None,
        left_context_size: int | None,
    ) -> GenerationResult:
        """Execute streamed generation with the model.

        Args:
            prompt_embeds: List of text embeddings from the LLM encoder, one per batch item.
            chunk_size: number of codes to be processed at a time
            left_context_size: number of codes immediately prior to each chunk to be processed as context

        Returns:
            Generator that will iteratively yield results as they become ready.
        """
        try:
            logger.info("Beginning streamed generation")

            if not isinstance(self.model, StreamGeriModel):
                raise TypeError(
                    f"Expected self.model to be a StreamGeriModel, but got {type(self.model).__name__} instead."
                )

            gen_kwargs = {
                "prompt_embeds": prompt_embeds,
                "chunk_size": chunk_size,
                "left_context_size": left_context_size,
            }
            # only pass in non-None arguments
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

            # Generate images using the model (returns PNG bytes directly)
            streamed_generator: Generator[torch.Tensor, None, None] = self.model.generate(**gen_kwargs)

            logger.info("Obtained generator object")
            return GenerationResult(status=Status.SUCCESS, streamed_generator=streamed_generator)

        except Exception as e:
            logger.exception("Generation failed: %s", str(e))
            return GenerationResult(status=Status.ERROR, error_message=f"Generation failed: {str(e)}")

    def shutdown(self) -> None:
        """Shutdown the executor and clean up resources."""
        logger.info("Shutting down ModelExecutor")

        if hasattr(self, "model"):
            del self.model
