"""The model executor manages generation operations."""

from __future__ import annotations

import torch

from cornserve.logging import get_logger
from cornserve.task_executors.geri.api import Status
from cornserve.task_executors.geri.models.base import GeriModel
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

    def shutdown(self) -> None:
        """Shutdown the executor and clean up resources."""
        logger.info("Shutting down ModelExecutor")

        if hasattr(self, "model"):
            del self.model
