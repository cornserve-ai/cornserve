"""Qwen-Image model wrapper using HuggingFace diffusers."""

from __future__ import annotations

import base64
import io

import torch
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline

from cornserve.logging import get_logger
from cornserve.task_executors.huggingface.api import HuggingFaceRequest, HuggingFaceResponse, Status
from cornserve.task_executors.huggingface.models.base import HFModel

logger = get_logger(__name__)


class QwenImageModel(HFModel):
    """Wrapper for Qwen-Image model using HuggingFace diffusers.

    Uses QwenImagePipeline for image generation.
    """

    def __init__(self, model_id: str) -> None:
        """Initialize the Qwen-Image model.

        Args:
            model_id: Model ID to load (e.g., "Qwen/Qwen-Image").
        """
        self.model_id = model_id
        logger.info("Loading Qwen-Image model: %s", model_id)

        # Load the pipeline with optimizations
        self.pipeline = QwenImagePipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        )

        logger.info("Successfully loaded Qwen-Image model")

    def generate(self, request: HuggingFaceRequest) -> HuggingFaceResponse:
        """Generate an image from a text prompt."""
        if (prompt := request.prompt) is None:
            raise ValueError("Prompt is required for Qwen-Image generation")

        if (height := request.height) is None:
            raise ValueError("Height is required for Qwen-Image generation")

        if (width := request.width) is None:
            raise ValueError("Width is required for Qwen-Image generation")

        if (num_inference_steps := request.num_inference_steps) is None:
            raise ValueError("num_inference_steps is required for Qwen-Image generation")

        # Generate image
        result = self.pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
        )

        # Get the generated image
        assert isinstance(result, QwenImagePipelineOutput)
        image = result.images[0]

        # Convert PIL Image to base64 PNG
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_png = base64.b64encode(buffer.getvalue()).decode("ascii")

        logger.debug("Successfully generated image")
        return HuggingFaceResponse(status=Status.SUCCESS, image=image_png)
