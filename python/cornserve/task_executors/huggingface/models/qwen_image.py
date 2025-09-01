"""Qwen-Image model wrapper using HuggingFace diffusers."""

from __future__ import annotations

import base64
import io
from typing import Any

import torch
from diffusers import QwenImagePipeline

from cornserve.logging import get_logger

logger = get_logger(__name__)


class QwenImageModel:
    """Wrapper for Qwen-Image model using HuggingFace diffusers.

    Uses QwenImagePipeline for image generation.
    """

    def __init__(self, model_id: str):
        """Initialize the Qwen-Image model.

        Args:
            model_id: Model ID to load (e.g., "Qwen/Qwen-Image").
        """
        self.model_id = model_id
        logger.info("Loading Qwen-Image model: %s", model_id)

        # Load the pipeline with optimizations
        self.pipe = QwenImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, use_safetensors=True)

        # Move to CUDA if available
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")
            logger.info("Moved Qwen-Image model to CUDA")

        # Enable memory optimizations
        try:
            self.pipe.enable_vae_slicing()
        except Exception as e:
            logger.warning("Could not enable VAE slicing: %s", e)

        logger.info("Successfully loaded Qwen-Image model")

    async def generate(self, prompt: str, height: int, width: int, num_inference_steps: int, **kwargs: Any) -> str:
        """Generate an image from a text prompt.

        Args:
            prompt: Text prompt for image generation.
            height: Height of generated image in pixels.
            width: Width of generated image in pixels.
            num_inference_steps: Number of denoising steps.
            **kwargs: Additional generation parameters.

        Returns:
            Base64-encoded PNG image string.
        """
        try:
            logger.debug("Generating image for prompt: %s", prompt[:100])

            # Add positive magic for better quality (English prompt)
            enhanced_prompt = prompt + ", Ultra HD, 4K, cinematic composition."

            # Generate image
            result = self.pipe(
                prompt=enhanced_prompt, height=height, width=width, num_inference_steps=num_inference_steps, **kwargs
            )

            # Get the generated image
            if hasattr(result, "images") and result.images:
                image = result.images[0]
            else:
                raise RuntimeError("No image generated")

            # Convert PIL Image to base64 PNG
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)

            image_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            logger.debug("Successfully generated image, size: %d bytes", len(image_b64))
            return image_b64

        except Exception as e:
            logger.exception("Error generating image: %s", e)
            raise RuntimeError(f"Error generating image: {e}") from e
