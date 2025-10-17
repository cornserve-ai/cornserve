"""QwenImage model implementation for Geri."""

from __future__ import annotations

import base64
import io

import torch
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from PIL import Image
from torch.nn.utils.rnn import pad_sequence

from cornserve.logging import get_logger
from cornserve.task_executors.geri.models.base import GeriModel

logger = get_logger(__name__)


class QwenImageModel(GeriModel):
    """Qwen-Image model implementation for text-to-image generation.

    This model uses the QwenImagePipeline from diffusers, but removes the text encoder
    components since text encoding is handled by vLLM and embeddings are received
    via the sidecar.
    """

    def __init__(self, model_id: str, torch_dtype: torch.dtype, torch_device: torch.device) -> None:
        """Initialize the QwenImage model.

        Args:
            model_id: Hugging Face model ID (e.g., "Qwen/Qwen-Image").
            torch_dtype: Data type for model weights.
            torch_device: Device to load the model on.
        """
        logger.info("Loading QwenImage model from %s", model_id)

        # First load on CPU to avoid allocating GPU memory for unused components.
        pipeline = QwenImagePipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
        self._embedding_dim = pipeline.text_encoder.config.hidden_size
        pipeline.text_encoder = None
        pipeline.tokenizer = None

        if torch_device.type not in ["cpu", "meta"]:
            pipeline = pipeline.to(torch_device)

        self.pipeline = pipeline

        self._torch_dtype = torch_dtype
        self._torch_device = torch_device
        logger.info("QwenImage model loaded successfully")

    @property
    def dtype(self) -> torch.dtype:
        """The data type of the model."""
        return self.pipeline.dtype

    @property
    def device(self) -> torch.device:
        """The device where the model is loaded."""
        return self.pipeline.device

    @property
    def embedding_dim(self) -> int:
        """The dimension of the prompt embeddings used by the model."""
        return self._embedding_dim

    def generate(
        self,
        prompt_embeds: list[torch.Tensor],
        height: int,
        width: int,
        num_inference_steps: int = 50,
    ) -> list[str]:
        """Generate images from prompt embeddings."""
        batch_size = len(prompt_embeds)
        max_seq_len = max([emb.size(0) for emb in prompt_embeds])

        # Pad and batch prompt embeddings
        padded_embeds = pad_sequence(prompt_embeds, batch_first=True, padding_value=0.0)
        lengths = torch.tensor([emb.size(0) for emb in prompt_embeds], device=self._torch_device)
        prompt_embeds_mask = (
            torch.arange(max_seq_len, device=self._torch_device).unsqueeze(0) < lengths.unsqueeze(1)
        ).long()

        logger.info(
            "Generating %d images with size %dx%d, %d inference steps", batch_size, height, width, num_inference_steps
        )

        result = self.pipeline(
            prompt_embeds=padded_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
        )

        # Cehck type and convert images to PNG bytes
        assert isinstance(result, QwenImagePipelineOutput)
        images = result.images
        assert isinstance(images, list) and all(isinstance(img, Image.Image) for img in images)

        images_png: list[str] = []
        for img in images:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            images_png.append(base64.b64encode(buffer.getvalue()).decode("ascii"))

        logger.info("Generated %d images successfully", len(images_png))

        return images_png
