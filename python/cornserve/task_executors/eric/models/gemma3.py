from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import numpy.typing as npt
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.gemma3.configuration_gemma3 import Gemma3Config

from .base import EricModel
from cornserve.task_executors.eric.schema import Modality
from cornserve.task_executors.eric.router.processor import BaseModalityProcessor
from cornserve.task_executors.eric.models.layers.norm import GemmaRMSNorm
from cornserve.task_executors.eric.models.layers.siglip import SiglipVisionModel


class Gemma3MultiModalProjector(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()

        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(config.vision_config.hidden_size, config.text_config.hidden_size)
        )

        self.mm_soft_emb_norm = GemmaRMSNorm(config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps)

        self.patches_per_image = int(config.vision_config.image_size // config.vision_config.patch_size)
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

    def forward(self, vision_outputs: torch.Tensor):
        batch_size, _, seq_length = vision_outputs.shape

        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size, seq_length, self.patches_per_image, self.patches_per_image
        )
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        projected_vision_outputs = torch.matmul(normed_vision_outputs, self.mm_input_projection_weight)
        return projected_vision_outputs.type_as(vision_outputs)


class Gemma3VisionEncoder(EricModel):
    def __init__(self, config: Gemma3Config) -> None:
        super().__init__()
        self.config = config

        self.vision_tower = SiglipVisionModel(config=config.vision_config)
        self.multi_modal_projector = Gemma3MultiModalProjector(config)

    @property
    def dtype(self) -> torch.dtype:
        return self.multi_modal_projector.mm_input_projection_weight.dtype

    @property
    def device(self) -> torch.device:
        return self.multi_modal_projector.mm_input_projection_weight.device

    @property
    def chunk_shape(self) -> tuple[int, ...]:
        """Fixed resolution ViT, so vision tokens worth one tile."""
        image_size: int = self.config.vision_config.image_size
        patch_size: int = self.config.vision_config.patch_size
        num_patches = image_size // patch_size
        return (1, num_patches**2, self.config.text_config.hidden_size)

    def forward(
        self,
        modality: Modality,
        batch: dict[str, list[torch.Tensor]],
    ) -> list[torch.Tensor]:
        """Forward pass of the model.

        Though it is not activated at the moment, Gemma 3 supports Pan & Scan.
        It checks the image's aspect ratio, and if it's very scewed (based on
        a configurable threshold), it will crop the images to patches. Finally,
        the whole image + all patches are passed to the model. For instance, if
        The image ends up with two patches (e.g., because it was very wide),
        `num_crops` will be 2 and `pixel_values` will be a list of 3 tensors.

        Only supports the image modality. `batch` is expected to have the following keys:
        - `pixel_values`: The pixel values of the images.
           Each [num_patches, 3, image_size (896), image_size (896)].
           The number of patches can be different for each image.
        - `num_crops`: The number of extra Pan & Scan crops for each image.
           If Pan & Scan is not enabled, this will be 0. Each [1,].
        """
        if modality != Modality.IMAGE:
            raise ValueError(f"Unsupported modality: {modality}")

        # Sanity check
        assert len(batch["pixel_values"]) == len(batch["num_crops"])
        for pixels in batch["pixel_values"]:
            assert pixels.ndim == 4
            assert pixels.shape[1] == 3
            assert pixels.shape[2] == pixels.shape[3] == self.config.vision_config.image_size

        # Batch
        pixel_values = torch.cat(batch["pixel_values"]).to(device=self.device, dtype=self.dtype)
        num_patches = torch.cat(batch["num_crops"]) + 1

        # Embedding
        image_features = self.vision_tower(pixel_values)
        image_embeds = self.multi_modal_projector(image_features)

        # Unbatch
        result = [e.flatten(0, 1) for e in image_embeds.split(num_patches.tolist())]

        return result


class ModalityProcessor(BaseModalityProcessor):
    """Gemma 3 modality processor."""

    def __init__(self, model_id: str) -> None:
        """Initialize the processor."""
        super().__init__(model_id=model_id)
        hf_processor = AutoProcessor.from_pretrained(model_id)
        self.image_processor = hf_processor.image_processor

    def get_image_processor(self) -> Callable | None:
        """Return the image processor."""

        def processor(image: npt.NDArray) -> dict[str, npt.NDArray]:
            """Invoke the HF processor and convert to dict."""
            # If we enable Pan & Scan, the batch dimension (0) may be larger than 1.
            return self.image_processor(images=[image], return_tensors="np").data

        return processor
