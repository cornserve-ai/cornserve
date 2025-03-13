import math
from typing import Callable

import torch
import torch.nn as nn
import numpy.typing as npt
from transformers import AutoProcessor
from transformers.models.llava_onevision.configuration_llava_onevision import LlavaOnevisionConfig
from transformers.models.llava_onevision.modeling_llava_onevision import (
    get_anyres_image_grid_shape,
    unpad_image,
)

from .base import EricModel
from .layers.activations import get_act_fn
from .layers.vit import init_vision_tower_for_llava
from cornserve.task_executors.eric.schema import Modality
from cornserve.task_executors.eric.router.processor import BaseModalityProcessor


class LlavaOneVisionMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaOnevisionConfig):
        super().__init__()

        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size, config.text_config.hidden_size, bias=config.multimodal_projector_bias
        )
        self.act = get_act_fn(config.projector_hidden_act)
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size, config.text_config.hidden_size, bias=config.multimodal_projector_bias
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LlavaOneVisionEncoder(EricModel):
    def __init__(self, config: LlavaOnevisionConfig) -> None:
        super().__init__()
        self.config = config

        # Initialize the vision tower only up to the required feature layer
        self.vision_tower = init_vision_tower_for_llava(config, require_post_norm=False)

        self.multi_modal_projector = LlavaOneVisionMultiModalProjector(config)
        self.image_newline = nn.Parameter(torch.empty(config.text_config.hidden_size))

    @property
    def dtype(self) -> torch.dtype:
        return self.multi_modal_projector.linear_1.weight.dtype

    @property
    def chunk_shape(self) -> tuple[int, ...]:
        """Fixed resolution ViT, so vision tokens worth one tile."""
        image_size: int = self.config.vision_config.image_size
        patch_size: int = self.config.vision_config.patch_size
        num_patches = image_size // patch_size
        return (1, num_patches**2, self.config.text_config.hidden_size)

    @property
    def device(self) -> torch.device:
        return self.multi_modal_projector.linear_1.weight.device

    def _select_image_features(self, image_features: torch.Tensor, *, strategy: str) -> torch.Tensor:
        if strategy == "default":
            return image_features[:, 1:]
        elif strategy == "full":
            return image_features

        raise ValueError(f"Unexpected select feature strategy: {strategy}")

    # Based on: https://github.com/haotian-liu/LLaVA/blob/main/llava/model/llava_arch.py
    def _merge_image_patch_embeddings(
        self,
        image_size: torch.Tensor,
        patch_embeddings: torch.Tensor,
        *,
        image_newline=None,
        vision_aspect_ratio="anyres_max_9",
        strategy: str,
    ) -> torch.Tensor:
        if strategy == "flat":
            return patch_embeddings.flatten(0, 1)

        if strategy.startswith("spatial"):
            height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size

            base_patch_embeds = patch_embeddings[0]
            if height * width != base_patch_embeds.shape[0]:
                raise ValueError("The number of patches is not consistent with the image size.")

            if patch_embeddings.shape[0] > 1:
                other_patch_embeds = patch_embeddings[1:]

                # Move to CPU to avoid floating-point errors
                orig_height, orig_width = image_size.tolist()

                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    (orig_height, orig_width),
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                num_patches = num_patch_height * num_patch_width

                # Image patches might be padded for batch processing
                other_patch_embeds = other_patch_embeds[:num_patches].view(
                    num_patch_height, num_patch_width, height, width, -1
                )

                if "unpad" in strategy:
                    other_patch_embeds = (
                        other_patch_embeds.permute(4, 0, 2, 1, 3).contiguous().flatten(1, 2).flatten(2, 3)
                    )
                    other_patch_embeds = unpad_image(other_patch_embeds, (orig_height, orig_width))
                    max_num_patches = int(vision_aspect_ratio.removeprefix("anyres_max_"))
                    channels, curr_height, curr_width = other_patch_embeds.shape
                    ratio = math.sqrt(curr_height * curr_width / (max_num_patches * height**2))
                    if ratio > 1.1:
                        other_patch_embeds = other_patch_embeds[None]
                        other_patch_embeds = nn.functional.interpolate(
                            other_patch_embeds, [int(curr_height // ratio), int(curr_width // ratio)], mode="bilinear"
                        )[0]
                    if image_newline is not None:
                        other_patch_embeds = torch.cat(
                            (
                                other_patch_embeds,
                                image_newline[:, None, None]
                                .expand(*other_patch_embeds.shape[:-1], 1)
                                .to(other_patch_embeds.device),
                            ),
                            dim=-1,
                        )
                    other_patch_embeds = other_patch_embeds.flatten(1, 2).transpose(0, 1)
                else:
                    other_patch_embeds = other_patch_embeds.permute(0, 2, 1, 3, 4).contiguous().flatten(0, 3)

                merged_patch_embeddings = torch.cat((base_patch_embeds, other_patch_embeds), dim=0)
            else:
                if "unpad" in strategy:
                    merged_patch_embeddings = torch.cat(
                        (base_patch_embeds, self.image_newline[None].to(base_patch_embeds.device)), dim=0
                    )
                else:
                    merged_patch_embeddings = base_patch_embeds

            return merged_patch_embeddings

        raise ValueError(f"Unexpected patch merge strategy: {strategy}")

    def _add_image_newline(
        self,
        video_features: torch.Tensor,
        videos: int = 1,
        frames: int = 1,
        strategy: str = "one_token",
    ) -> torch.Tensor:
        if strategy == "one_token":
            video_features = video_features.reshape(videos, frames * video_features.shape[1], -1)
            image_newline = self.image_newline[None, None, :].repeat(videos, 1, 1).to(video_features.device)
            video_features = torch.cat((video_features, image_newline), dim=1)
            return video_features
        raise ValueError(f"Unexpected video newline strategy: {strategy}")

    def apply_pooling(self, features: torch.Tensor, scale: int = 2) -> torch.Tensor:
        """Apply pooling with fixed width and height scale features."""
        vision_config = self.config.vision_config
        height = width = vision_config.image_size // vision_config.patch_size
        batch_frames, _, dim = features.shape
        features = features.view(batch_frames, height, width, -1)
        features = features.permute(0, 3, 1, 2)

        # TODO support other pooling types config
        height, width = features.shape[2:]
        scaled_shape = [math.ceil(height / scale), math.ceil(width / scale)]
        image_feature = nn.functional.interpolate(features, size=scaled_shape, mode="bilinear")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(batch_frames, -1, dim)
        return image_feature

    def get_image_embeddings(
        self,
        pixel_values: list[torch.Tensor],
        image_sizes: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Compute image embeddings."""
        # Sanity check
        assert len(pixel_values) == len(image_sizes)
        assert isinstance(pixel_values, list)
        assert isinstance(image_sizes, list)
        for item in pixel_values:
            assert isinstance(item, torch.Tensor)
            assert item.ndim == 4
            assert item.shape[1] == 3
            assert item.shape[2] == item.shape[3] == self.config.vision_config.image_size
        for item in image_sizes:
            assert isinstance(item, torch.Tensor)

        # Stack pixel values and embed with vision tower and projector
        stacked_pixel_values = torch.cat(pixel_values, dim=0).to(device=self.device, dtype=self.dtype)
        image_features = self.vision_tower(stacked_pixel_values)
        image_features = self._select_image_features(
            image_features,
            strategy=self.config.vision_feature_select_strategy,
        )
        image_features = self.multi_modal_projector(image_features)

        # Unbatch images
        num_tiles_per_image = [v.shape[0] for v in pixel_values]
        patch_embeddings = image_features.split(num_tiles_per_image)

        # Spatially merge patch embeddings
        # Images are scaled differently (aiming for patches worth roughly 9 tiles)
        # based on their original resolution.
        return [
            self._merge_image_patch_embeddings(
                image_sizes[i].squeeze(0),
                patch_features_batch,
                image_newline=self.image_newline,
                strategy="spatial_unpad",
            )
            for i, patch_features_batch in enumerate(patch_embeddings)
        ]

    def get_video_embeddings(self, pixel_values_videos: list[torch.Tensor]) -> list[torch.Tensor]:
        """Compute video embeddings."""
        # Sanity check
        assert isinstance(pixel_values_videos, list)
        for item in pixel_values_videos:
            assert isinstance(item, torch.Tensor)
            assert item.ndim == 4
            assert item.shape[1] == 3
            assert item.shape[2] == item.shape[3] == self.config.vision_config.image_size

        # Stack pixel values and embed with vision tower and projector
        stacked_pixel_values = torch.cat(pixel_values_videos, dim=0).to(device=self.device, dtype=self.dtype)
        video_features = self.vision_tower(stacked_pixel_values)
        video_features = self._select_image_features(
            video_features,
            strategy=self.config.vision_feature_select_strategy,
        )
        video_features = self.multi_modal_projector(video_features)
        # Videos are uniformly pooled spatially by a factor of (roughly) 2
        video_features = self.apply_pooling(video_features)

        # Unbatch videos
        num_tiles_per_video = [v.shape[0] for v in pixel_values_videos]
        video_embeddings = video_features.split(num_tiles_per_video)

        # Add image newline tokens. Somehow this is just one per video.
        return [
            self._add_image_newline(features.squeeze(0), videos=1, frames=features.shape[0])
            for features in video_embeddings
        ]

    def forward(
        self,
        modality: Modality,
        batch: dict[str, list[torch.Tensor]],
    ) -> list[torch.Tensor]:
        """Forward pass of the model.

        For images, `batch` is expected to have the following keys:
        - `pixel_values`: The pixel values of the images.
            Each [num_tiles, 3, image_size (384), image_size (384)].
            The number of tiles can be different for each image.
        - `image_sizes`: The height and width of the images. Each [2,].

        For videos, `batch` is expected to have the following keys:
        - `pixel_values_videos`: The pixel values of the images.
            Each [num_tiles, 3, image_size (384), image_size (384)].
            The number of tiles can be different for each image.
        """
        # Batch
        match modality:
            case Modality.IMAGE:
                return self.get_image_embeddings(batch["pixel_values"], batch["image_sizes"])
            case Modality.VIDEO:
                return self.get_video_embeddings(batch["pixel_values_videos"])
            case _:
                raise ValueError(f"Unsupported modality: {modality}.")

class ModalityProcessor(BaseModalityProcessor):
    """Llava OneVision modality processor."""

    def __init__(self, model_id: str) -> None:
        """Initialize the processor."""
        super().__init__(model_id=model_id)
        hf_processor = AutoProcessor.from_pretrained(model_id)
        self.image_processor = hf_processor.image_processor
        self.video_processor = hf_processor.video_processor

    def get_image_processor(self) -> Callable | None:
        """Return the image processor."""

        def processor(image: npt.NDArray) -> dict[str, npt.NDArray]:
            """Invoke the HF processor and convert to dict."""
            out = self.image_processor.preprocess(images=[image], return_tensors="np")
            # Batch size is going to be 1, so squeeze it out
            return {k: v.squeeze(0) for k, v in out.data.items()}

        return processor

    def get_video_processor(self) -> Callable | None:
        """Return the video processor."""

        def processor(video: npt.NDArray) -> dict[str, npt.NDArray]:
            """Invoke the HF processor and convert to dict."""
            out = self.video_processor.preprocess(videos=[video], return_tensors="np")
            # Batch size is going to be 1, so squeeze it out
            return {k: v.squeeze(0) for k, v in out.data.items()}

        return processor
