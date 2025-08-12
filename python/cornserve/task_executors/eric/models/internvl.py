from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import numpy.typing as npt
from transformers import BatchEncoding, PretrainedConfig
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.internvl.configuration_internvl import InternVLConfig

from .base import EricModel
from cornserve.task_executors.eric.schema import Modality
from cornserve.task_executors.eric.router.processor import BaseModalityProcessor
from cornserve.task_executors.eric.models.layers.intern_vit import InternVisionModel


class InternVLChatModel(EricModel):
    def __init__(self, config: InternVLConfig) -> None:
        super().__init__()
        self.config = config

        num_hidden_layers = (config.vision_config.num_hidden_layers + config.select_layer + 1) % config.vision_config.num_hidden_layers
        self.vision_model = InternVisionModel(
            config.vision_config,
            num_hidden_layers_override=num_hidden_layers,
        )

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.text_config.hidden_size
        self.downsample_ratio = config.downsample_ratio
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio)**2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio)**2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.vision_model.embeddings.patch_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.vision_model.embeddings.patch_embedding.weight.device

    @property
    def chunk_shape(self) -> tuple[int, ...]:
        """Fixed resolution ViT followed by pooling.

        4096 vision tokens are pooled to 256 tokens.
        Hidden size is 2560 for 4B, 3840 for 12B, and 5376 for 27B.
        This is with Pan & Scan disabled, so any image is just resized to 896x896.
        """
        # HACK: The multimodal projector adapter outputs different hidden sizes, but
        # our sidecar currently expects a fixed size. Use the GCD of three known Gemma 3
        # hidden sizes (2560, 3840, 5376), but eventually, the sidecar should be able to
        # handle different hidden sizes.
        return (1, self.config.mm_tokens_per_image, 256)

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            pass
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward(
        self,
        modality: Modality,
        adapter_name: str,
        batch: dict[str, list[torch.Tensor]],
    ) -> list[torch.Tensor]:
        """Forward pass of the model.

        For images, `batch` is expected to have the following keys:
        - `pixel_values_flat`: The pixel values of the images.
           Each [num_patches, 3, image_size (896), image_size (896)].
           The number of patches can be different for each image.
        - `image_num_patches`: The number of patches for each image.
           If Pan & Scan is not enabled, this will be 0. Each [1,].
        """


class BaseInternVLProcessor:
    """
    This model doesn't define its own HF processor,
    so we implement our own one here.

    The code to insert image tokens is based on:
    https://huggingface.co/OpenGVLab/InternVL2-1B/blob/main/modeling_internvl_chat.py#L252
    """

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        min_dynamic_patch: int | None = None,
        max_dynamic_patch: int | None = None,
        dynamic_image_size: bool | None = None,
    ) -> None:
        super().__init__()

        self.config = config

        image_size: int = config.vision_config.image_size
        patch_size: int = config.vision_config.patch_size

        if min_dynamic_patch is None:
            min_dynamic_patch = config.min_dynamic_patch
        assert isinstance(min_dynamic_patch, int)

        if max_dynamic_patch is None:
            max_dynamic_patch = config.max_dynamic_patch
        assert isinstance(max_dynamic_patch, int)

        if dynamic_image_size is None:
            dynamic_image_size = config.dynamic_image_size
        assert isinstance(dynamic_image_size, bool)

        self.num_image_token = int(
            (image_size // patch_size)**2 * (config.downsample_ratio**2))
        self.image_size = image_size
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail: bool = config.use_thumbnail

    def get_image_repl(
        self,
        feature_size: int,
        num_patches: Optional[int],
    ) -> PromptUpdateDetails[str]:
        raise NotImplementedError

    def resolve_min_max_num(
        self,
        *,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        use_thumbnail: Optional[bool] = None,
    ) -> tuple[int, int]:
        min_dynamic_patch = (self.min_dynamic_patch if min_dynamic_patch
                             is None else min_dynamic_patch)
        max_dynamic_patch = (self.max_dynamic_patch if max_dynamic_patch
                             is None else max_dynamic_patch)
        dynamic_image_size = (self.dynamic_image_size if dynamic_image_size
                              is None else dynamic_image_size)
        use_thumbnail = (self.use_thumbnail
                         if use_thumbnail is None else use_thumbnail)

        return resolve_internvl_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
        )

    def resolve_target_ratios(
        self,
        *,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        use_thumbnail: Optional[bool] = None,
    ) -> list[tuple[int, int]]:
        min_num, max_num = self.resolve_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
        )

        return get_internvl_target_ratios(min_num, max_num)

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        target_ratios = self.resolve_target_ratios(
            use_thumbnail=False,  # Applied in calculate_targets
        )

        num_patches, _, _ = calculate_internvl_targets(
            orig_width=image_width,
            orig_height=image_height,
            image_size=self.image_size,
            target_ratios=target_ratios,
            use_thumbnail=self.use_thumbnail,
        )

        return num_patches * self.num_image_token

    def _images_to_pixel_values_lst(
        self,
        images: list[Image.Image],
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
    ) -> list[torch.Tensor]:
        min_num, max_num = self.resolve_min_max_num(
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=False,  # Applied in image_to_pixel_values
        )

        return [
            image_to_pixel_values_internvl(
                image,
                input_size=self.image_size,
                min_num=min_num,
                max_num=max_num,
                use_thumbnail=self.use_thumbnail,
            ) for image in images
        ]

    def _preprocess_image(
        self,
        text: list[str],
        images: list[Image.Image],
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
    ) -> tuple[list[str], dict[str, torch.Tensor]]:
        if len(images) == 0:
            image_inputs = {}
        else:
            pixel_values_lst = self._images_to_pixel_values_lst(
                images,
                min_dynamic_patch=min_dynamic_patch,
                max_dynamic_patch=max_dynamic_patch,
                dynamic_image_size=dynamic_image_size,
            )
            image_inputs: dict[str, NestedTensors] = {
                "pixel_values_flat":
                torch.cat(pixel_values_lst),
                "image_num_patches":
                torch.tensor([len(item) for item in pixel_values_lst]),
            }

            for pixel_values in pixel_values_lst:
                num_patches = pixel_values.shape[0]
                feature_size = num_patches * self.num_image_token

                image_repl = self.get_image_repl(feature_size, num_patches)
                text = [t.replace('<image>', image_repl.full, 1) for t in text]
        return text, image_inputs

    def _make_batch_input(self,
                          input_item: Optional[Union[Any, list[Any]]] = None):
        if input_item is None:
            input_item = []
        if not isinstance(input_item, list):
            input_item = [input_item]
        return input_item

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        images: Optional[Union[Image.Image, list[Image.Image]]] = None,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> Mapping[str, NestedTensors]:
        text, images = [self._make_batch_input(x) for x in (text, images)]

        text, image_inputs = self._preprocess_image(
            text=text,
            images=images,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
        )

        return {
            **BatchEncoding(tensor_type=return_tensors),
            **image_inputs,
        }


class InternVLProcessor(BaseInternVLProcessor):
    """
    HF Processor for InternVLChatModel with extended video processing logic.

    Code for video processing is adapted from video example:
    https://huggingface.co/OpenGVLab/InternVL3-1B#inference-with-transformers
    """

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        video_token: Optional[str] = None,
    ) -> None:
        super().__init__(
            config=config,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
        )
        # add extra video token for video processing
        self.video_token = video_token

    def _videos_to_pixel_values_lst(
        self,
        videos: list[npt.NDArray],
        dynamic_image_size: Optional[bool] = None,
    ) -> list[torch.Tensor]:
        min_num, max_num = self.resolve_min_max_num(
            min_dynamic_patch=1,
            max_dynamic_patch=1,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=False,  # Applied in image_to_pixel_values
        )

        return [
            video_to_pixel_values_internvl(
                video,
                input_size=self.image_size,
                min_num=min_num,
                max_num=max_num,
                use_thumbnail=False,
            ) for video in videos
        ]

    def _preprocess_video(
        self,
        text: list[str],
        videos: list[npt.NDArray],
        dynamic_image_size: Optional[bool] = None,
    ):
        if len(videos) == 0 or not self.supports_video:
            video_inputs = {}
        else:
            pixel_values_lst_video = self._videos_to_pixel_values_lst(
                videos,
                dynamic_image_size=dynamic_image_size,
            )
            video_inputs: dict[str, NestedTensors] = {
                "pixel_values_flat_video":
                torch.cat(pixel_values_lst_video),
                "video_num_patches":
                torch.tensor([len(item) for item in pixel_values_lst_video]),
            }

            for pixel_values in pixel_values_lst_video:
                num_patches = pixel_values.shape[0]

                video_repl = self.get_video_repl(self.num_image_token,
                                                 num_patches, self.video_token)
                text = [t.replace('<video>', video_repl.full, 1) for t in text]
        return text, video_inputs

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        images: Optional[Union[Image.Image, list[Image.Image]]] = None,
        videos: Optional[Union[npt.NDArray, list[npt.NDArray]]] = None,
        min_dynamic_patch: Optional[int] = None,
        max_dynamic_patch: Optional[int] = None,
        dynamic_image_size: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
    ) -> Mapping[str, NestedTensors]:
        text, images, videos = [
            self._make_batch_input(x) for x in (text, images, videos)
        ]

        text, image_inputs = self._preprocess_image(
            text=text,
            images=images,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_dynamic_patch,
            dynamic_image_size=dynamic_image_size,
        )

        text, video_inputs = self._preprocess_video(
            text=text,
            videos=videos,
            dynamic_image_size=dynamic_image_size,
        )

        return {
            **BatchEncoding(tensor_type=return_tensors),
            **image_inputs,
            **video_inputs,
        }

    def get_image_repl(
        self,
        feature_size: int,
        num_patches: Optional[int],
    ) -> PromptUpdateDetails[str]:
        repl_features = IMG_CONTEXT * feature_size
        repl_full = IMG_START + repl_features + IMG_END

        return PromptUpdateDetails.select_text(repl_full, IMG_CONTEXT)

    def get_video_repl(
        self,
        feature_size: int,
        num_patches: Optional[int] = None,
        video_context_token: str = IMG_CONTEXT,
    ) -> PromptUpdateDetails[str]:
        repl_features = video_context_token * self.num_image_token
        repl_features_with_sep = IMG_START + repl_features + IMG_END
        # num_patches is equal to num_frames
        repl_full = ''.join([
            f'Frame{i+1}: {repl_features_with_sep}' for i in range(num_patches)
        ])

        return PromptUpdateDetails.select_text(repl_full, video_context_token)


class ModalityProcessor(BaseModalityProcessor):
    """Gemma 3 modality processor."""

    def __init__(self, model_id: str) -> None:
        """Initialize the processor."""
        super().__init__(model_id=model_id)
        hf_processor = AutoProcessor.from_pretrained(model_id)
        self.image_processor = hf_processor.image_processor

    def get_image_processor(self) -> Callable | None:
        """Return the image processor."""

        def processor(image: npt.NDArray) -> dict[str, torch.Tensor]:
            """Invoke the HF processor and convert to dict."""
            # If we enable Pan & Scan, the batch dimension (0) may be larger than 1.
            return self.image_processor(images=[image], return_tensors="pt").data

        return processor
