import math
from typing import Callable

import torch
import torch.nn as nn
import numpy.typing as npt
from transformers import AutoProcessor
from transformers.models.llava_onevision.configuration_llava_onevision import (
    LlavaOnevisionConfig,
)
from transformers.models.llava_onevision.modeling_llava_onevision import (
    get_anyres_image_grid_shape,
    unpad_image,
)


from .base import EricModel
from .layers.activations import get_act_fn
from .layers.siglip import SiglipVisionModel
from .layers.vit import init_vision_tower_for_llava
from cornserve.task_executors.eric.schema import Modality
from cornserve.task_executors.eric.distributed import parallel
from cornserve.task_executors.eric.router.processor import BaseModalityProcessor
from cornserve.task_executors.eric.utils import distributed as dist_utils


class LlavaOnevisionMultiModalProjector(nn.Module):

    def __init__(self, config: LlavaOnevisionConfig):
        super().__init__()

        self.linear_1 = nn.Linear(
            config.vision_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )
        self.act = get_act_fn(config.projector_hidden_act)
        self.linear_2 = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.hidden_size,
            bias=config.multimodal_projector_bias,
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class LlavaOnevisionEncoder(EricModel):

    def __init__(self, *, config: LlavaOnevisionConfig) -> None:
        super().__init__()
        self.config = config

        # Initialize the vision tower only up to the required feature layer
        self.vision_tower = init_vision_tower_for_llava(config, require_post_norm=False)

        self.multi_modal_projector = LlavaOnevisionMultiModalProjector(config)
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

    def _validate_image_sizes(self, data: torch.Tensor) -> torch.Tensor:
        expected_dims = (2,)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape)

            if actual_dims != expected_dims:
                expected_expr = str(expected_dims)
                raise ValueError(
                    f"The expected shape of image sizes per image per batch "
                    f"is {expected_expr}. You supplied {tuple(d.shape)}."
                )

        for d in data:
            _validate_shape(d)

        return data

    def _validate_image_pixel_values(
        self, data: torch.Tensor | list[torch.Tensor]
    ) -> torch.Tensor | list[torch.Tensor]:

        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)

        def _validate_shape(d: torch.Tensor):
            actual_dims = tuple(d.shape[1:])

            if actual_dims != expected_dims:
                expected_expr = ("num_patches", *map(str, expected_dims))
                raise ValueError(
                    "The expected shape of pixel values per image per batch "
                    f"is {expected_expr}. You supplied {tuple(d.shape)}."
                )

        for d in data:
            _validate_shape(d)

        return data

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> LlavaOnevisionImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of pixel values. " f"Got type: {type(pixel_values)}"
                )

            if not isinstance(image_sizes, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of image sizes. " f"Got type: {type(image_sizes)}"
                )

            return LlavaOnevisionImagePixelInputs(
                type="pixel_values",
                data=self._validate_image_pixel_values(flatten_bn(pixel_values)),
                image_sizes=self._validate_image_sizes(
                    flatten_bn(image_sizes, concat=True)
                ),
            )

        if image_embeds is not None:
            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError(
                    "Incorrect type of image embeds. " f"Got type: {type(image_embeds)}"
                )

            return LlavaOnevisionImageEmbeddingInputs(
                type="image_embeds",
                data=flatten_bn(image_embeds),
            )

        raise AssertionError("This line should be unreachable.")

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> Optional[LlavaOnevisionVideoPixelInputs]:
        """
        A legal video input should have the following dimensions:
        {
            "pixel_values_videos" :
                List[b, Tensor(nb_frames, nb_channels, height, width)]
        }
        """
        pixel_values = kwargs.pop("pixel_values_videos", None)

        if pixel_values is None:
            return None

        if not (
            is_list_of(pixel_values, (torch.Tensor))  # different shape videos
            or isinstance(pixel_values, torch.Tensor)
        ):  # same shape videos
            raise ValueError(
                "Incorrect type of pixel values. " f"Got type: {type(pixel_values)}"
            )

        return LlavaOnevisionVideoPixelInputs(
            type="pixel_values_videos",
            data=pixel_values,
        )

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        modalities = {}

        # Preserve the order of modalities if there are multiple of them
        # from the order of kwargs.
        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "images" not in modalities
            ):
                modalities["images"] = self._parse_and_validate_image_input(**kwargs)
            if (
                input_key in ("pixel_values_videos", "video_embeds")
                and "videos" not in modalities
            ):
                modalities["videos"] = self._parse_and_validate_video_input(**kwargs)

        return modalities

    def _select_image_features(
        self, image_features: torch.Tensor, *, strategy: str
    ) -> torch.Tensor:
        if strategy == "default":
            return image_features[:, 1:]
        elif strategy == "full":
            return image_features

        raise ValueError(f"Unexpected select feature strategy: {strategy}")

    def _image_pixels_to_features(
        self,
        vision_tower: Union[CLIPVisionModel, SiglipVisionModel],
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        image_features = vision_tower(pixel_values)
        return self._select_image_features(
            image_features,
            strategy=self.config.vision_feature_select_strategy,
        )

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
            height = width = (
                self.config.vision_config.image_size
                // self.config.vision_config.patch_size
            )

            base_patch_embeds = patch_embeddings[0]
            if height * width != base_patch_embeds.shape[0]:
                raise ValueError(
                    "The number of patches is not consistent with the " "image size."
                )

            if patch_embeddings.shape[0] > 1:
                other_patch_embeds = patch_embeddings[1:]

                # Move to CPU to avoid floating-point errors
                orig_height, orig_width = image_size.tolist()

                # image_aspect_ratio == "anyres"
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
                        other_patch_embeds.permute(4, 0, 2, 1, 3)
                        .contiguous()
                        .flatten(1, 2)
                        .flatten(2, 3)
                    )
                    other_patch_embeds = unpad_image(
                        other_patch_embeds, (orig_height, orig_width)
                    )
                    max_num_patches = int(
                        vision_aspect_ratio.removeprefix("anyres_max_")
                    )
                    channels, curr_height, curr_width = other_patch_embeds.shape
                    ratio = math.sqrt(
                        curr_height * curr_width / (max_num_patches * height**2)
                    )
                    if ratio > 1.1:
                        other_patch_embeds = other_patch_embeds[None]
                        other_patch_embeds = nn.functional.interpolate(
                            other_patch_embeds,
                            [int(curr_height // ratio), int(curr_width // ratio)],
                            mode="bilinear",
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
                    other_patch_embeds = other_patch_embeds.flatten(1, 2).transpose(
                        0, 1
                    )
                else:
                    other_patch_embeds = (
                        other_patch_embeds.permute(0, 2, 1, 3, 4)
                        .contiguous()
                        .flatten(0, 3)
                    )

                merged_patch_embeddings = torch.cat(
                    (base_patch_embeds, other_patch_embeds), dim=0
                )
            else:
                if "unpad" in strategy:
                    merged_patch_embeddings = torch.cat(
                        (
                            base_patch_embeds,
                            self.image_newline[None].to(base_patch_embeds.device),
                        ),
                        dim=0,
                    )
                else:
                    merged_patch_embeddings = base_patch_embeds

            return merged_patch_embeddings

        raise ValueError(f"Unexpected patch merge strategy: {strategy}")

    def _process_image_pixels(
        self,
        inputs: LlavaOnevisionImagePixelInputs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        assert self.vision_tower is not None

        pixel_values = inputs["data"]

        if isinstance(pixel_values, torch.Tensor):
            b, num_patches, c, h, w = pixel_values.shape
            stacked_pixel_values = pixel_values.view(b * num_patches, c, h, w)
            stacked_image_features = self._image_pixels_to_features(
                self.vision_tower, stacked_pixel_values
            )
            stacked_patch_embeddings = self.multi_modal_projector(
                stacked_image_features
            )

            return stacked_patch_embeddings.view(
                b, num_patches, *stacked_patch_embeddings.shape[1:]
            )

        num_patches_per_batch = [v.shape[0] for v in pixel_values]
        stacked_pixel_values = torch.cat(pixel_values)
        stacked_image_features = self._image_pixels_to_features(
            self.vision_tower, stacked_pixel_values
        )

        return [
            self.multi_modal_projector(image_features)
            for image_features in torch.split(
                stacked_image_features, num_patches_per_batch
            )
        ]

    def _process_image_input(
        self,
        image_input: LlavaOnevisionImageInputs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if image_input["type"] == "image_embeds":
            return [image_input["data"]]

        patch_embeddings = self._process_image_pixels(image_input)

        image_sizes = image_input.get("image_sizes")
        if image_sizes is None:
            batch_size = len(image_input["data"])
            vision_config = self.config.vision_config
            default_height = default_width = vision_config.image_size
            image_sizes = torch.as_tensor(
                [[default_height, default_width] for _ in range(batch_size)]
            )

        return [
            self._merge_image_patch_embeddings(
                image_sizes[i],
                patch_features_batch,
                image_newline=self.image_newline,
                strategy="spatial_unpad",
            )
            for i, patch_features_batch in enumerate(patch_embeddings)
        ]

    def _add_image_newline(
        self,
        video_features: torch.Tensor,
        videos: int = 1,
        frames: int = 1,
        strategy: str = "one_token",
    ) -> torch.Tensor:
        if strategy == "one_token":
            video_features = video_features.reshape(
                videos, frames * video_features.shape[1], -1
            )
            image_newline = (
                self.image_newline[None, None, :]
                .repeat(videos, 1, 1)
                .to(video_features.device)
            )
            video_features = torch.cat((video_features, image_newline), dim=1)
            return video_features
        raise ValueError(f"Unexpected video newline strategy: {strategy}")

    def _video_pixels_to_features(
        self,
        vision_tower: Union[CLIPVisionModel, SiglipVisionModel],
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        video_features = vision_tower(pixel_values)
        video_features = self._select_image_features(
            video_features,
            strategy=self.config.vision_feature_select_strategy,
        )
        video_features = self.multi_modal_projector(video_features)
        video_features = self.apply_pooling(video_features)
        return video_features

    def _process_video_pixels(self, inputs: LlavaOnevisionVideoPixelInputs):
        assert self.vision_tower is not None

        video_pixels = inputs["data"]

        if isinstance(video_pixels, torch.Tensor):
            b, num_videos, frames, c, h, w = video_pixels.shape
            pixel_values = video_pixels.view(b * num_videos * frames, c, h, w)
            stacked_embeddings = self._video_pixels_to_features(
                self.vision_tower, pixel_values
            )
            stacked_embeddings = self._add_image_newline(
                stacked_embeddings,
                videos=b * num_videos,
                frames=frames,
                strategy="one_token",
            )
            return stacked_embeddings
        elif is_list_of(video_pixels, torch.Tensor):
            stacked_embeddings = []
            for video_pixel in video_pixels:
                num_videos, frames, c, h, w = video_pixel.shape
                pixel_values = video_pixel.view(num_videos * frames, c, h, w)
                embeddings = self._video_pixels_to_features(
                    self.vision_tower, pixel_values
                )
                embeddings = self._add_image_newline(
                    embeddings, videos=num_videos, frames=frames, strategy="one_token"
                )
                stacked_embeddings.append(embeddings)
            return stacked_embeddings
        else:
            raise ValueError(f"Unsupported type of video input {type(video_pixels)}")

    def apply_pooling(self, image_features, stride=2):
        vision_config = self.config.vision_config
        height = width = vision_config.image_size // vision_config.patch_size
        batch_frames, _, dim = image_features.shape
        image_features = image_features.view(batch_frames, height, width, -1)
        image_features = image_features.permute(0, 3, 1, 2)

        # TODO support other pooling types config
        height, width = image_features.shape[2:]
        scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
        image_feature = nn.functional.interpolate(
            image_features, size=scaled_shape, mode="bilinear"
        )
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(batch_frames, -1, dim)
        return image_feature

    def get_multimodal_embeddings(self, **kwargs) -> Optional[tuple[torch.Tensor, ...]]:
        modalities = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not modalities:
            return None

        # The result multimodal_embeddings is tuple of tensors, with each
        # tensor correspoending to a multimodal data item (image or video).
        multimodal_embeddings: tuple[torch.Tensor, ...] = ()

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in modalities:
            if modality == "images":
                image_input = modalities["images"]
                vision_embeddings = self._process_image_input(image_input)
                multimodal_embeddings += tuple(vision_embeddings)
            if modality == "videos":
                video_input = modalities["videos"]
                video_embeddings = self._process_video_pixels(video_input)
                multimodal_embeddings += tuple(video_embeddings)

        return multimodal_embeddings

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[tuple[torch.Tensor, ...]] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.language_model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                [self.config.image_token_index, self.config.video_token_index],
            )
        return inputs_embeds

    def get_input_embeddings_v0(
        self,
        input_ids: torch.Tensor,
        image_input: Optional[NestedTensors] = None,
        video_input: Optional[NestedTensors] = None,
    ) -> torch.Tensor:

        inputs_embeds = self.get_input_embeddings(input_ids)
        if image_input is not None:
            image_embeds = self._process_image_input(image_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                image_embeds,
                placeholder_token_id=self.config.image_token_index,
            )

        if video_input is not None:
            video_embeds = self._process_video_pixels(video_input)
            inputs_embeds = merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                video_embeds,
                placeholder_token_id=self.config.video_token_index,
            )

        return inputs_embeds

    def old_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Run forward pass for LlaVA-Onevision.
        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            pixel_values_videos: Pixels in each frames for each input videos.
        """
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner from
        # `get_multimodal_embeddings` and `get_input_embeddings`, this
        # condition is only for v0 compatibility.
        elif inputs_embeds is None:
            image_input = self._parse_and_validate_image_input(**kwargs)
            video_input = self._parse_and_validate_video_input(**kwargs)

            if image_input is None and video_input is None:
                inputs_embeds = None
            else:
                inputs_embeds = self.get_input_embeddings_v0(
                    input_ids, image_input=image_input, video_input=video_input
                )
                input_ids = None

        hidden_states = self.language_model.model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

        return hidden_states

    def forward(
        self,
        modality: Modality,
        batch: dict[str, list[torch.Tensor]],
    ) -> list[torch.Tensor]:
        """Forward pass of the model.

        For images, `batch` is expected to have the following keys:
        - `pixel_values`: The pixel values of the images. Each [seq_len, 6 * patch_size (14) * patch_size (14)].
        - `image_sizes`: The height and width of the images. Each [1, 2].

        For videos, `batch` is expected to have the following keys:
        - `pixel_values_videos`: The pixel values of the videos. Each [seq_len, 6 * patch_size (14) * patch_size (14)].
        """
        # Batch
        match modality:
            case Modality.IMAGE:
                pixel_values = torch.cat(batch["pixel_values"], dim=0).to(
                    device=self.device, dtype=self.dtype
                )
                grid_thw = torch.cat(batch["image_grid_thw"], dim=0).to(
                    device=self.device
                )
            case Modality.VIDEO:
                pixel_values = torch.cat(batch["pixel_values_videos"], dim=0).to(
                    device=self.device, dtype=self.dtype
                )
                grid_thw = torch.cat(batch["video_grid_thw"], dim=0).to(
                    device=self.device
                )
            case _:
                raise ValueError(f"Unsupported modality: {modality}.")

        # Unbatch

        return result


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
            return out.data.squeeze(0)

        return processor

    def get_video_processor(self) -> Callable | None:
        """Return the video processor."""

        def processor(video: npt.NDArray) -> dict[str, npt.NDArray]:
            """Invoke the HF processor and convert to dict."""
            out = self.video_processor.preprocess(videos=[video], return_tensors="np")
            return out.data.squeeze(0)

        return processor
