from functools import partial
from typing import Callable

import torch
import numpy.typing as npt
from torch import nn
from einops import rearrange
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniConfig
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig
from flash_attn import flash_attn_varlen_func
from flash_attn.layers.rotary import apply_rotary_emb

from . import qwen2_5_vl
from .base import EricModel
from .layers.linear import RowParallelLinear, QKVParallelLinear
from cornserve.task_executors.eric.api import Modality
from cornserve.task_executors.eric.distributed import parallel
from cornserve.task_executors.eric.router.processor import BaseModalityProcessor
from cornserve.task_executors.eric.utils import distributed as dist_utils


def apply_rotary_pos_emb_vision(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    t_ = t.float()
    cos = freqs.cos()
    sin = freqs.sin()

    output = apply_rotary_emb(t_, cos, sin).type_as(t)

    return output


class Qwen2_5_VisionAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        projection_size: int,
    ) -> None:
        super().__init__()
        # Per attention head and per partition values.
        self.tp_group = parallel.get_tensor_parallel_group()
        self.tp_size = self.tp_group.world_size
        self.tp_rank = self.tp_group.rank
        self.hidden_size_per_attention_head = dist_utils.divide(projection_size, num_heads)
        self.num_attention_heads_per_partition = dist_utils.divide(num_heads, self.tp_size)
        self.embed_dim = embed_dim

        # self.qkv = ColumnParallelLinear(input_size=embed_dim, output_size=3 * projection_size)
        self.qkv = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.hidden_size_per_attention_head,
            total_num_heads=num_heads,
            total_num_kv_heads=num_heads,
            bias=True,
            gather_from_names=("q", "k", "v"),
        )
        self.proj = RowParallelLinear(input_size=projection_size, output_size=embed_dim)

    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # [s, b, 3 * head * head_dim]
        seq_len, bs, _ = qkv.shape
        if self.tp_size > 1:
            # qkv = all_gather_interleave(qkv, self.embed_dim, self.tp_size)
            qkv = self.tp_group.all_gather(qkv)

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head * head_dim]
        q, k, v = qkv.chunk(3, dim=2)

        # 3 * [s, b, head * head_dim]
        if self.tp_size > 1:
            splitter = partial(dist_utils.split_tensor_along_last_dim, num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
            v = splitter(v)[self.tp_rank]

        # 3 * [s, b, head * head_dim] -> 3 * [s, b, head, head_dim]
        new_shape = (seq_len, bs, self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
        q, k, v = (x.view(*new_shape) for x in (q, k, v))
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        max_seqlen: int | None = None,  # Only used for Flash Attention
    ) -> torch.Tensor:
        # [s, b, c] --> [s, b, head * 3 * head_dim]
        x, _ = self.qkv(x)

        # [s, b, 3 * head * head_dim] -> 3 * [s, b, head, head_dim]
        q, k, v = self.split_qkv(x)
        batch_size = q.shape[1]

        q, k, v = (rearrange(x, "s b ... -> b s ...").contiguous() for x in (q, k, v))
        if rotary_pos_emb is not None:
            q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
            k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        # from vllm_flash_attn.flash_attn_interface import (
        #   flash_attn_varlen_func)

        q, k, v = (rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])

        output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=0,
            causal=False,
        )

        context_layer = rearrange(output, "(b s) ... -> b s ...", b=batch_size)
        context_layer = rearrange(context_layer, "b s h d -> s b (h d)").contiguous()

        output, _ = self.proj(context_layer)
        return output


# Patch the Qwen2_5_VisionAttention class in the qwen2_5_vl module.
# The only difference is that this one uses `QKVParallelLinear` instead of `ColumnParallelLinear`,
# because the Omni model's checkpoint saved Q, K, nad V separately.
qwen2_5_vl.Qwen2_5_VisionAttention = Qwen2_5_VisionAttention


class Qwen2_5OmniEncoder(EricModel):
    def __init__(self, config: Qwen2_5OmniConfig) -> None:
        super().__init__()

        self.config = config

        vision_config = Qwen2_5_VLConfig()
        vision_config.vision_config = Qwen2_5_VLVisionConfig(
            **config.thinker_config.vision_config.to_dict(),
        )
        vision_config.rms_norm_eps = getattr(config.thinker_config.text_config, "rms_norm_eps", 1e-6)
        self.visual = qwen2_5_vl.Qwen2_5_VisionTransformer(vision_config)

        audio_config = config.thinker_config.audio_config
        audio_config._attn_implementation_autoset = True
        audio_config._attn_implementation = "flash_attention_2"
        #
        # from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
        # if parallel.get_tensor_parallel_group().world_size > 1:
        #     self.audio_tower = parallelize_module(
        #         Qwen2_5OmniAudioEncoder(audio_config),
        #         device_mesh=parallel.get_tensor_parallel_group().device_mesh,
        #         parallelize_plan={
        #             "layers.*.self_attn.q": ColwiseParallel(),
        #             "layers.*.self_attn.k": ColwiseParallel(),
        #             "layers.*.self_attn.v": ColwiseParallel(),
        #             "layers.*.self_attn.proj": RowwiseParallel(),
        #             "layers.*.mlp.gate_proj": RowwiseParallel(),
        #             "layers.*.mlp.up_proj": RowwiseParallel(),
        #             "layers.*.mlp.down_proj": RowwiseParallel(),
        #         },
        #     )
        # else:
        self.audio_tower = Qwen2_5OmniAudioEncoder(audio_config)

    @property
    def dtype(self) -> torch.dtype:
        return self.visual.dtype

    @property
    def device(self) -> torch.device:
        return self.visual.device

    @property
    def chunk_shape(self) -> tuple[int, ...]:
        return (1, self.visual.out_hidden_size)

    def forward(
        self,
        modality: Modality,
        batch: dict[str, list[torch.Tensor]],
    ) -> list[torch.Tensor]:
        """Forward pass of the model.

        For images, `batch` is expected to have the following keys:
        - `pixel_values`: The pixel values of the images. Each [seq_len, 6 * patch_size (14) * patch_size (14)].
        - `image_grid_thw`: The grid size of the images, Each [1, 3].

        For videos, `batch` is expected to have the following keys:
        - `pixel_values_videos`: The pixel values of the videos. Each [seq_len, 6 * patch_size (14) * patch_size (14)].
        - `video_grid_thw`: The grid size of the videos, Each [1, 3].

        For audios, `batch` is expected to have the following keys:
        - `input_audio_features`: The audio Mel spectrogram features. Each [feature_size (128), seq_len].
        - `audio_feature_lengths`: The lengths of the audio features. Each [1,].
        """
        # Batch
        match modality:
            case Modality.IMAGE | Modality.VIDEO:
                return self.visual(modality, batch)
            case Modality.AUDIO:
                input_features = torch.cat(batch["input_audio_features"], dim=1).to(
                    device=self.device, dtype=self.dtype
                )
                audio_feature_lengths = torch.cat(batch["audio_feature_lengths"], dim=0).to(device=self.device)
                aftercnn_lengths, audio_output_lengths = self.audio_tower._get_feat_extract_output_lengths(
                    audio_feature_lengths,
                )
                audio_features = self.audio_tower(
                    input_features,
                    feature_lens=audio_feature_lengths,
                    aftercnn_lens=aftercnn_lengths,
                ).last_hidden_state
                return audio_features.split(audio_output_lengths.tolist())
            case _:
                raise ValueError(f"Unsupported modality: {modality}")


class ModalityProcessor(BaseModalityProcessor):
    """Qwen2.5-Omni modality processor."""

    def __init__(self, model_id: str) -> None:
        """Initialize the processor."""
        super().__init__(model_id=model_id)
        self.hf_processor = AutoProcessor.from_pretrained(model_id)

    def get_image_processor(self) -> Callable | None:
        """Return the image processor."""

        def processor(image: npt.NDArray) -> dict[str, npt.NDArray]:
            """Invoke the HF processor and convert to dict."""
            return self.hf_processor.image_processor(images=[image], videos=None, return_tensors="np").data

        return processor

    def get_audio_processor(self) -> Callable | None:
        """Return the audio processor."""

        def processor(audio: npt.NDArray) -> dict[str, npt.NDArray]:
            """Invoke the HF processor and convert to dict."""
            data = self.hf_processor.feature_extractor(
                [audio],
                padding="max_length",
                sampling_rate=self.hf_processor.feature_extractor.sampling_rate,
                return_attention_mask=True,
                return_tensors="pt",
            ).data

            input_features = data.pop("input_features")
            attention_mask = data.pop("attention_mask")
            input_features = input_features.permute(0, 2, 1)[attention_mask.bool()].permute(1, 0)
            return dict(
                input_audio_features=input_features.numpy(),
                audio_feature_lengths=attention_mask.sum(-1).numpy(),
                feature_attention_mask=attention_mask.numpy(),
            )

        return processor

    def get_video_processor(self) -> Callable | None:
        """Return the video processor."""

        def processor(video: npt.NDArray) -> dict[str, npt.NDArray]:
            """Invoke the HF processor and convert to dict."""
            # TODO: Some models (e.g., Qwen 2 VL, QWen 2.5 VL, Qwen 2.5 Omni) support passing `min_pixels` and
            #       `max_pixel` to the imgae and video processors. See vLLM's VLM offline inference example.
            #       In general, we should be able to pass in arbitrary processor-specific kwargs via requests
            #       and fallback to model-specific defaults if not provided.
            #       The defaults below were taken from HF Transformers `Qwen2_5OmniProcessorKwargs_defaults`.
            data = self.hf_processor.video_processor(
                videos=[video],
                min_pixels=128 * 28 * 28,
                max_pixels=768 * 28 * 28,
                return_tensors="pt",
            ).data
            return {k: v.numpy() for k, v in data.items()}

        return processor
