import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
from einops import rearrange
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniAudioEncoderConfig, Qwen2_5OmniConfig
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig
from cornserve.task_executors.eric.models.layers.activations import get_act_fn
from flash_attn import flash_attn_varlen_func
from flash_attn.layers.rotary import apply_rotary_emb

from . import qwen2_5_vl
from .base import EricModel
from .layers.linear import ColumnParallelLinear, RowParallelLinear, QKVParallelLinear
from cornserve.task_executors.eric.api import Modality
from cornserve.task_executors.eric.distributed import parallel
from cornserve.task_executors.eric.router.processor import BaseModalityProcessor
from cornserve.task_executors.eric.utils import distributed as dist_utils


class Qwen2_5OmniAudioFlashAttention2(nn.Module):
    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig) -> None:
        super().__init__()
        self.tp_group = parallel.get_tensor_parallel_group()
        self.tp_size = self.tp_group.world_size
        self.tp_rank = self.tp_group.rank

        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = dist_utils.divide(self.embed_dim, self.num_heads)
        self.num_heads_per_partition = dist_utils.divide(config.encoder_attention_heads, self.tp_size)

        self.is_causal = False

        self.k_proj = ColumnParallelLinear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = ColumnParallelLinear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = ColumnParallelLinear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = RowParallelLinear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ):
        seq_length, all_dim = hidden_states.size()
        query_states = self.q_proj(hidden_states)[0]
        query_states = query_states.reshape(seq_length, self.num_heads_per_partition, self.head_dim)

        key_states = self.k_proj(hidden_states)[0]
        key_states = key_states.reshape(seq_length, self.num_heads_per_partition, self.head_dim)
        value_states = self.v_proj(hidden_states)[0]
        value_states = value_states.reshape(seq_length, self.num_heads_per_partition, self.head_dim)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(
            query_states, key_states, value_states, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, dropout_p=0.0
        )
        attn_output = attn_output.reshape(seq_length, dist_utils.divide(all_dim, self.tp_size))
        attn_output = self.out_proj(attn_output)[0]

        return attn_output


class Qwen2_5OmniAudioEncoderLayer(nn.Module):
    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Qwen2_5OmniAudioFlashAttention2(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = get_act_fn(config.activation_function)
        self.activation_dropout = config.activation_dropout
        self.fc1 = ColumnParallelLinear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = RowParallelLinear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)[0]
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)[0]
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        return outputs


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


class Qwen2_5OmniAudioEncoder(nn.Module):
    def __init__(self, config: Qwen2_5OmniAudioEncoderConfig) -> None:
        super().__init__()
        self.config = config

        self.dropout = config.dropout

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.conv1 = nn.Conv1d(self.num_mel_bins, embed_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.positional_embedding = SinusoidsPositionEmbedding(self.max_source_positions, embed_dim)
        self.audio_bos_eos_token = nn.Embedding(2, config.output_dim)
        self.layers = nn.ModuleList([Qwen2_5OmniAudioEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.ln_post = nn.LayerNorm(config.d_model)
        self.avg_pooler = nn.AvgPool1d(2, stride=2)
        self.proj = nn.Linear(config.d_model, config.output_dim)

    def forward(self, input_features, feature_lens, aftercnn_lens):
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths = torch.where(chunk_lengths == 0, self.n_window * 2, chunk_lengths)

        chunk_list = input_features.split(chunk_lengths.tolist(), dim=1)
        padded_feature, padded_mask, padded_mask_after_cnn = self.padded_and_mask_function(
            chunk_list, chunk_lengths, padding_value=0, padding_side="right"
        )
        padded_embed = nn.functional.gelu(self.conv1(padded_feature)) * padded_mask
        padded_embed = nn.functional.gelu(self.conv2(padded_embed)).transpose(1, 2)

        padded_embed = padded_embed + self.positional_embedding.positional_embedding[
            : padded_embed.shape[1], :
        ].unsqueeze(0).to(padded_embed.dtype)
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_seqlens = torch.cat(
            (
                torch.zeros(1, device=padded_mask_after_cnn.device, dtype=torch.int32),
                padded_mask_after_cnn.sum(1).cumsum(0),
            )
        ).to(torch.int32)

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
            )
            hidden_states = layer_outputs[0]

        hidden_states_list = hidden_states.split(aftercnn_lens.tolist(), dim=0)
        token_audio_list = []
        for each_audio_states in hidden_states_list:
            each_audio_states = self.avg_pooler(each_audio_states.transpose(0, 1)).transpose_(0, 1)
            each_audio_states = self.ln_post(each_audio_states)
            each_audio_states = self.proj(each_audio_states)
            token_audio_list.append(each_audio_states)
        return torch.cat(token_audio_list, dim=0)

    def padded_and_mask_function(self, tensor_list, tensor_len, padding_value=0, padding_side="right"):
        """
        Pads a sequence of tensors to their maximum length on indicated `padding_side`.
        Then prepares a mask so that pad tokens are not attended to.
        """
        max_len = tensor_len.max()
        dim = tensor_list[0].shape[0]
        padded_tensor = torch.full(
            size=(len(tensor_list), dim, max_len),
            fill_value=padding_value,
            dtype=self.conv1.weight.dtype,
            device=tensor_list[0].device,
        )

        batch_mask = torch.zeros(
            (len(tensor_len), max_len),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(tensor_len):
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :length] = tensor_list[i]

        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max()
        batch_mask_after_cnn = torch.zeros(
            (len(tensor_len), max_len_after_cnn),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = 1
        return (
            padded_tensor,
            batch_mask.unsqueeze(1),
            batch_mask_after_cnn.bool(),
        )

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths


def apply_rotary_pos_emb_vision(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    t_ = t.float()
    cos = freqs.cos()
    sin = freqs.sin()

    output = apply_rotary_emb(t_, cos, sin).type_as(t)

    return output


def all_gather_interleave(local_tensor, hidden_size: int, tp_size: int):
    """All-gather the input tensor interleavely across model parallel group."""
    import torch.distributed as dist

    gathered_tensors = [torch.zeros_like(local_tensor) for _ in range(tp_size)]
    dist.all_gather(gathered_tensors, local_tensor, group=parallel.get_tensor_parallel_group().process_group)

    gathered_tensors_split = [torch.split(tensor, hidden_size // tp_size, -1) for tensor in gathered_tensors]
    ordered_tensors = [tensor for pair in zip(*gathered_tensors_split) for tensor in pair]
    result_tensor = torch.cat(ordered_tensors, dim=-1)
    return result_tensor


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
            qkv = all_gather_interleave(qkv, self.embed_dim, self.tp_size)

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
                )
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
