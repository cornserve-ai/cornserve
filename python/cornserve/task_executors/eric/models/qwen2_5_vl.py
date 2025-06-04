"""Qwen2.5-VL Vision Transformer."""

from functools import partial
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy.typing as npt
from einops import rearrange
from transformers.models.auto.image_processing_auto import AutoImageProcessor
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from flash_attn import flash_attn_varlen_func
from flash_attn.layers.rotary import apply_rotary_emb

from .base import EricModel
from .layers.norm import RMSNorm
from .layers.activations import get_act_fn
from .layers.linear import ColumnParallelLinear, RowParallelLinear
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


def cast_overflow_tensors(
    tensors: torch.Tensor,
    offset: float = 1000,
) -> torch.Tensor:
    if tensors.isinf().any() or tensors.isnan().any():
        clamp_value = torch.finfo(tensors.dtype).max - offset
        tensors = torch.clamp(tensors, min=-clamp_value, max=clamp_value)
    return tensors


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

        self.qkv = ColumnParallelLinear(input_size=embed_dim, output_size=3 * projection_size)
        self.proj = RowParallelLinear(input_size=projection_size, output_size=embed_dim)

    def split_qkv(self, qkv: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # [s, b, 3 * head * head_dim]
        seq_len, bs, _ = qkv.shape
        if self.tp_size > 1:
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


class Qwen2_5_VisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = False,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
    ) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(in_features, hidden_features, bias=bias)
        self.up_proj = ColumnParallelLinear(in_features, hidden_features, bias=bias)
        self.down_proj = RowParallelLinear(hidden_features, in_features, bias=bias)
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor):
        x_gate, _ = self.gate_proj(x)
        x_gate = self.act_fn(x_gate)
        x_up, _ = self.up_proj(x)
        x_down, _ = self.down_proj(x_gate * x_up)
        return x_down


class Qwen2_5_VisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        norm_layer: Callable[[int], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Qwen2_5_VisionAttention(embed_dim=dim, num_heads=num_heads, projection_size=dim)
        self.mlp = Qwen2_5_VisionMLP(dim, mlp_hidden_dim, act_fn=act_fn, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb, max_seqlen=max_seqlen)

        x = x + self.mlp(self.norm2(x))
        return x


class Qwen2_5_VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_channels, hidden_size, kernel_size=kernel_size, stride=kernel_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, C = x.shape
        x = x.view(L, -1, self.temporal_patch_size, self.patch_size, self.patch_size)
        x = self.proj(x).view(L, self.hidden_size)
        return x


class Qwen2_5_VisionPatchMerger(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Callable[[int], nn.Module] | None = None,
        spatial_merge_size: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.ln_q = norm_layer(context_dim)
        self.mlp = nn.ModuleList(
            [
                ColumnParallelLinear(self.hidden_size, self.hidden_size, bias=True),
                nn.GELU(),
                RowParallelLinear(self.hidden_size, d_model, bias=True),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln_q(x)
        x = x.view(-1, self.hidden_size)

        mlp_fc1, mlp_act, mlp_fc2 = self.mlp
        x_parallel, _ = mlp_fc1(x)
        x_parallel = mlp_act(x_parallel)
        out, _ = mlp_fc2(x_parallel)
        return out


class Qwen2_5_VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._freqs_cached: torch.Tensor = None  # type: ignore

    def update_freqs_cache(self, seqlen: int) -> None:
        if seqlen > self._seq_len_cached:
            seqlen *= 2
            self._seq_len_cached = seqlen
            self.inv_freq = 1.0 / (
                self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float, device=self.inv_freq.device) / self.dim)
            )
            seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(seq, self.inv_freq)
            self._freqs_cached = freqs

    def forward(self, seqlen: int) -> torch.Tensor:
        self.update_freqs_cache(seqlen)
        return self._freqs_cached[:seqlen]


class Qwen2_5_VisionTransformer(EricModel):
    def __init__(self, config: Qwen2_5_VLConfig) -> None:
        super().__init__()
        vision_config = config.vision_config
        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        in_channels = vision_config.in_channels
        depth = vision_config.depth

        self.hidden_size = vision_config.hidden_size
        self.out_hidden_size = vision_config.out_hidden_size
        self.num_heads = vision_config.num_heads

        # args for get_window_index
        self.window_size = vision_config.window_size
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.fullatt_block_indexes = vision_config.fullatt_block_indexes
        self.spatial_merge_unit = self.spatial_merge_size**2

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=self.hidden_size,
        )

        norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        norm_layer = partial(RMSNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                Qwen2_5_VisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=vision_config.intermediate_size,
                    act_fn=get_act_fn(vision_config.hidden_act),
                    norm_layer=norm_layer,
                )
                for layer_idx in range(depth)
            ]
        )
        self.merger = Qwen2_5_VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            norm_layer=norm_layer,
            spatial_merge_size=self.spatial_merge_size,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    @property
    def chunk_shape(self) -> tuple[int, ...]:
        return (1, self.out_hidden_size)

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            hpos_ids = (
                hpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            wpos_ids = (
                wpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def get_window_index(self, grid_thw):
        window_index: list = []
        cu_window_seqlens: list = [0]
        window_index_id = 0
        vit_merger_window_size = self.window_size // self.spatial_merge_size // self.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            llm_grid_h = grid_h // self.spatial_merge_size
            llm_grid_w = grid_w // self.spatial_merge_size
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t, num_windows_h, vit_merger_window_size, num_windows_w, vit_merger_window_size
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t, num_windows_h * num_windows_w, vit_merger_window_size, vit_merger_window_size
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_unit + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()
        window_index = torch.cat(window_index, dim=0)
        return window_index, cu_window_seqlens

    def compute_attn_mask_seqlen(
        self,
        cu_seqlens: torch.Tensor,
    ) -> tuple[int | None, list[int] | None]:
        max_seqlen, seqlens = None, None
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        return max_seqlen, seqlens

    def forward(
        self,
        # x: torch.Tensor,
        # grid_thw: torch.Tensor,
        modality: Modality,
        batch: dict[str, list[torch.Tensor]],
    ) -> list[torch.Tensor]:
        """Forward pass of the model.

        For images, `batch` is expected to have the following keys:
        - `pixel_values`: The pixel values of the images. Each [seq_len, 6 * patch_size (14) * patch_size (14)].
        - `image_grid_thw`: The grid size of the images. Each [1, 3].

        For videos, `batch` is expected to have the following keys:
        - `pixel_values_videos`: The pixel values of the videos. Each [seq_len, 6 * patch_size (14) * patch_size (14)].
        - `video_grid_thw`: The grid size of the videos. Each [1, 3].
        """
        # Batch
        match modality:
            case Modality.IMAGE:
                pixel_values = torch.cat(batch["pixel_values"], dim=0).to(device=self.device, dtype=self.dtype)
                grid_thw = torch.cat(batch["image_grid_thw"], dim=0).to(device=self.device)
            case Modality.VIDEO:
                pixel_values = torch.cat(batch["pixel_values_videos"], dim=0).to(device=self.device, dtype=self.dtype)
                grid_thw = torch.cat(batch["video_grid_thw"], dim=0).to(device=self.device)
            case _:
                raise ValueError(f"Unsupported modality: {modality}")

        # patchify
        hidden_states = self.patch_embed(pixel_values)

        # compute position embedding
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # windows attention
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=hidden_states.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        # compute cu_seqlens
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)

        # transformers
        hidden_states = hidden_states.unsqueeze(1)

        # pre-compute seqlens for window/full attn to reduce cuMemcpy operations
        max_seqlen_full, seqlens_full = self.compute_attn_mask_seqlen(cu_seqlens)
        max_seqlen_window, seqlens_window = self.compute_attn_mask_seqlen(cu_window_seqlens)
        for layer_num, blk in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
                max_seqlen_now = max_seqlen_full
            else:
                cu_seqlens_now = cu_window_seqlens
                max_seqlen_now = max_seqlen_window

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                rotary_pos_emb=rotary_pos_emb,
                max_seqlen=max_seqlen_now,
            )

        # For Qwen2.5-VL-3B, float16 will overflow at last block
        # for long visual tokens sequences.
        if hidden_states.dtype == torch.float16:
            hidden_states = cast_overflow_tensors(hidden_states)

        # adapter
        hidden_states = self.merger(hidden_states)
        reverse_indices = torch.argsort(window_index)
        hidden_states = hidden_states[reverse_indices, :]

        # Unbatch
        seqlens = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).squeeze(0) // self.spatial_merge_unit
        result = hidden_states.squeeze(0).split(seqlens.tolist(), dim=0)

        return result


class ModalityProcessor(BaseModalityProcessor):
    """Qwen2.5-VL modality processor."""

    def __init__(self, model_id: str) -> None:
        """Initialize the processor."""
        super().__init__(model_id=model_id)
        self.hf_processor = AutoImageProcessor.from_pretrained(model_id)

    def get_image_processor(self) -> Callable | None:
        """Return the image processor."""

        def processor(image: npt.NDArray) -> dict[str, npt.NDArray]:
            """Invoke the HF processor and convert to dict."""
            return self.hf_processor(images=[image], return_tensors="np").data

        return processor

    def get_video_processor(self) -> Callable | None:
        """Return the video processor."""

        def processor(video: npt.NDArray) -> dict[str, npt.NDArray]:
            """Invoke the HF processor and convert to dict."""
            return self.hf_processor(images=None, videos=[video], return_tensors="np").data

        return processor
