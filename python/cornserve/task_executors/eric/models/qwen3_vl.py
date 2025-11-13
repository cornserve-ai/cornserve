"""Qwen3-VL Vision Transformer implementation for Eric."""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy.typing as npt
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.video_utils import VideoMetadata

from .base import EricModel
from .layers.activations import get_act_fn
from .layers.linear import ColumnParallelLinear, RowParallelLinear
from .qwen2_5_vl import Qwen2_5_VisionAttention, Qwen2_5_VisionRotaryEmbedding
from cornserve.task_executors.eric.api import Modality
from cornserve.task_executors.eric.router.processor import BaseModalityProcessor, thread_local


class Qwen3_VisionPatchEmbed(nn.Module):
    """Module to produce patch embeddings of an image or video.

    Uses Linear instead of Conv3d for better performance. When kernel_size == stride,
    Conv3d is mathematically equivalent to Linear but has significant overhead.
    """

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels

        input_dim = in_channels * temporal_patch_size * patch_size * patch_size
        self.proj = nn.Linear(input_dim, embed_dim, bias=True)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Convert Conv3d weights to Linear weights during loading.

        HuggingFace checkpoints have Conv3d weights with shape [out, in, kt, kh, kw].
        We need to reshape them to Linear weights with shape [out, in*kt*kh*kw].
        """
        weight_key = f"{prefix}proj.weight"

        if weight_key in state_dict:
            conv3d_weight = state_dict[weight_key]
            # Conv3d weight shape: [out_channels, in_channels, kt, kh, kw]
            if conv3d_weight.dim() == 5:
                out_channels, in_channels, kt, kh, kw = conv3d_weight.shape
                # Reshape to Linear weight: [out_channels, in_channels * kt * kh * kw]
                linear_weight = conv3d_weight.reshape(out_channels, in_channels * kt * kh * kw)
                state_dict[weight_key] = linear_weight

        # Call the parent implementation to actually load the weights
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Qwen3_VisionMLP(nn.Module):
    """Two layer MLP module for the Qwen3 vision encoder."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_name: str,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.linear_fc1 = ColumnParallelLinear(in_features, hidden_features, bias=bias)
        self.linear_fc2 = RowParallelLinear(hidden_features, in_features, bias=bias)
        self.act_fn = get_act_fn(act_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.linear_fc1(x)
        hidden = self.act_fn(hidden)
        output, _ = self.linear_fc2(hidden)
        return output


class Qwen3_VisionBlock(nn.Module):
    """A single transformer block for the Qwen3 vision encoder."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_name: str,
        norm_eps: float,
    ) -> None:
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Qwen2_5_VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
        )
        self.mlp = Qwen3_VisionMLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_name=act_name,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            max_seqlen=max_seqlen,
        )
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3_VisionPatchMerger(nn.Module):
    """Downsamples patch embeddings before feeding them to the LLM."""

    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_eps: float,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        if self.use_postshuffle_norm:
            context_dim = self.hidden_size

        self.norm = nn.LayerNorm(context_dim, eps=norm_eps)
        self.linear_fc1 = ColumnParallelLinear(self.hidden_size, self.hidden_size, bias=True)
        self.act_fn = nn.GELU()
        self.linear_fc2 = RowParallelLinear(self.hidden_size, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)

        hidden, _ = self.linear_fc1(x)
        hidden = self.act_fn(hidden)
        output, _ = self.linear_fc2(hidden)
        return output


class Qwen3_VisionTransformer(EricModel):
    """Vision encoder for Qwen3-VL models."""

    def __init__(self, config: Qwen3VLConfig) -> None:
        super().__init__()

        vision_config = config.vision_config

        self.in_channels = vision_config.in_channels
        self.embed_dim = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.patch_size = vision_config.patch_size
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.deepstack_visual_indexes = vision_config.deepstack_visual_indexes or []
        self.out_hidden_size = vision_config.out_hidden_size

        self.num_position_embeddings = vision_config.num_position_embeddings
        self.num_grid_per_side = int(math.sqrt(self.num_position_embeddings))

        self.patch_embed = Qwen3_VisionPatchEmbed(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
        )
        self.pos_embed = nn.Embedding(self.num_position_embeddings, self.embed_dim)

        head_dim = self.embed_dim // self.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        norm_eps = getattr(vision_config, "rms_norm_eps", 1e-6)
        self.blocks = nn.ModuleList(
            [
                Qwen3_VisionBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=vision_config.intermediate_size,
                    act_name=vision_config.hidden_act,
                    norm_eps=norm_eps,
                )
                for _ in range(vision_config.depth)
            ]
        )

        self.merger = Qwen3_VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=self.embed_dim,
            norm_eps=norm_eps,
            spatial_merge_size=self.spatial_merge_size,
            use_postshuffle_norm=False,
        )

        if self.deepstack_visual_indexes:
            self.deepstack_merger_list = nn.ModuleList(
                [
                    Qwen3_VisionPatchMerger(
                        d_model=vision_config.out_hidden_size,
                        context_dim=self.embed_dim,
                        norm_eps=norm_eps,
                        spatial_merge_size=self.spatial_merge_size,
                        use_postshuffle_norm=True,
                    )
                    for _ in range(len(self.deepstack_visual_indexes))
                ]
            )
        else:
            self.deepstack_merger_list = None

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    @property
    def chunk_shape(self) -> tuple[int, ...]:
        return (1, self.out_hidden_size * (1 + len(self.deepstack_visual_indexes)))

    def forward(
        self,
        modality: Modality,
        adapter_name: str,
        batch: dict[str, list[torch.Tensor]],
    ) -> list[torch.Tensor]:
        match modality:
            case Modality.IMAGE:
                pixel_values = torch.cat(batch["pixel_values"], dim=0).to(device=self.device, dtype=self.dtype)
                grid_thw = torch.cat(batch["image_grid_thw"], dim=0).to(device=self.device)
            case Modality.VIDEO:
                pixel_values = torch.cat(batch["pixel_values_videos"], dim=0).to(device=self.device, dtype=self.dtype)
                grid_thw = torch.cat(batch["video_grid_thw"], dim=0).to(device=self.device)
            case _:
                raise ValueError(f"Unsupported modality: {modality}")

        embeddings = self.patch_embed(pixel_values)
        embeddings = embeddings + self.fast_pos_embed_interpolate(grid_thw).to(device=self.device, dtype=self.dtype)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        embeddings = embeddings.unsqueeze(1)
        rotary_pos_emb = rotary_pos_emb.to(embeddings.device)
        max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens)

        hidden_states_list: list[torch.Tensor] = []
        for layer_idx, block in enumerate(self.blocks):
            embeddings = block(
                embeddings,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                max_seqlen=max_seqlen,
            )
            if layer_idx in self.deepstack_visual_indexes:
                hidden_states_list.append(embeddings)

        embeddings = self.merger(embeddings)

        if self.deepstack_visual_indexes:
            assert self.deepstack_merger_list is not None
            processed_hidden_states = [embeddings]
            for idx, hidden_state in enumerate(hidden_states_list):
                processed_hidden_states.append(self.deepstack_merger_list[idx](hidden_state))
            embeddings = torch.cat(processed_hidden_states, dim=1)

        seqlens = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]) // self.spatial_merge_unit
        return list(embeddings.split(seqlens.tolist(), dim=0))

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        device = grid_thw.device

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            frames = int(num_frames.item()) if isinstance(num_frames, torch.Tensor) else int(num_frames)
            h_val = int(height.item()) if isinstance(height, torch.Tensor) else int(height)
            w_val = int(width.item()) if isinstance(width, torch.Tensor) else int(width)

            merged_h, merged_w = h_val // merge_size, w_val // merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)
            if frames > 1:
                coords = coords.repeat(frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        rotary_pos_emb = freq_table[pos_ids].flatten(1)
        return rotary_pos_emb

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        device = self.device
        dtype = self.pos_embed.weight.dtype

        for t, h, w in grid_thw:
            frames = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
            height = int(h.item()) if isinstance(h, torch.Tensor) else int(h)
            width = int(w.item()) if isinstance(w, torch.Tensor) else int(w)

            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, height, device=device, dtype=torch.float32)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, width, device=device, dtype=torch.float32)

            h_floor = h_idxs.to(torch.long)
            w_floor = w_idxs.to(torch.long)
            h_ceil = torch.clamp(h_floor + 1, max=self.num_grid_per_side - 1)
            w_ceil = torch.clamp(w_floor + 1, max=self.num_grid_per_side - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
            h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
            h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1 - dh_grid - dw_grid + w11

            h_floor_grid_idx = h_floor_grid * self.num_grid_per_side
            h_ceil_grid_idx = h_ceil_grid * self.num_grid_per_side
            idx00 = h_floor_grid_idx + w_floor_grid
            idx01 = h_floor_grid_idx + w_ceil_grid
            idx10 = h_ceil_grid_idx + w_floor_grid
            idx11 = h_ceil_grid_idx + w_ceil_grid

            indices = torch.stack([idx00, idx01, idx10, idx11], dim=0).reshape(4, -1)
            weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1).to(dtype)

            embeds = self.pos_embed(indices)
            weighted_embeds = embeds * weights
            combined = weighted_embeds.sum(dim=0)

            merge_size = self.spatial_merge_size
            embed_dim = self.pos_embed.embedding_dim
            combined = combined.view(height * width, embed_dim)
            repeated = combined.unsqueeze(0).expand(frames, -1, -1).contiguous()
            repeated = (
                repeated.view(frames, height // merge_size, merge_size, width // merge_size, merge_size, embed_dim)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(-1, embed_dim)
            )

            outputs.append(repeated)

        return torch.cat(outputs, dim=0)

    def compute_attn_mask_seqlen(self, cu_seqlens: torch.Tensor) -> int:
        return int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())


class ModalityProcessor(BaseModalityProcessor):
    """Qwen3-VL modality processor."""

    def __init__(self, model_id: str) -> None:
        super().__init__(model_id=model_id)
        self.hf_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def get_image_processor(self) -> Callable | None:
        def processor(image: npt.NDArray) -> dict[str, torch.Tensor]:
            return self.hf_processor.image_processor(images=[image], videos=None, return_tensors="pt").data

        return processor

    def get_video_processor(self) -> Callable | None:
        def processor(video: npt.NDArray) -> dict[str, torch.Tensor]:
            """Invoke the HF processor and convert to dict."""

            # Access the loader's metadata which has the actual FPS captured from the video file
            loader_metadata = thread_local.loader.video_loader.last_video_metadata
            metadata = VideoMetadata(
                fps=loader_metadata["fps"],
                duration=loader_metadata["duration"],
                total_num_frames=loader_metadata["total_num_frames"],
                frames_indices=loader_metadata["frames_indices"],
                video_backend="opencv",
            )

            out = self.hf_processor.video_processor(
                videos=[video],
                video_metadata=[metadata],
                do_sample_frames=False,
                return_tensors="pt",
            )
            return out.data

        return processor
