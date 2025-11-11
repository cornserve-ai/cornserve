import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy.typing as npt
from transformers.models.auto.processing_auto import AutoProcessor

from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
    Qwen3OmniMoeVisionEncoderConfig,
    Qwen3OmniMoeAudioEncoderConfig,
)

from cornserve.task_executors.eric.models.layers.activations import get_act_fn
from flash_attn import flash_attn_varlen_func
from .qwen2_5_vl import Qwen2_5_VisionRotaryEmbedding, Qwen2_5_VisionAttention
from .qwen2_5_omni import SinusoidsPositionEmbedding

from .base import EricModel
from .layers.linear import ColumnParallelLinear, RowParallelLinear
from cornserve.task_executors.eric.api import Modality
from cornserve.task_executors.eric.distributed import parallel
from cornserve.task_executors.eric.router.processor import BaseModalityProcessor
from cornserve.task_executors.eric.utils import distributed as dist_utils


class Qwen3_VisionPatchEmbed(nn.Module):
    """Module to produce patch embeddings of a video/image.

    Qwen3 vision encoder's forward pass uses this to patchify inputs
    before running them through transformer blocks.
    """

    def __init__(
        self,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.embed_dim = embed_dim

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_all_patches, _ = x.shape
        x = x.view(num_all_patches, -1, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(x).view(num_all_patches, self.embed_dim)


class Qwen3_VisionMLP(nn.Module):
    """Two layer MLP module for Qwen3 vision encoder.

    Qwen3 vision encoder's transformer block uses this right after applying attention.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_name: str,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.linear_fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
        )
        self.linear_fc2 = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
        )
        self.act_fn = get_act_fn(act_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mlp_output = self.linear_fc2(self.act_fn(self.linear_fc1(x)[0]))[0]
        return mlp_output


class Qwen3_VisionBlock(nn.Module):
    """A single transformer block for Qwen3 vision encoder."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_name: str,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)

        # It's alright to reuse Qwen2.5 vision attention here.
        self.attn = Qwen2_5_VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
        )
        self.mlp = Qwen3_VisionMLP(
            dim,
            mlp_hidden_dim,
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
    """Concatenates and compresses neighboring patch embeddings.

    Qwen3 vision encoder applies this on hidden states after running through
    transformer blocks.
    """

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

        self.ln_q = nn.LayerNorm(context_dim, eps=norm_eps)
        self.mlp = nn.ModuleList(
            [
                ColumnParallelLinear(
                    self.hidden_size,
                    self.hidden_size,
                    bias=True,
                ),
                nn.GELU(),
                RowParallelLinear(
                    self.hidden_size,
                    d_model,
                    bias=True,
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.ln_q(x.view(-1, self.hidden_size))
        else:
            x = self.ln_q(x).view(-1, self.hidden_size)

        mlp_fc1, mlp_act, mlp_fc2 = self.mlp
        x_parallel, _ = mlp_fc1(x)
        x_parallel = mlp_act(x_parallel)
        out, _ = mlp_fc2(x_parallel)
        return out


class Qwen3_VisionTransformer(EricModel):
    """Vision encoder for Qwen3.

    Qwen3OmniEncoder uses this as a sub-component.
    """

    def __init__(
        self,
        vision_config: Qwen3OmniMoeVisionEncoderConfig,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()

        self.in_channels = vision_config.in_channels
        self.embed_dim = vision_config.hidden_size
        self.num_heads = vision_config.num_heads

        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.deepstack_visual_indexes = vision_config.deepstack_visual_indexes
        self.out_hidden_size = vision_config.out_hidden_size

        self.num_position_embeddings = vision_config.num_position_embeddings
        self.num_grid_per_side = int(math.sqrt(self.num_position_embeddings))

        self.patch_embed = Qwen3_VisionPatchEmbed(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
        )

        # Embedding table needed for interpolating arbitrary input image coords
        # into fixed grid.
        # Stores a unique vector for each grid position.
        self.pos_embed = torch.nn.Embedding(self.num_position_embeddings, self.embed_dim).to(
            device=self.device, dtype=self.dtype
        )

        head_dim = self.embed_dim // self.num_heads
        self.rotary_pos_emb = Qwen2_5_VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [
                Qwen3_VisionBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=vision_config.intermediate_size,
                    act_name=vision_config.hidden_act,
                    norm_eps=rms_norm_eps,
                )
                for layer_idx in range(vision_config.depth)
            ]
        )
        self.merger = Qwen3_VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=self.embed_dim,
            norm_eps=rms_norm_eps,
            spatial_merge_size=self.spatial_merge_size,
            use_postshuffle_norm=False,
        )

        if self.deepstack_visual_indexes is not None:
            self.merger_list = nn.ModuleList(
                [
                    Qwen3_VisionPatchMerger(
                        d_model=vision_config.out_hidden_size,
                        context_dim=self.embed_dim,
                        norm_eps=rms_norm_eps,
                        spatial_merge_size=self.spatial_merge_size,
                        use_postshuffle_norm=True,
                    )
                    for layer_idx in range(len(self.deepstack_visual_indexes))
                ]
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

    def forward(
        self,
        modality: Modality,
        adapter_name: str,
        batch: dict[str, list[torch.Tensor]],
    ) -> list[torch.Tensor]:
        """Forward pass of the model.

        For images, `batch` is expected to have the following keys:
        - `pixel_values`: The pixel values of the images. Each [seq_len, 6 * patch_size (16) * patch_size (16)].
        - `image_grid_thw`: The grid size of the images. Each [1, 3].

        For videos, `batch` is expected to have the following keys:
        - `pixel_values_videos`: The pixel values of the videos. Each [seq_len, 6 * patch_size (16) * patch_size (16)].
        - `video_grid_thw`: The grid size of the videos. Each [1, 3].
        """

        # Concatenate inputs in the batch into one tensor
        match modality:
            case Modality.IMAGE:
                pixel_values = torch.cat(batch["pixel_values"], dim=0).to(device=self.device, dtype=self.dtype)
                grid_thw = torch.cat(batch["image_grid_thw"], dim=0).to(device=self.device)
            case Modality.VIDEO:
                pixel_values = torch.cat(batch["pixel_values_videos"], dim=0).to(device=self.device, dtype=self.dtype)
                grid_thw = torch.cat(batch["video_grid_thw"], dim=0).to(device=self.device)
            case _:
                raise ValueError(f"Unsupported modality: {modality}")

        # Make patch embeddings from the input
        embeddings = self.patch_embed(pixel_values)

        # Provide absolute position embedding
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        embeddings = embeddings + pos_embeds

        # Get rotary embeddings
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        # Compute cu_seqlens for FlashAttention
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), "constant", 0)

        embeddings = embeddings.unsqueeze(1)
        rotary_pos_emb = rotary_pos_emb.to(embeddings.device)
        max_seqlen = self.compute_attn_mask_seqlen(cu_seqlens)

        # As we run the embeddings through transformer blocks, intermediate results produced by
        # layers specified in deepstack_visual_indexes will be preserved & stored in here.
        hidden_states_list = []
        deepstack_visual_indexes = self.deepstack_visual_indexes

        # Iterate through transformer blocks
        for layer_num, blk in enumerate(self.blocks):
            embeddings = blk(
                embeddings,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
                max_seqlen=max_seqlen,
            )
            if deepstack_visual_indexes is not None and layer_num in deepstack_visual_indexes:
                hidden_states_list.append(embeddings)

        # Downsample
        embeddings = self.merger(embeddings)

        if deepstack_visual_indexes is not None:
            processed_hidden_states_list = [embeddings]

            for idx, x in enumerate(hidden_states_list):
                # Downsample
                x = self.merger_list[idx](x)
                processed_hidden_states_list.append(x)

            # Now, we concatenate all these hidden states into
            # one giant vector
            embeddings = torch.cat(processed_hidden_states_list, dim=1)

        # Unbatch
        seqlens = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]) // self.spatial_merge_unit
        result = embeddings.split(seqlens.tolist(), dim=0)

        return list(result)

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

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        outputs = []
        for t, h, w in grid_thw:
            # Plot h evenly spaced points from 0 to self.num_grid_per_side - 1.
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h, device=self.device, dtype=torch.float32)
            # Plot w evenly spaced points from 0 to self.num_grid_per_side - 1.
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w, device=self.device, dtype=torch.float32)

            # For each point, we need to find the four grid points that surround it.
            h_floor = h_idxs.to(torch.long)
            w_floor = w_idxs.to(torch.long)
            h_ceil = torch.clamp(h_floor + 1, max=self.num_grid_per_side - 1)
            w_ceil = torch.clamp(w_floor + 1, max=self.num_grid_per_side - 1)

            # The weights for interpolation depend on how close the target point is to each
            # of its four neighbors.
            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            # Create (h, w) sized tensors that we can operate on.
            dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
            h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
            h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

            # For each point in (h, w), compute weights of its four neighbors
            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1 - dh_grid - dw_grid + w11  # same as (1 - dh_grid) * (1 - dw_grid)

            # Convert the 2D coordinates of our four neighbors into 1D indices we can
            # use to index self.pos_embed
            # The formula is index = row * num_columns + column
            h_floor_grid_idx = h_floor_grid * self.num_grid_per_side
            h_ceil_grid_idx = h_ceil_grid * self.num_grid_per_side
            idx00 = h_floor_grid_idx + w_floor_grid
            idx01 = h_floor_grid_idx + w_ceil_grid
            idx10 = h_ceil_grid_idx + w_floor_grid
            idx11 = h_ceil_grid_idx + w_ceil_grid

            # Stack the indices and weights
            indices = torch.stack([idx00, idx01, idx10, idx11], dim=0).reshape(4, -1)
            weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1).to(self.dtype)

            # Fetch the embedding vectors for all four
            # neighbors for every point in our target grid.
            embeds = self.pos_embed(indices)

            # We apply the weights. Each of the 4 neighbor's embeddings is scaled
            # by its corresponding weight. Then, sum them up.
            weighted_embeds = embeds * weights
            combined = weighted_embeds.sum(dim=0)

            # Reshape the result from a flat list back into a 2D grid representation.
            m_size = self.spatial_merge_size
            embed_dim = self.pos_embed.embedding_dim
            combined = combined.view(h * w, embed_dim)
            repeated = combined.unsqueeze(0).expand(t, -1, -1).contiguous()
            repeated = (
                repeated.view(t, h // m_size, m_size, w // m_size, m_size, embed_dim)
                .permute(0, 1, 3, 2, 4, 5)
                .reshape(-1, embed_dim)
            )

            outputs.append(repeated)

        return torch.cat(outputs, dim=0)

    def compute_attn_mask_seqlen(self, cu_seqlens: torch.Tensor) -> int:
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        return max_seqlen


class Qwen3OmniMoeAudioAttention(nn.Module):
    def __init__(self, config: Qwen3OmniMoeAudioEncoderConfig) -> None:
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

        # the `bias=True` in this line differs from `Qwen2_5OmniAudioFlashAttention2`
        self.k_proj = ColumnParallelLinear(self.embed_dim, self.embed_dim, bias=True)

        self.v_proj = ColumnParallelLinear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = ColumnParallelLinear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = RowParallelLinear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        seq_length, all_dim = hidden_states.size()
        query_states = self.q_proj(hidden_states)[0]
        query_states = query_states.reshape(seq_length, self.num_heads_per_partition, self.head_dim)

        key_states = self.k_proj(hidden_states)[0]
        key_states = key_states.reshape(seq_length, self.num_heads_per_partition, self.head_dim)
        value_states = self.v_proj(hidden_states)[0]
        value_states = value_states.reshape(seq_length, self.num_heads_per_partition, self.head_dim)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p=self.dropout,
        )
        attn_output = attn_output.reshape(seq_length, dist_utils.divide(all_dim, self.tp_size))
        attn_output = self.out_proj(attn_output)[0]

        return attn_output


class Qwen3OmniMoeAudioEncoderLayer(nn.Module):
    def __init__(self, config: Qwen3OmniMoeAudioEncoderConfig) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Qwen3OmniMoeAudioAttention(config)
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


class Qwen3OmniAudioEncoder(nn.Module):
    """Audio encoder for Qwen3.

    Qwen3OmniEncoder uses this as a sub-component.
    """

    def __init__(self, config: Qwen3OmniMoeAudioEncoderConfig) -> None:
        super().__init__()
        self.dropout = config.dropout

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.positional_embedding = SinusoidsPositionEmbedding(self.max_source_positions, embed_dim)
        self.layers = nn.ModuleList([Qwen3OmniMoeAudioEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.ln_post = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * ((((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2),
            config.d_model,
            bias=False,
        )

        self.proj1 = ColumnParallelLinear(
            config.d_model,
            config.d_model,
            bias=True,
        )
        self.act = get_act_fn(config.activation_function)

        self.proj2 = RowParallelLinear(
            config.d_model,
            config.output_dim,
            bias=True,
        )

        self.n_window_infer = config.n_window_infer
        self.conv_chunksize = config.conv_chunksize

    def forward(
        self,
        input_features,
        feature_lens,
    ) -> list[torch.Tensor]:
        aftercnn_lens = self._get_feat_extract_output_lengths(feature_lens)
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
        feature_lens_after_cnn = self._get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = nn.utils.rnn.pad_sequence(
            [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn],
            batch_first=True,
        )
        padded_feature = padded_feature.unsqueeze(1)

        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = F.gelu(self.conv2d1(chunk))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)
        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
        for cnn_len in aftercnn_lens:
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32)

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)[0]
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)[0]

        # hidden_states is currently one big tensor of shape (total_valid_tokens, output_dim),
        # so we need to unbatch it before returning.
        unbatched_tuple = torch.split(hidden_states, aftercnn_lens.tolist(), dim=0)
        return list(unbatched_tuple)

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor) -> torch.LongTensor:
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        return output_lengths


class Qwen3OmniEncoder(EricModel):
    """Top-level encoder for Qwen3-Omni."""

    def __init__(self, config: Qwen3OmniMoeConfig) -> None:
        super().__init__()

        self.config = config
        thinker_config = config.thinker_config

        rms_norm_eps = getattr(thinker_config.text_config, "rms_norm_eps", 1e-6)
        self.visual = Qwen3_VisionTransformer(thinker_config.vision_config, rms_norm_eps)

        self.audio_tower = Qwen3OmniAudioEncoder(thinker_config.audio_config)

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
        adapter_name: str,
        batch: dict[str, list[torch.Tensor]],
    ) -> list[torch.Tensor]:
        """Forward pass of the model.

        For images, `batch` is expected to have the following keys:
        - `pixel_values`: The pixel values of the images. Each [seq_len, 6 * patch_size (16) * patch_size (16)].
        - `image_grid_thw`: The grid size of the images, Each [1, 3].

        For videos, `batch` is expected to have the following keys:
        - `pixel_values_videos`: The pixel values of the videos. Each [seq_len, 6 * patch_size (16) * patch_size (16)].
        - `video_grid_thw`: The grid size of the videos, Each [1, 3].

        For audios, `batch` is expected to have the following keys:
        - `input_audio_features`: The audio Mel spectrogram features. Each [feature_size (128), seq_len].
        - `audio_feature_lengths`: The lengths of the audio features. Each [1,].
        """
        # Batch
        match modality:
            case Modality.IMAGE | Modality.VIDEO:
                return self.visual(modality, adapter_name, batch)
            case Modality.AUDIO:
                input_features = torch.cat(batch["input_audio_features"], dim=1).to(
                    device=self.device, dtype=self.dtype
                )
                audio_feature_lengths = torch.cat(batch["audio_feature_lengths"], dim=0).to(device=self.device)
                audio_features = self.audio_tower(
                    input_features,
                    audio_feature_lengths,
                )
                return audio_features
            case _:
                raise ValueError(f"Unsupported modality: {modality}")


class ModalityProcessor(BaseModalityProcessor):
    """Qwen3-Omni modality processor."""

    def __init__(self, model_id: str) -> None:
        """Initialize the processor."""
        super().__init__(model_id=model_id)

        # Ends up being type `Qwen3OmniMoeProcessor`
        self.hf_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    def get_image_processor(self) -> Callable | None:
        """Return the image processor."""

        def processor(image: npt.NDArray) -> dict[str, torch.Tensor]:
            """Invoke the HF processor and convert to dict."""
            return self.hf_processor.image_processor(images=[image], videos=None, return_tensors="pt").data

        return processor

    def get_audio_processor(self) -> Callable | None:
        """Return the audio processor."""

        def processor(audio: npt.NDArray) -> dict[str, torch.Tensor]:
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
                input_audio_features=input_features,
                audio_feature_lengths=attention_mask.sum(-1),
                feature_attention_mask=attention_mask,
            )

        return processor

    def get_video_processor(self) -> Callable | None:
        """Return the video processor."""

        def processor(video: npt.NDArray) -> dict[str, torch.Tensor]:
            """Invoke the HF processor and convert to dict."""
            out = self.hf_processor.video_processor(
                videos=[video],
                min_pixels=128 * 32 * 32,
                max_pixels=768 * 32 * 32,
                return_tensors="pt",
            )
            return out.data

        return processor
