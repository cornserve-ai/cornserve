"""Sequence parallelism wrapper for diffusers DiT pipelines.

Patches a ``QwenImagePipeline``'s transformer attention layers to use
Ulysses SP (All-to-All), and provides utilities for scattering/gathering
latent tokens across the SP group.

Design principles:
- Monkey-patch the attention processor, not the whole transformer.
- Only patch when sp_size > 1; sp_size=1 is a no-op (backward compatible).
- All NCCL collectives go through :class:`SPGroup` from ``distributed/parallel.py``.
"""

from __future__ import annotations

import base64
import io
import os

import torch
import torch.nn.functional as F  # noqa: N812
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline, calculate_shift
from PIL import Image

from cornserve.logging import get_logger
from cornserve.task_executors.geri.distributed.parallel import SPGroup

logger = get_logger(__name__)

ENV_MAX_DOUBLE_BLOCKS = "CORNSERVE_GERI_QWEN_IMAGE_MAX_DOUBLE_BLOCKS"
ENV_MAX_SINGLE_BLOCKS = "CORNSERVE_GERI_QWEN_IMAGE_MAX_SINGLE_BLOCKS"


def read_positive_int_env(name: str) -> int | None:
    """Read an environment variable as a positive integer, returning None if unset."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"Environment variable {name} must be > 0, got: {parsed}")
    return parsed


def apply_qwen_image_dev_layer_overrides(pipeline) -> None:
    """Truncate transformer block counts based on environment variable overrides."""
    transformer = pipeline.transformer

    max_double_blocks = read_positive_int_env(ENV_MAX_DOUBLE_BLOCKS)
    max_single_blocks = read_positive_int_env(ENV_MAX_SINGLE_BLOCKS)

    if max_double_blocks is not None and hasattr(transformer, "transformer_blocks"):
        current = len(transformer.transformer_blocks)
        target = min(current, max_double_blocks)
        if target < current:
            logger.warning(
                "Applying dev layer cap for Qwen-Image double-stream blocks: %d -> %d (%s)",
                current,
                target,
                ENV_MAX_DOUBLE_BLOCKS,
            )
            transformer.transformer_blocks = transformer.transformer_blocks[:target]

    if max_single_blocks is not None and hasattr(transformer, "single_transformer_blocks"):
        current = len(transformer.single_transformer_blocks)
        target = min(current, max_single_blocks)
        if target < current:
            logger.warning(
                "Applying dev layer cap for Qwen-Image single-stream blocks: %d -> %d (%s)",
                current,
                target,
                ENV_MAX_SINGLE_BLOCKS,
            )
            transformer.single_transformer_blocks = transformer.single_transformer_blocks[:target]


# ---------------------------------------------------------------------------
# SP-aware attention processor (replaces QwenDoubleStreamAttnProcessor2_0)
# ---------------------------------------------------------------------------


class SPQwenDoubleStreamAttnProcessor:
    """SP-aware attention processor for QwenImage's double-stream (joint) attention.

    Replaces ``QwenDoubleStreamAttnProcessor2_0`` when ``sp_size > 1``.

    In the SP layout, image tokens are split across ranks along the
    sequence dimension, while text tokens are fully replicated.

    Algorithm per rank:
    1. Compute Q, K, V projections locally.
    2. For text tokens: already full on each rank.
    3. For image tokens: each rank has ``S_img / sp_size`` tokens.
    4. Concatenate text + image to get joint Q, K, V.
       - joint Q/K/V shape: ``[B, S_txt + S_img_local, H, D]``
    5. All-to-All on the joint sequence: scatter on heads, gather on seq
       → each rank now has full joint sequence with ``H / sp_size`` heads.
    6. Local SDPA (full seq, head shard).
    7. All-to-All back: scatter on seq, gather on heads.
    8. Split output back into text and image parts.
    """

    def __init__(self, sp_group: SPGroup) -> None:
        """Initialize with the SP group reference."""
        self.sp_group = sp_group

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,  # Image stream [B, S_img_local, C]
        encoder_hidden_states: torch.Tensor | None = None,  # Text stream [B, S_txt, C]
        encoder_hidden_states_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb=None,
    ) -> torch.Tensor:
        """Execute SP joint attention."""
        if encoder_hidden_states is None:
            raise ValueError("SPQwenDoubleStreamAttnProcessor requires encoder_hidden_states (text stream)")

        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Compute QKV for text stream
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention: [B, S, H, D]
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))
        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE — image RoPE is already split (since image tokens are split)
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # Concatenate for joint attention: [text, image]
        # Text is replicated, image is split
        # Joint seq: [B, S_txt + S_img_local, H, D]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # --- Ulysses SP: All-to-All ---
        # Transpose to [B, H, S, D] for All-to-All (standard attention layout)
        joint_query = joint_query.transpose(1, 2)  # [B, H, S_local, D]
        joint_key = joint_key.transpose(1, 2)
        joint_value = joint_value.transpose(1, 2)

        # All-to-All: seq-split → head-split
        # scatter on heads (dim=1), gather on seq (dim=2)
        joint_query = self.sp_group.all_to_all(joint_query, scatter_dim=1, gather_dim=2)
        joint_key = self.sp_group.all_to_all(joint_key, scatter_dim=1, gather_dim=2)
        joint_value = self.sp_group.all_to_all(joint_value, scatter_dim=1, gather_dim=2)

        # Local attention: full joint sequence, subset of heads
        joint_hidden_states = F.scaled_dot_product_attention(
            joint_query,
            joint_key,
            joint_value,
            dropout_p=0.0,
            is_causal=False,
        )

        # All-to-All back: head-split → seq-split
        # scatter on seq (dim=2), gather on heads (dim=1)
        joint_hidden_states = self.sp_group.all_to_all(joint_hidden_states, scatter_dim=2, gather_dim=1)

        # Transpose back to [B, S, H, D]
        joint_hidden_states = joint_hidden_states.transpose(1, 2)

        # Flatten heads: [B, S, H*D]
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output.contiguous())
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)

        txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())

        return img_attn_output, txt_attn_output


# ---------------------------------------------------------------------------
# SP-aware single-stream attention processor
# ---------------------------------------------------------------------------


class SPQwenSingleStreamAttnProcessor:
    """SP-aware attention processor for QwenImage's single-stream blocks.

    Used in the later blocks of the transformer where text and image streams
    are already merged into a single hidden state.
    """

    def __init__(self, sp_group: SPGroup) -> None:
        """Initialize with the SP group reference."""
        self.sp_group = sp_group

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb=None,
    ) -> torch.Tensor:
        """Execute SP single-stream attention."""
        # QKV projection
        qkv = attn.to_qkv(hidden_states)
        split_size = qkv.shape[-1] // 3
        query, key, value = qkv.split(split_size, dim=-1)

        # Reshape: [B, S, H, D]
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # QK norm
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE
        if image_rotary_emb is not None:
            query = apply_rotary_emb_qwen(query, image_rotary_emb, use_real=False)
            key = apply_rotary_emb_qwen(key, image_rotary_emb, use_real=False)

        # --- Ulysses SP: All-to-All ---
        query = query.transpose(1, 2)  # [B, H, S, D]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        query = self.sp_group.all_to_all(query, scatter_dim=1, gather_dim=2)
        key = self.sp_group.all_to_all(key, scatter_dim=1, gather_dim=2)
        value = self.sp_group.all_to_all(value, scatter_dim=1, gather_dim=2)

        out = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        out = self.sp_group.all_to_all(out, scatter_dim=2, gather_dim=1)

        out = out.transpose(1, 2)  # [B, S, H, D]
        out = out.flatten(2, 3)
        out = out.to(query.dtype)

        # Output projection
        out = attn.to_out[0](out)
        if len(attn.to_out) > 1:
            out = attn.to_out[1](out)

        return out


# ---------------------------------------------------------------------------
# Pipeline patching
# ---------------------------------------------------------------------------


def patch_pipeline_for_sp(pipeline, sp_group: SPGroup) -> None:
    """Monkey-patch a QwenImagePipeline's transformer for sequence parallelism.

    Replaces attention processors in all transformer blocks with SP-aware
    versions that use Ulysses All-to-All attention.

    Also patches the RoPE embedding module so that it returns scattered
    image frequencies (each rank gets its position encoding shard) while
    keeping text frequencies fully replicated.

    Args:
        pipeline: A ``QwenImagePipeline`` instance.
        sp_group: The SP group for this set of workers.
    """
    if sp_group.world_size == 1:
        return

    transformer = pipeline.transformer

    # Patch double-stream blocks
    if hasattr(transformer, "transformer_blocks"):
        for i, block in enumerate(transformer.transformer_blocks):
            if hasattr(block, "attn"):
                block.attn.processor = SPQwenDoubleStreamAttnProcessor(sp_group)
                logger.debug("Patched double-stream block %d with SP attention.", i)

    # Patch single-stream blocks
    if hasattr(transformer, "single_transformer_blocks"):
        for i, block in enumerate(transformer.single_transformer_blocks):
            if hasattr(block, "attn"):
                block.attn.processor = SPQwenSingleStreamAttnProcessor(sp_group)
                logger.debug("Patched single-stream block %d with SP attention.", i)

    # Patch the RoPE module to return scattered image frequencies.
    # The transformer's forward computes RoPE and passes it to each block.
    # We wrap it so that image frequencies are scattered across SP ranks.
    original_rope = transformer.pos_embed
    original_rope_forward = original_rope.forward

    def sp_rope_forward(*args, **kwargs):
        """Wrapper that scatters image RoPE frequencies across SP ranks."""
        img_freqs, txt_freqs = original_rope_forward(*args, **kwargs)
        # Scatter image frequencies along the sequence dim (dim=0)
        img_freqs_local = sp_group.scatter(img_freqs, dim=0)
        # Text frequencies stay fully replicated
        return img_freqs_local, txt_freqs

    original_rope.forward = sp_rope_forward

    # Store SP group reference on the pipeline for use during generation
    pipeline._sp_group = sp_group

    logger.info(
        "Pipeline patched for SP: %d double-stream blocks, %d single-stream blocks, RoPE patched.",
        len(getattr(transformer, "transformer_blocks", [])),
        len(getattr(transformer, "single_transformer_blocks", [])),
    )


# ---------------------------------------------------------------------------
# Latent scatter / gather utilities
# ---------------------------------------------------------------------------


def scatter_latents(latents: torch.Tensor, sp_group: SPGroup, dim: int = 1) -> torch.Tensor:
    """Split packed latent tokens across SP ranks.

    QwenImage packs latents as ``[B, num_patches, C]`` where ``num_patches``
    is the spatial token dimension. We split along this dimension.

    Args:
        latents: Full latent tensor ``[B, S, C]``.
        sp_group: The SP group.
        dim: The sequence dimension to split along (default 1).

    Returns:
        This rank's shard of the latents.
    """
    return sp_group.scatter(latents, dim=dim)


def gather_latents(local_latents: torch.Tensor, sp_group: SPGroup, dim: int = 1) -> torch.Tensor:
    """Gather latent token shards from all SP ranks.

    Args:
        local_latents: This rank's latent shard ``[B, S_local, C]``.
        sp_group: The SP group.
        dim: The sequence dimension to gather along (default 1).

    Returns:
        Full latent tensor ``[B, S, C]``.
    """
    return sp_group.all_gather(local_latents, dim=dim)


# ---------------------------------------------------------------------------
# Load + patch helper for non-rank-0 workers
# ---------------------------------------------------------------------------


def load_and_patch_pipeline_for_sp(
    model_id: str,
    torch_dtype: torch.dtype,
    torch_device: torch.device,
    sp_group: SPGroup,
) -> tuple:
    """Load a QwenImagePipeline and patch it for SP.

    Used by non-rank-0 SP workers that need to load the same pipeline.

    Args:
        model_id: Hugging Face model ID.
        torch_dtype: Data type for model weights.
        torch_device: Device to load onto.
        sp_group: The SP group.

    Returns:
        Tuple of (pipeline, embedding_dim).
    """
    logger.info("Loading QwenImage pipeline from %s for SP worker.", model_id)

    pipeline = QwenImagePipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    embedding_dim = pipeline.text_encoder.config.hidden_size

    apply_qwen_image_dev_layer_overrides(pipeline)

    # Remove text encoder (embeddings come via sidecar/broadcast)
    pipeline.text_encoder = None
    pipeline.tokenizer = None

    if torch_device.type not in ["cpu", "meta"]:
        pipeline = pipeline.to(torch_device)

    patch_pipeline_for_sp(pipeline, sp_group)

    return pipeline, embedding_dim


# ---------------------------------------------------------------------------
# SP generation entry point
# ---------------------------------------------------------------------------


@torch.inference_mode()
def execute_sp_generate(
    pipeline,
    sp_group: SPGroup,
    prompt_embeds: torch.Tensor | None,
    prompt_embeds_mask: torch.Tensor | None,
    height: int,
    width: int,
    num_inference_steps: int,
    batch_size: int,
    max_seq_len: int,
    initial_latents: torch.Tensor | None = None,
    output_type: str = "png",
) -> list[str] | torch.Tensor | None:
    """Execute SP-parallel image generation.

    Rank 0 provides the actual prompt_embeds; other ranks receive via broadcast.
    All ranks run the same denoising pipeline in lockstep (for NCCL collectives).
    Only rank 0 runs VAE decode and returns results.

    The pipeline's transformer has been patched so that:
    - Attention processors use Ulysses SP (All-to-All).
    - RoPE module scatters image frequencies across SP ranks.

    We call the transformer's ``forward()`` directly in a manual denoising loop
    so we can scatter latents and pass correct ``img_shapes``.

    Args:
        pipeline: The (patched) QwenImagePipeline.
        sp_group: The SP group.
        prompt_embeds: [B, S, C] prompt embeddings (rank 0 only; None for others).
        prompt_embeds_mask: [B, S] mask (rank 0 only; None for others).
        height: Image height in pixels.
        width: Image width in pixels.
        num_inference_steps: Number of denoising steps.
        batch_size: Number of images in the batch.
        max_seq_len: Maximum sequence length of prompt embeddings.
        initial_latents: Pre-generated packed latents [B, num_patches, C] for
            deterministic testing. If None, latents are generated from random noise.
        output_type: Output format, either "png" (base64 strings) or "latent" (raw tensor).

    Returns:
        List of base64-encoded PNG strings on rank 0; None on other ranks.
    """
    device = torch.device("cuda")
    dtype = pipeline.dtype if hasattr(pipeline, "dtype") else torch.bfloat16

    # --- Broadcast prompt embeddings from rank 0 to all ---
    if sp_group.rank == 0:
        assert prompt_embeds is not None
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask.to(device=device)
    else:
        # Allocate buffers — use joint_attention_dim for text embedding dimension
        hidden_size = pipeline.transformer.config.joint_attention_dim
        prompt_embeds = torch.empty(batch_size, max_seq_len, hidden_size, device=device, dtype=dtype)
        prompt_embeds_mask = torch.empty(batch_size, max_seq_len, device=device, dtype=torch.long)

    sp_group.broadcast(prompt_embeds, src=0)
    sp_group.broadcast(prompt_embeds_mask, src=0)

    # --- Prepare latents ---
    if initial_latents is not None:
        # Use pre-generated latents (for deterministic testing)
        latents = initial_latents.to(device=device, dtype=dtype)
    else:
        # prepare_latents expects pre-pack channels (in_channels // 4 = 16) and
        # internally calls _pack_latents to produce [B, patches, in_channels].
        num_channels_latents = pipeline.transformer.config.in_channels // 4
        latents = pipeline.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator=None,
        )
    # Broadcast from rank 0 so all ranks start with same noise
    sp_group.broadcast(latents, src=0)

    # latents shape after prepare_latents + pack: [B, num_patches, C]
    # where num_patches = (latent_h // 2) * (latent_w // 2)
    # and C = in_channels (= num_channels_latents * 4 after packing 2x2 patches)

    # Scatter latents across SP ranks along the patch/sequence dimension
    latents = scatter_latents(latents, sp_group, dim=1)

    # --- Prepare timesteps ---
    vae_scale_factor = pipeline.vae_scale_factor
    latent_h = 2 * (int(height) // (vae_scale_factor * 2))
    latent_w = 2 * (int(width) // (vae_scale_factor * 2))
    num_patches = (latent_h // 2) * (latent_w // 2)

    mu = calculate_shift(num_patches)
    scheduler = pipeline.scheduler
    scheduler.set_timesteps(num_inference_steps, device=device, mu=mu)
    timesteps = scheduler.timesteps

    # Full image spatial dims for RoPE computation.
    # The patched pos_embed.forward will compute RoPE for these full dims
    # and then scatter the image portion across SP ranks.
    img_shapes = [(1, latent_h // 2, latent_w // 2)] * batch_size

    # --- Denoising loop ---
    # Compute txt_seq_lens from prompt_embeds_mask (needed by transformer's pos_embed)
    txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

    for _i, t in enumerate(timesteps):
        timestep = t.expand(batch_size).to(device=device, dtype=dtype)

        # All ranks call transformer.forward in lockstep.
        # The patched attention layers perform NCCL All-to-All collectives.
        # The patched RoPE computes full frequencies and scatters image portion.
        noise_pred = pipeline.transformer(
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            timestep=timestep / 1000,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]

        # Scheduler step (local per-rank, elementwise)
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # --- Gather latents and decode (rank 0 only) ---
    full_latents = gather_latents(latents, sp_group, dim=1)

    if output_type == "latent":
        # Skip VAE decode — return raw latents for benchmarking
        if sp_group.rank == 0:
            return full_latents
        return None

    if sp_group.rank == 0:
        # Unpack latents
        full_latents = pipeline._unpack_latents(full_latents, height, width, vae_scale_factor)
        full_latents = full_latents.to(pipeline.vae.dtype)

        # Denormalize latents (same as pipeline.__call__)
        latents_mean = (
            torch.tensor(pipeline.vae.config.latents_mean)
            .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
            .to(full_latents.device, full_latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(pipeline.vae.config.latents_std).view(
            1, pipeline.vae.config.z_dim, 1, 1, 1
        ).to(full_latents.device, full_latents.dtype)
        full_latents = full_latents / latents_std + latents_mean

        # VAE decode
        images = pipeline.vae.decode(full_latents, return_dict=False)[0][:, :, 0]
        images = pipeline.image_processor.postprocess(images, output_type="pil")

        # Encode as base64 PNG
        images_png: list[str] = []
        for img in images:
            buffer = io.BytesIO()
            assert isinstance(img, Image.Image)
            img.save(buffer, format="PNG")
            images_png.append(base64.b64encode(buffer.getvalue()).decode("ascii"))

        logger.info("SP rank 0: generated %d images successfully.", len(images_png))
        return images_png

    return None
