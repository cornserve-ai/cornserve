"""QwenImage model implementation for Geri."""

from __future__ import annotations

import base64
import io
import os
from typing import Any, cast

import torch
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig

from cornserve.logging import get_logger
from cornserve.task_executors.geri.executor.worker import broadcast_generate_command
from cornserve.task_executors.geri.models.base import BatchGeriModel
from cornserve.task_executors.geri.models.sp_wrapper import execute_sp_generate, patch_pipeline_for_sp

logger = get_logger(__name__)

ENV_MAX_DOUBLE_BLOCKS = "CORNSERVE_GERI_QWEN_IMAGE_MAX_DOUBLE_BLOCKS"
ENV_MAX_SINGLE_BLOCKS = "CORNSERVE_GERI_QWEN_IMAGE_MAX_SINGLE_BLOCKS"
ENV_EAGER_TEXT_ENCODER = "CORNSERVE_GERI_QWEN_IMAGE_EAGER_TEXT_ENCODER"


def env_flag(name: str) -> bool:
    """Return True if the environment variable is set to a truthy value."""
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def read_positive_int_env(name: str) -> int | None:
    """Read an environment variable as a positive integer, returning None if unset."""
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None

    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got: {value!r}") from exc

    if parsed <= 0:
        raise ValueError(f"Environment variable {name} must be > 0, got: {parsed}")

    return parsed


def apply_qwen_image_dev_layer_overrides(pipeline: QwenImagePipeline) -> None:
    """Apply optional development-only DiT layer caps via environment variables."""
    max_double_blocks = read_positive_int_env(ENV_MAX_DOUBLE_BLOCKS)
    max_single_blocks = read_positive_int_env(ENV_MAX_SINGLE_BLOCKS)

    if max_double_blocks is not None and hasattr(pipeline.transformer, "transformer_blocks"):
        double_blocks = pipeline.transformer.transformer_blocks
        if max_double_blocks < len(double_blocks):
            pipeline.transformer.transformer_blocks = torch.nn.ModuleList(list(double_blocks)[:max_double_blocks])
            logger.warning(
                "Applying dev layer cap for Qwen-Image double-stream blocks: %d -> %d (%s)",
                len(double_blocks),
                max_double_blocks,
                ENV_MAX_DOUBLE_BLOCKS,
            )

    if max_single_blocks is not None and hasattr(pipeline.transformer, "single_transformer_blocks"):
        single_blocks = pipeline.transformer.single_transformer_blocks
        if max_single_blocks < len(single_blocks):
            pipeline.transformer.single_transformer_blocks = torch.nn.ModuleList(
                list(single_blocks)[:max_single_blocks]
            )
            logger.warning(
                "Applying dev layer cap for Qwen-Image single-stream blocks: %d -> %d (%s)",
                len(single_blocks),
                max_single_blocks,
                ENV_MAX_SINGLE_BLOCKS,
            )


class QwenImageModel(BatchGeriModel):
    """Qwen-Image model implementation for text-to-image generation.

    This model supports two input modes:
    - Prompt-embedding mode, where embeddings are received externally (via sidecar).
    - Direct-text mode, where Geri performs prompt encoding internally.

    Supports sequence parallelism (SP) when ``sp_size > 1``. The pipeline's
    transformer attention layers are patched with SP-aware processors, and
    latent tokens are split across GPUs during denoising.
    """

    def __init__(
        self,
        model_id: str,
        torch_dtype: torch.dtype,
        torch_device: torch.device,
        config: PretrainedConfig | None = None,  # ignore
    ) -> None:
        """Initialize the model with its ID and data type.

        Args:
            model_id: Hugging Face model ID.
            torch_dtype: Data type for model weights (e.g., torch.bfloat16).
            torch_device: Device to load the model on (e.g., torch.device("cuda")).
            config: If supplied, may be used to configure the model's components.
        """
        logger.info("Loading QwenImage model from %s", model_id)

        self.model_id = model_id
        self._torch_dtype = torch_dtype
        self._torch_device = torch_device
        self.eager_text_encoder = env_flag(ENV_EAGER_TEXT_ENCODER)

        # First load on CPU to avoid allocating GPU memory for unused components.
        pipeline = QwenImagePipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
        apply_qwen_image_dev_layer_overrides(pipeline)

        self._embedding_dim = pipeline.text_encoder.config.hidden_size

        if self.eager_text_encoder:
            self.text_encoder_loaded = True
        else:
            self.drop_text_encoder_components(pipeline)
            self.text_encoder_loaded = False

        if torch_device.type not in ["cpu", "meta"]:
            pipeline = pipeline.to(torch_device)

        pipeline.vae.enable_tiling()
        self.pipeline = pipeline
        self._sp_group = None  # Set by patch_for_sp() if sp_size > 1

        logger.info("QwenImage model loaded successfully")

    @staticmethod
    def drop_text_encoder_components(pipeline: QwenImagePipeline) -> None:
        """Drop text-encoder components from a pipeline to save memory."""
        pipeline.text_encoder = None
        pipeline.tokenizer = None

    def ensure_text_encoder_loaded(self) -> None:
        """Lazily load text encoder + tokenizer for direct-text mode."""
        if self.text_encoder_loaded:
            return

        logger.info("Loading Qwen-Image text encoder lazily for direct-text mode.")
        text_encoder = cast(
            Any,
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                subfolder="text_encoder",
                torch_dtype=self._torch_dtype,
            ),
        )

        if self._torch_device.type not in ["cpu", "meta"]:
            text_encoder = cast(Any, text_encoder.to(device=self._torch_device))

        text_encoder.eval()
        tokenizer = Qwen2Tokenizer.from_pretrained(self.model_id, subfolder="tokenizer")

        self.pipeline.text_encoder = text_encoder
        self.pipeline.tokenizer = tokenizer
        self.text_encoder_loaded = True

    @torch.inference_mode()
    def encode_prompts(self, prompts: list[str]) -> list[torch.Tensor]:
        """Encode raw text prompts into per-request embedding tensors."""
        if not prompts:
            return []

        stripped_prompts = [prompt.strip() for prompt in prompts]
        if any(not prompt for prompt in stripped_prompts):
            raise ValueError("Prompt text must be non-empty.")

        self.ensure_text_encoder_loaded()

        prompt_embeds, prompt_embeds_mask = self.pipeline.encode_prompt(
            prompt=stripped_prompts,
            num_images_per_prompt=1,
        )

        prompt_embeds = prompt_embeds.to(device=self._torch_device, dtype=self._torch_dtype)
        if prompt_embeds_mask is None:
            prompt_embeds_mask = torch.ones(prompt_embeds.shape[:2], device=self._torch_device, dtype=torch.long)
        else:
            prompt_embeds_mask = prompt_embeds_mask.to(device=self._torch_device)

        per_request_embeds: list[torch.Tensor] = []
        for batch_idx in range(prompt_embeds.shape[0]):
            seq_len = int(prompt_embeds_mask[batch_idx].sum().item())
            per_request_embeds.append(prompt_embeds[batch_idx, :seq_len, :].contiguous())

        return per_request_embeds

    def patch_for_sp(self, sp_group) -> None:
        """Patch the pipeline for sequence parallelism.

        Called by the engine after model loading when ``sp_size > 1``.

        Args:
            sp_group: The :class:`SPGroup` instance.
        """
        patch_pipeline_for_sp(self.pipeline, sp_group)
        self._sp_group = sp_group
        logger.info("QwenImage model patched for SP (sp_size=%d).", sp_group.world_size)

    @property
    def sp_enabled(self) -> bool:
        """Whether sequence parallelism is active."""
        return self._sp_group is not None and self._sp_group.world_size > 1

    @property
    def dtype(self) -> torch.dtype:
        """The data type of the model."""
        return self.pipeline.dtype

    @property
    def device(self) -> torch.device:
        """The device where the model is loaded."""
        return self.pipeline.device

    @property
    def embedding_dim(self) -> int:
        """The dimension of the prompt embeddings used by the model."""
        return self._embedding_dim

    def generate(
        self,
        prompt_embeds: list[torch.Tensor],
        height: int,
        width: int,
        num_inference_steps: int = 50,
    ) -> list[str]:
        """Generate images from prompt embeddings.

        When SP is enabled, this method distributes denoising across SP ranks
        and only rank 0 returns the decoded images.

        Args:
            prompt_embeds: List of [seq_len, hidden_size] tensors, one per batch item.
            height: Height of the generated image in pixels.
            width: Width of the generated image in pixels.
            num_inference_steps: Number of denoising steps to perform.

        Returns:
            Generated images as base64-encoded PNG strings.
        """
        batch_size = len(prompt_embeds)
        max_seq_len = max(emb.size(0) for emb in prompt_embeds)

        # Pad and batch prompt embeddings
        padded_embeds = pad_sequence(prompt_embeds, batch_first=True, padding_value=0.0)
        lengths = torch.tensor([emb.size(0) for emb in prompt_embeds], device=self._torch_device)
        prompt_embeds_mask = (
            torch.arange(max_seq_len, device=self._torch_device).unsqueeze(0) < lengths.unsqueeze(1)
        ).long()

        logger.info(
            "Generating %d images with size %dx%d, %d inference steps (SP=%s)",
            batch_size,
            height,
            width,
            num_inference_steps,
            self.sp_enabled,
        )

        if self.sp_enabled:
            return self._generate_sp(padded_embeds, prompt_embeds_mask, height, width, num_inference_steps)

        return self._generate_single(padded_embeds, prompt_embeds_mask, height, width, num_inference_steps)

    def _generate_single(
        self,
        padded_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        height: int,
        width: int,
        num_inference_steps: int,
    ) -> list[str]:
        """Single-GPU generation (original path)."""
        result = self.pipeline(
            prompt_embeds=padded_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
        )

        # Check type and convert images to PNG bytes
        assert isinstance(result, QwenImagePipelineOutput)
        images = result.images
        assert isinstance(images, list) and all(isinstance(img, Image.Image) for img in images)

        images_png: list[str] = []
        for img in images:
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            images_png.append(base64.b64encode(buffer.getvalue()).decode("ascii"))

        logger.info("Generated %d images successfully", len(images_png))
        return images_png

    def _generate_sp(
        self,
        padded_embeds: torch.Tensor,
        prompt_embeds_mask: torch.Tensor,
        height: int,
        width: int,
        num_inference_steps: int,
    ) -> list[str]:
        """SP-parallel generation (rank 0 drives, all ranks participate)."""
        assert self._sp_group is not None
        batch_size = padded_embeds.shape[0]
        max_seq_len = padded_embeds.shape[1]

        # Broadcast generation command to non-rank-0 workers so they enter
        # the same pipeline call and participate in NCCL collectives.
        broadcast_generate_command(
            sp_group=self._sp_group,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
        )

        # Execute SP generation (rank 0 has the actual embeddings)
        result = execute_sp_generate(
            pipeline=self.pipeline,
            sp_group=self._sp_group,
            prompt_embeds=padded_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
        )

        assert result is not None, "Rank 0 should always return results."
        if not isinstance(result, list) or not all(isinstance(item, str) for item in result):
            raise RuntimeError("SP image generation returned an unexpected output type.")
        return result

    @staticmethod
    def find_embedding_dim(model_id: str, config: PretrainedConfig | None = None) -> int:
        """Find the embedding dimension of the model as indicated in HF configs.

        Used for obtaining the embedding dimension without instantiating the model.

        Args:
            model_id: Will be used to obtain the hidden size from HF.
            config: If supplied, the lookup to HF using model_id will be skipped, and the
                hidden size will be extracted directly from config.
        """
        if isinstance(config, Qwen2_5_VLConfig):
            return config.hidden_size
        config = AutoConfig.from_pretrained(
            model_id,
            subfolder="text_encoder",
            trust_remote_code=True,
        )
        return config.hidden_size
