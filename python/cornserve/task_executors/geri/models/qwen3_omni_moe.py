"""Qwen3-Omni model implementation for Geri."""

from __future__ import annotations

from collections.abc import Generator

import numpy as np
import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig,
    Qwen3OmniMoeConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCausalConvNet,
    Qwen3OmniMoeCausalTransConvNet,
    Qwen3OmniMoeCode2WavDecoderBlock,
    Qwen3OmniMoeCode2WavTransformerModel,
    Qwen3OmniMoeConvNeXtBlock,
    SnakeBeta,
)

from cornserve.logging import get_logger
from cornserve.task_executors.eric.executor.loader import (
    get_safetensors_weight_dict,
    set_default_torch_dtype,
)
from cornserve.task_executors.geri.models.base import StreamGeriModel

logger = get_logger(__name__)


class Qwen3OmniMoeCode2Wav(StreamGeriModel, nn.Module):
    """Vocoder for Qwen3-Omni that supports streaming outputs."""

    def __init__(
        self,
        model_id: str,
        torch_dtype: torch.dtype,
        torch_device: torch.device,
        config: PretrainedConfig | None = None,
    ) -> None:
        """Initialize the model with its ID and data type.

        Args:
            model_id: Hugging Face model ID.
            torch_dtype: Data type for model weights (e.g., torch.bfloat16).
            torch_device: Device to load the model on (e.g., torch.device("cuda")).
            config: If supplied, will be used to configure the model.
        """
        # A special case: the parent configuration was provided
        if isinstance(config, Qwen3OmniMoeConfig):
            model_config = config.code2wav_config

        # Handles None config and any other cases
        elif not isinstance(config, Qwen3OmniMoeCode2WavConfig):
            try:
                hf_config: PretrainedConfig = AutoConfig.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                )
                if not isinstance(hf_config, Qwen3OmniMoeCode2WavConfig):
                    raise TypeError(f"Expected Qwen3OmniMoeCode2WavConfig, but got {type(hf_config).__name__} instead.")
                model_config = hf_config.code2wav_config
            except Exception as e:
                raise FileNotFoundError(f"Could not load model {model_id}: {e}") from e

        # Otherwise, we already have the right config type
        else:
            model_config = config

        # Initialize components
        nn.Module.__init__(self)
        with set_default_torch_dtype(torch_dtype), torch_device:
            self.initialize(model_config)

        # Load weights
        weight_dict = get_safetensors_weight_dict(
            model_id,
            weight_prefixes=["code2wav."],
            strip_prefixes=True,
        )
        incompatible = self.load_state_dict(weight_dict, strict=False)
        if incompatible.missing_keys:
            raise ValueError(f"Missing weights in the model: {incompatible.missing_keys}")

    def initialize(self, config: Qwen3OmniMoeCode2WavConfig):
        """Initialize the model's components with the given config."""
        self.config = config

        self.total_upsample = np.prod(config.upsample_rates + config.upsampling_ratios)
        self.pre_transformer = Qwen3OmniMoeCode2WavTransformerModel._from_config(config)

        self.code_embedding = nn.Embedding(
            config.codebook_size * config.num_quantizers,
            config.hidden_size,
        )

        self.register_buffer(
            "code_offset", torch.arange(config.num_quantizers).view(1, -1, 1) * config.codebook_size, persistent=False
        )

        upsample = []
        for factor in config.upsampling_ratios:
            upsample.append(
                nn.ModuleList(
                    [
                        Qwen3OmniMoeCausalTransConvNet(config.hidden_size, config.hidden_size, factor, factor),
                        Qwen3OmniMoeConvNeXtBlock(config.hidden_size),
                    ]
                )
            )
        self.upsample = nn.ModuleList(upsample)

        decoder = [Qwen3OmniMoeCausalConvNet(config.hidden_size, config.decoder_dim, 7)]
        for i in range(len(config.upsample_rates)):
            decoder.append(Qwen3OmniMoeCode2WavDecoderBlock(config, i))
        output_dim = config.decoder_dim // 2 ** len(config.upsample_rates)
        decoder += [
            SnakeBeta(output_dim),
            Qwen3OmniMoeCausalConvNet(output_dim, 1, 7),
        ]
        self.decoder = nn.ModuleList(decoder)

    def forward(self, codes):
        """A single forward pass of the model."""
        if codes.shape[1] != self.config.num_quantizers:
            raise ValueError(f"Expected {self.config.num_quantizers} layer of codes, got {codes.shape[1]}")
        hidden = self.code_embedding(codes + self.code_offset).mean(1)
        hidden = self.pre_transformer(inputs_embeds=hidden).last_hidden_state
        hidden = hidden.permute(0, 2, 1)
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)

    # TODO: support input streaming as well (requires state management)
    def generate(self, prompt_embeds: list[torch.Tensor]) -> Generator[torch.Tensor, None, None]:
        """Generate streamed outputs from prompt embeddings.

        Generated wav chunks should be concatenated along the last dimension (i.e., dim=-1).
        """
        chunk_size = 300
        left_context_size = 25

        # When inputs are not streamed, prompt_embeds simply holds the full input
        codes = prompt_embeds[0] if len(prompt_embeds) == 1 else torch.cat(prompt_embeds, dim=-1)

        start_index = 0
        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self(codes_chunk)
            start_index = end_index
            yield wav_chunk

    @property
    def dtype(self) -> torch.dtype:
        """The data type of the model."""
        return self.code_embedding.weight.dtype

    @property
    def device(self) -> torch.device:
        """The device where the model is loaded."""
        return self.code_embedding.weight.device

    @property
    def embedding_dim(self) -> int:
        """The dimension of the prompt embeddings used by the model."""
        return self.code_embedding.embedding_dim
