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
from cornserve.task_executors.geri.models.base import StreamGeriModel
from cornserve.task_executors.utils import get_safetensors_weight_dict, set_default_torch_dtype

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

        # We already have the right config type
        elif isinstance(config, Qwen3OmniMoeCode2WavConfig):
            model_config = config

        # Handles None config and any other cases
        else:
            try:
                model_config = Qwen3OmniMoeCode2Wav._get_config(model_id)
            except Exception as e:
                raise FileNotFoundError(f"Could not load model {model_id}: {e}") from e

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
        self.num_quantizers = config.num_quantizers

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

    def generate(
        self,
        prompt_embeds: list[torch.Tensor],
        chunk_size: int | None = None,
        left_context_size: int | None = None,
    ) -> Generator[list[torch.Tensor | None], None, None]:
        """Generate streamed outputs from prompt embeddings."""
        if chunk_size is None:
            chunk_size = 300
        if left_context_size is None:
            left_context_size = 25

        # Each element of `prompt_embeds` has shape (seqlen, num_quantizers).
        # First, we transpose to (num_quantizers, seqlen).
        prompt_embeds = [torch.transpose(embed, 0, 1).to(dtype=int) for embed in prompt_embeds]

        # To batch, we pad them all to the same seqlen and then stack them along dim 0.
        code_lens = [embed.shape[-1] for embed in prompt_embeds]
        max_seqlen = max(code_lens)
        prompt_embeds = [
            nn.functional.pad(embed, (0, max_seqlen - embed.shape[-1]), value=0) for embed in prompt_embeds
        ]
        codes = torch.stack(prompt_embeds, dim=0)

        start_index = 0
        while start_index < codes.shape[-1]:
            end_index = min(start_index + chunk_size, codes.shape[-1])
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self(codes_chunk)

            # Unbatch, and for each sequence, slice up to where it was really supposed to end
            context_end = context_size * self.total_upsample
            res = []
            for i, chunk in enumerate(wav_chunk):
                if start_index < code_lens[i]:
                    # slice_size = upsample * (number of real unpadded codes)
                    slice_size = self.total_upsample * (min(end_index, code_lens[i]) - start_index)
                    res.append(chunk[..., context_end : context_end + slice_size])
                else:
                    # If a request is finished early, indicate with None
                    res.append(None)
            yield res
            start_index = end_index

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
        return self.num_quantizers

    @staticmethod
    def _get_config(model_id: str) -> Qwen3OmniMoeCode2WavConfig:
        """Fetches config from HF."""
        hf_config: PretrainedConfig = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        if not isinstance(hf_config, Qwen3OmniMoeConfig):
            raise TypeError(f"Expected Qwen3OmniMoeConfig, but got {type(hf_config).__name__} instead.")
        return hf_config.code2wav_config

    @staticmethod
    def find_embedding_dim(model_id: str, config: PretrainedConfig | None = None) -> int:
        """Find the embedding dimension of the model as indicated in HF configs.

        Used for obtaining the embedding dimension without instantiating the model.

        Args:
            model_id: Will be used to obtain the hidden size from HF.
            config: If supplied, the lookup to HF using model_id will be skipped, and the
                hidden size will be extracted directly from config.
        """
        if isinstance(config, Qwen3OmniMoeConfig):
            return config.code2wav_config.num_quantizers
        if isinstance(config, Qwen3OmniMoeCode2WavConfig):
            return config.num_quantizers
        return Qwen3OmniMoeCode2Wav._get_config(model_id).num_quantizers
