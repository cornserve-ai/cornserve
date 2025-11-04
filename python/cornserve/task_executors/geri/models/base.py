"""Base class for Geri models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator

import torch
from transformers.configuration_utils import PretrainedConfig


class GeriModel(ABC):
    """Base class for all Geri generative models."""

    @abstractmethod
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
            config: If supplied, may be used to configure the model's components.
        """

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """The data type of the model."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """The device where the model is loaded."""

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """The dimension of the prompt embeddings used by the model."""

    @staticmethod
    @abstractmethod
    def find_embedding_dim(model_id: str, config: PretrainedConfig | None = None) -> int:
        """Find the embedding dimension of the model as indicated in HF configs.

        Used for obtaining the embedding dimension without instantiating the model.

        Args:
            model_id: Will be used to obtain the hidden size from HF.
            config: If supplied, the lookup to HF using model_id will be skipped, and the
                hidden size will be extracted directly from config.
        """


class BatchGeriModel(GeriModel):
    """Geri Model that does not stream.

    Expects full inputs and returns outputs all at once.
    """

    # TODO: generalize to handle more flexible inputs
    @abstractmethod
    def generate(
        self,
        prompt_embeds: list[torch.Tensor],
        height: int,
        width: int,
        num_inference_steps: int = 50,
    ) -> list[str]:
        """Generate images from prompt embeddings.

        Args:
            prompt_embeds: Text embeddings from the LLM encoder.
                List of [seq_len, hidden_size] tensors, one per batch item.
            height: Height of the generated image in pixels.
            width: Width of the generated image in pixels.
            num_inference_steps: Number of denoising steps to perform.

        Returns:
            Generated multimodal content as base64-encoded bytes.
            For images, bytes are in PNG format.
        """


class StreamGeriModel(GeriModel):
    """Geri Model that streams."""

    # TODO: generalize to handle more flexible inputs
    @abstractmethod
    def generate(
        self,
        prompt_embeds: list[torch.Tensor],
        chunk_size: int,
        left_context_size: int,
    ) -> Generator[list[torch.Tensor | None], None, None]:
        """Generate streamed outputs from prompt embeddings."""
