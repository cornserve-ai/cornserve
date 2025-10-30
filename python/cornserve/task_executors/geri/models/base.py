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
    def generate(self, prompt_embeds: list[torch.Tensor]) -> Generator[torch.Tensor, None, None]:
        """Generate streamed outputs from prompt embeddings."""
