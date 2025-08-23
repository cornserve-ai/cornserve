"""Base class for Geri models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class GeriModel(ABC):
    """Base class for all Geri generative models.

    All models should implement the generate method with a consistent interface
    that the engine can invoke uniformly regardless of the specific model type.
    """

    @abstractmethod
    def __init__(self, model_id: str, torch_dtype: torch.dtype, torch_device: torch.device) -> None:
        """Initialize the model with its ID and data type.

        Args:
            model_id: Hugging Face model ID.
            torch_dtype: Data type for model weights (e.g., torch.bfloat16).
            torch_device: Device to load the model on (e.g., torch.device("cuda")).
        """

    @abstractmethod
    def generate(
        self,
        prompt_embeds: list[torch.Tensor],
        height: int,
        width: int,
        num_inference_steps: int = 50,
    ) -> list[bytes]:
        """Generate images from prompt embeddings.

        Args:
            prompt_embeds: Text embeddings from the LLM encoder.
                List of [seq_len, hidden_size] tensors, one per batch item.
            height: Height of the generated image in pixels.
            width: Width of the generated image in pixels.
            num_inference_steps: Number of denoising steps to perform.

        Returns:
            Generated multimodal content as bytes. For images, bytes are in PNG format.
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
