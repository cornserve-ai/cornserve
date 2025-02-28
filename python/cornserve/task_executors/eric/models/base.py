"""Base class for all models in Eric."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from cornserve.task_executors.eric.schema import Modality


class EricModel(nn.Module, ABC):
    """Base class for all models in Eric."""

    @abstractmethod
    def forward(
        self, modality: Modality, batch: dict[str, list[torch.Tensor]]
    ) -> list[torch.Tensor]:
        """Forward pass for the model.

        Args:
            modality: The modality of the data.
            batch: The input data.

        Returns:
            A list of output tensors.
        """

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype:
        """Return the data type of the model's embeddings."""

    @property
    @abstractmethod
    def chunk_shape(self) -> tuple[int, ...]:
        """Return the shape of the chunks to be sent to the sidecar."""
