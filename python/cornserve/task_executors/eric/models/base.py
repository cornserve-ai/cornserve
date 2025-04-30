"""Base class for all models in Eric."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import torch
import torch.nn as nn
import numpy.typing as npt

from cornserve.task_executors.eric.schema import Modality


class EricModel(nn.Module, ABC):
    """Base class for all models in Eric."""

    @abstractmethod
    def forward(self, modality: Modality, batch: dict[str, list[torch.Tensor]]) -> list[torch.Tensor]:
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
    def device(self) -> torch.device:
        """Return the device where inputs should be in."""

    @property
    @abstractmethod
    def chunk_shape(self) -> tuple[int, ...]:
        """Return the shape of the chunks to be sent to the sidecar."""


class BaseModalityProcessor:
    """Base class for modality processors.

    Each model definition module contains a `ModalityProcessor` class that
    inherits from this class. It should override `get_image_processor`,
    `get_video_processor`, etc. to return the appropriate processor for the
    given modality. The processor should be a callable that takes the input
    modality data as a Numpy array and returns the processed data as
    """

    def __init__(self, model_id: str) -> None:
        """Initialize the processor."""
        self.model_id = model_id

    def get_image_processor(self) -> Callable | None:
        """Get the image processor for this modality."""
        return None

    def get_video_processor(self) -> Callable | None:
        """Get the video processor for this modality."""
        return None

    def process(self, modality: Modality, data: npt.NDArray) -> dict[str, npt.NDArray]:
        """Process the input data for the given modality."""
        match modality:
            case Modality.IMAGE:
                image_processor = self.get_image_processor()
                if image_processor is None:
                    raise ValueError("Image processor not available.")
                return image_processor(data)
            case Modality.VIDEO:
                video_processor = self.get_video_processor()
                if video_processor is None:
                    raise ValueError("Video processor not available.")
                return video_processor(data)
            case _:
                raise ValueError(f"Unsupported modality: {modality}")
