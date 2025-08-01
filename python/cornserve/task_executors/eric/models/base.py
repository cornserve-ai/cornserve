"""Base class for all models in Eric."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeVar, Generic

import torch
import torch.nn as nn
import numpy.typing as npt
from transformers.configuration_utils import PretrainedConfig

from cornserve.task_executors.eric.schema import Modality


class EricModel(nn.Module, ABC):
    """Base class for all models in Eric."""

    @abstractmethod
    def forward(
        self,
        modality: Modality,
        adapter_name: str,
        batch: dict[str, list[torch.Tensor]],
    ) -> list[torch.Tensor]:
        """Forward pass for the model.

        Args:
            modality: The modality of the data.
            adapter_name: The name of the adapter to use.
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


ConfigT = TypeVar("ConfigT", bound=PretrainedConfig)


class BaseAdapterModule(nn.Module, Generic[ConfigT], ABC):
    """Base class for a adapter module."""

    def __init__(self, adapter_cls: Callable[[ConfigT], nn.Module]) -> None:
        """Initialize the adapter module."""
        super().__init__()
        self.default_adapter_name: str | None = None
        self.adapter_cls = adapter_cls
        self.adapters: dict[str, nn.Module] = {}

    def make_adapter(self, name: str, config: ConfigT) -> nn.Module:
        """Create an adapter with the name and init arguments and return it."""
        if self.default_adapter_name is None:
            self.default_adapter_name = name
        adapter = self.adapter_cls(config).eval()
        self.adapters[name] = adapter
        return adapter

    def forward(self, name: str, *args, **kwargs):
        """Forward pass for the adapter with the given name."""
        if name not in self.adapters:
            raise ValueError(f"Adapter '{name}' not found.")
        return self.adapters[name](*args, **kwargs)

    def named_modules(
        self,
        memo: set[nn.Module] | None = None,
        prefix: str = "",
        remove_duplicate: bool = True,
    ):
        """Override named_modules to return modules of the default adapter."""
        if self.default_adapter_name is None:
            raise ValueError("No default adapter name set.")
        return self.adapters[self.default_adapter_name].named_modules(memo, prefix, remove_duplicate)


class BaseModalityProcessor:
    """Base class for modality processors.

    Each model definition module contains a `ModalityProcessor` class that
    inherits from this class. It should override `get_image_processor`,
    `get_video_processor`, etc. to return the appropriate processor for the
    given modality. The processor should be a callable that takes the input
    modality data as a Numpy array and returns the processed data as a
    dictionary of Numpy arrays.
    """

    def __init__(self, model_id: str) -> None:
        """Initialize the processor."""
        self.model_id = model_id

    def get_image_processor(self) -> Callable | None:
        """Get the image processor for this modality.

        The callable sould take a single image numpy array.
        """
        return None

    def get_audio_processor(self) -> Callable | None:
        """Get the audio processor for this modality.

        The callable should take a tuple of (audio data numpy array, sample rate).
        """
        return None

    def get_video_processor(self) -> Callable | None:
        """Get the video processor for this modality.

        The callable should take a single video numpy array.
        """
        return None

    def process(self, modality: Modality, data: npt.NDArray) -> dict[str, torch.Tensor]:
        """Process the input data for the given modality."""
        match modality:
            case Modality.IMAGE:
                image_processor = self.get_image_processor()
                if image_processor is None:
                    raise ValueError("Image processor not available.")
                return image_processor(data)
            case Modality.AUDIO:
                audio_processor = self.get_audio_processor()
                if audio_processor is None:
                    raise ValueError("Audio processor not available.")
                return audio_processor(data)
            case Modality.VIDEO:
                video_processor = self.get_video_processor()
                if video_processor is None:
                    raise ValueError("Video processor not available.")
                return video_processor(data)
            case _:
                raise ValueError(f"Unsupported modality: {modality}")
