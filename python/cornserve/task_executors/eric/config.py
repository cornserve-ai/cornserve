"""Configuration for the Eric task executor.

Config values will be supplied by the Task Manager when Eric is launched.

Config values should be kept in sync with the built-in `EricDescriptor`.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt, model_validator
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.processing_auto import AutoProcessor

from cornserve.task_executors.eric.api import Modality


class ModelConfig(BaseModel):
    """Config related to instantiating and executing the model."""

    # Hugging Face model ID
    id: str

    # Tensor parallel degree
    tp_size: PositiveInt = 1

    # Modality type
    modality: Modality

    # HF config
    # This will be replaced with the real HF config of the model ID
    _hf_config: PretrainedConfig

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context) -> None:
        """Post-init hook to load the HF config."""
        # Load the HF config
        try:
            self._hf_config = AutoConfig.from_pretrained(self.id)
        except Exception as e:
            raise ValueError(f"Failed to load HF config for model {self.id}") from e

    @property
    def hf_config(self) -> PretrainedConfig:
        """Return the HF config."""
        if self._hf_config is None:
            raise ValueError("HF config not loaded")
        return self._hf_config


class ServerConfig(BaseModel):
    """Serving config."""

    # Host to bind to
    host: str = "0.0.0.0"

    # Port to bind to
    port: PositiveInt = 8000


class ImageDataConfig(BaseModel):
    """Configuration related to downloading and processing image data."""


class VideoDataConfig(BaseModel):
    """Configuration related to downloading and processing video data."""

    # Number of frames to sample from the video (`None` for all frames)
    max_num_frames: PositiveInt | None = 32


class AudioDataConfig(BaseModel):
    """Configuration related to downloading and processing audio data."""

    # Sampling rate to use when loading audio data.
    # When the server is configured to embed audio data, this will be
    # populated from the HF AutoProcessor's attribute so match the model's
    # expected sampling rate.
    sampling_rate: PositiveInt | None = None


class ModalityConfig(BaseModel):
    """Modality processing config."""

    # Number of modality processing workers to spawn
    num_workers: PositiveInt = 12

    # Image-specific processor config
    image_config: ImageDataConfig = ImageDataConfig()

    # Video-specific processor config
    video_config: VideoDataConfig = VideoDataConfig()

    # Audio-specific processor config
    audio_config: AudioDataConfig = AudioDataConfig()

    @model_validator(mode="after")
    def validator(self) -> Self:
        """Validate the config for correctness."""
        if self.image_config is None and self.video_config is None:
            raise ValueError("At least one modality processor config must be set.")

        return self


class SidecarConfig(BaseModel):
    """Sidecar config for the engine."""

    # The sender sidecar ranks to register with
    ranks: list[NonNegativeInt]


class EricConfig(BaseModel):
    """Main configuration class for Eric."""

    model: ModelConfig
    server: ServerConfig
    modality: ModalityConfig
    sidecar: SidecarConfig

    @model_validator(mode="after")
    def validator(self) -> Self:
        """Audit the config for correctness and apply any transformations."""
        if self.model.tp_size != len(self.sidecar.ranks):
            raise ValueError(
                f"Tensor parallel rank ({self.model.tp_size}) "
                f"must match number of sender sidecar ranks ({self.sidecar.ranks})"
            )

        if self.model.modality == Modality.AUDIO and self.modality.audio_config.sampling_rate is None:
            processor = AutoProcessor.from_pretrained(self.model.id)
            feature_extractor = getattr(processor, "feature_extractor", None)
            if feature_extractor is None:
                raise ValueError(
                    "In order to determine the audio sampling rate, we attempted to "
                    "access the HF AutoProcessor's `feature_extractor`, but it didn't exist."
                )
            sampling_rate = getattr(feature_extractor, "sampling_rate", None)
            if sampling_rate is None:
                raise ValueError(
                    "In order to determine the audio sampling rate, we attempted to "
                    "access the HF AutoProcessor's `feature_extractor.sampling_rate`, "
                    "but it didn't exist."
                )
            self.modality.audio_config.sampling_rate = sampling_rate

        return self
