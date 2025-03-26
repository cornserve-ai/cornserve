"""Configuration for the Eric task executor.

Config values will be supplied by the Task Manager when Eric is launched.

Config values should be kept in sync with `cornserve.task_executors.launch`.
"""

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt, model_validator
from transformers import AutoConfig, PretrainedConfig
from typing_extensions import Self


class ModelConfig(BaseModel):
    """Config related to instantiating and executing the model."""

    # Hugging Face model ID
    id: str

    # Tensor parallel degree
    tp_size: PositiveInt = 1

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

    # Number of frames to sample from the video
    max_num_frames: PositiveInt = 32


class ModalityConfig(BaseModel):
    """Modality processing config."""

    # Number of modality processing workers to spawn
    num_workers: PositiveInt = 12

    # Image-specific processor config
    image_config: ImageDataConfig = ImageDataConfig()

    # Video-specific processor config
    video_config: VideoDataConfig = VideoDataConfig()

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
        """Audit the config for correctness."""
        if self.model.tp_size != len(self.sidecar.ranks):
            raise ValueError(
                f"Tensor parallel rank ({self.model.tp_size}) "
                f"must match number of sender sidecar ranks ({self.sidecar.ranks})"
            )

        return self
