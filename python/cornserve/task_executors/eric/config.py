"""Configuration for the Eric task executor.

Config values will be supplied by the Task Manager when Eric is launched.
"""

from transformers import AutoConfig, PretrainedConfig
from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt, model_validator


class ModelConfig(BaseModel):
    """Config related to instantiating and executing the model."""

    # Hugging Face model ID
    id: str

    # Tensor parallel degree
    tp_size: PositiveInt = 1

    # HF config
    # This will be replaced with the real HF config of the model ID
    hf_config: PretrainedConfig = PretrainedConfig()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validator(self) -> Self:
        """Validate the config for correctness."""
        # Load the HF config
        try:
            self.hf_config = AutoConfig.from_pretrained(self.id)
        except Exception as e:
            raise ValueError(f"Failed to load HF config for model {self.id}") from e

        return self


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
