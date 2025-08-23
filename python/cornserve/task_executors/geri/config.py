"""Configuration for the Geri task executor.

Config values will be supplied by the Task Manager when Geri is launched.

Config values should be kept in sync with the built-in `GeriDescriptor`.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, NonNegativeInt, PositiveInt, model_validator

from cornserve.task_executors.geri.api import Modality


class ModelConfig(BaseModel):
    """Config related to instantiating and executing the model."""

    # Hugging Face model ID
    id: str

    # Modality type
    modality: Modality

    # Embedding dimension
    _embed_dim: int

    def model_post_init(self, context, /) -> None:
        """Post-initialization hook for the model."""
        pass


class ServerConfig(BaseModel):
    """Serving config."""

    # Host to bind to
    host: str = "0.0.0.0"

    # Port to bind to
    port: PositiveInt = 8000

    # Maximum batch size to run the generator with
    max_batch_size: PositiveInt | None = 1


class SidecarConfig(BaseModel):
    """Sidecar config for the engine."""

    # The receiver sidecar ranks to register with
    ranks: list[NonNegativeInt]


class GeriConfig(BaseModel):
    """Main configuration class for Geri."""

    model: ModelConfig
    server: ServerConfig
    sidecar: SidecarConfig

    @model_validator(mode="after")
    def validator(self) -> Self:
        """Audit the config for correctness and apply any transformations."""
        # For now, we don't use tensor parallelism, so we just need one rank
        if len(self.sidecar.ranks) != 1:
            raise ValueError(f"Currently only single-rank sidecar is supported, got {len(self.sidecar.ranks)} ranks")

        return self
