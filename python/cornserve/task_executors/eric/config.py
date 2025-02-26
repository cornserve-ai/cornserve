"""Configuration for the Eric task executor.

Config values will be supplied by the Task Manager when Eric is launched.
"""

from __future__ import annotations

from pydantic import BaseModel, NonNegativeInt, PositiveInt, model_validator

from cornserve.task_executors.eric.schema import Modality


class ModelConfig(BaseModel):
    """Config related to instantiating and executing the model."""

    # Hugging Face model ID
    id: str

    # Tensor parallel degree
    tp_size: PositiveInt = 1


class ServerConfig(BaseModel):
    """Serving config."""

    # Host to bind to
    host: str = "0.0.0.0"

    # Port to bind to
    port: PositiveInt = 8000


class ModalityConfig(BaseModel):
    """Modality processing config."""

    # Modality to process
    ty: Modality = Modality.IMAGE

    # Number of modality processing workers to spawn
    num_workers: PositiveInt = 12


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
    def audit(self) -> EricConfig:
        """Audit the config for correctness."""
        if self.model.tp_size != len(self.sidecar.ranks):
            raise ValueError(
                f"Tensor parallel rank ({self.model.tp_size}) "
                f"must match number of sender sidecar ranks ({self.sidecar.ranks})"
            )

        return self
