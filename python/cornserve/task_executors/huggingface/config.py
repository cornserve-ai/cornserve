"""Configuration for HuggingFace task executor."""

from __future__ import annotations

from pydantic import BaseModel, PositiveInt, field_validator

from cornserve.task_executors.huggingface.api import ModelType


class ServerConfig(BaseModel):
    """Server configuration.

    Attributes:
        host: Host to bind to.
        port: Port to listen on.
        max_batch_size: Maximum batch size for inference.
    """

    host: str = "0.0.0.0"
    port: int = 8000
    max_batch_size: PositiveInt = 1

    @field_validator("max_batch_size")
    @classmethod
    def _validate_max_batch_size(cls, v: int) -> int:
        if v > 1:
            raise ValueError("max_batch_size > 1 is not supported yet")
        return v


class ModelConfig(BaseModel):
    """Model configuration.

    Attributes:
        id: Model ID to load.
        model_type: Type of model.
    """

    id: str
    model_type: ModelType


class HuggingFaceConfig(BaseModel):
    """Configuration for HuggingFace task executor.

    Attributes:
        model: Model configuration.
        server: Server configuration.
    """

    model: ModelConfig
    server: ServerConfig
