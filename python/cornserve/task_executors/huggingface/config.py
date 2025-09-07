"""Configuration for HuggingFace task executor."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import PositiveInt, field_validator

from cornserve.task_executors.huggingface.api import ModelType


@dataclass
class ServerConfig:
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


@dataclass
class ModelConfig:
    """Model configuration.

    Attributes:
        id: Model ID to load.
        model_type: Type of model.
    """

    id: str
    model_type: ModelType


@dataclass
class HuggingFaceConfig:
    """Configuration for HuggingFace task executor.

    Attributes:
        task_type: Type of task to execute.
        model: Model configuration.
        server: Server configuration.
    """

    model: ModelConfig
    server: ServerConfig
