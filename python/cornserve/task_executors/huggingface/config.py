"""Configuration for HuggingFace task executor."""

from __future__ import annotations

from dataclasses import dataclass

from cornserve.task_executors.huggingface.api import ModelType


@dataclass
class ServerConfig:
    """Server configuration.

    Attributes:
        host: Host to bind to.
        port: Port to listen on.
    """

    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class ModelConfig:
    """Model configuration.

    Attributes:
        id: Model ID to load.
        max_batch_size: Maximum batch size for inference.
    """

    id: str
    model_type: ModelType
    max_batch_size: int = 1


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
