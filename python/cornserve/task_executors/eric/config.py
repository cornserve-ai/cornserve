"""Configuration for the Eric task executor.

Config values will be supplied by the Task Manager when Eric is launched.
"""

from dataclasses import dataclass

from cornserve.task_executors.eric.schema import Modality


@dataclass
class ModelConfig:
    """Config related to instantiating and executing the model."""

    # Hugging Face model ID
    id: str

    # Tensor parallel degree
    tp_size: int = 1


@dataclass
class ServerConfig:
    """Serving config."""

    # Host to bind to
    host: str = "0.0.0.0"

    # Port to bind to
    port: int = 8000


@dataclass
class ModalityConfig:
    """Modality processing config."""

    # Modality to process
    ty: Modality = Modality.IMAGE

    # Number of modality processing workers to spawn
    num_workers: int = 12


@dataclass
class SidecarConfig:
    """Sidecar config for the engine."""

    # The sender sidecar ranks to register with
    ranks: list[int]


@dataclass
class EricConfig:
    """Eric encodes multimodal data into embeddings."""

    model: ModelConfig
    server: ServerConfig
    modality: ModalityConfig
    sidecar: SidecarConfig
