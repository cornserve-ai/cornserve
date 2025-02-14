from dataclasses import dataclass

from cornserve.task_executors.eric.models import Modality


@dataclass
class EricConfig:
    """Eric encodes multimodal data into embeddings."""

    # Hugging Face model ID
    model_id: str

    # Modality to process
    modality: Modality = Modality.IMAGE

    # Tensor parallel degree
    tp_size: int = 1

    # Host to bind to
    host: str = "0.0.0.0"

    # Port to bind to
    port: int = 8000

    # Number of modality processing workers to spawn
    num_modality_workers: int = 12
