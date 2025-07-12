"""Build-in task for modality encoders."""

from __future__ import annotations

import enum

from cornserve.task.base import TaskInput, TaskOutput, UnitTask
from cornserve.task.forward import DataForward, Tensor


class Modality(enum.StrEnum):
    """Supported modalities for encoder tasks."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class EncoderInput(TaskInput):
    """Input model for encoder tasks.

    Attributes:
        model_id: The ID of the model to use for the task. If this is an
            adapter-supported model, this should be the name of the adapter.
        data_urls: The URLs of the data to encode.
    """

    model_id: str
    data_urls: list[str]


class EncoderOutput(TaskOutput):
    """Output model for encoder tasks.

    Attributes:
        embeddings: The embeddings from the encoder.
    """

    embeddings: list[DataForward[Tensor]]


class EncoderTask(UnitTask[EncoderInput, EncoderOutput]):
    """A task that invokes an encoder.

    Attributes:
        modality: Modality of data this encoder can embed.
        model_id: The ID of the model to use for the task.
        adapter_model_ids: Some models support multiple adapters and allow the
            base model to be shared (e.g., Gemma 3). This list specifies model IDs
            from which to load adapters. Base model weights are loaded from `model_id`.
        max_batch_size: Maximum batch size to use for the serving system.
    """

    modality: Modality
    model_id: str
    adapter_model_ids: list[str] = []
    max_batch_size: int = 1

    def make_record_output(self, task_input: EncoderInput) -> EncoderOutput:
        """Create a task output for task invocation recording."""
        return EncoderOutput(embeddings=[DataForward[Tensor]() for _ in task_input.data_urls])

    def validate_input(self, task_input: EncoderInput) -> None:
        """Validate the input for the encoder task."""
        if not task_input.data_urls:
            raise ValueError("Data URLs cannot be empty.")

        if task_input.model_id not in [self.model_id, *self.adapter_model_ids]:
            raise ValueError(
                f"Model ID {task_input.model_id} does not match task model ID {self.model_id} "
                f"or any adapter model IDs {self.adapter_model_ids}."
            )

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        return f"encoder-{self.modality.lower()}-{self.model_id.split('/')[-1].lower()}"
