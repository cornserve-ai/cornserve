"""Build-in task for modality encoders."""

from __future__ import annotations

import enum

from cornserve.task.base import TaskInput, TaskOutput, UnitTask
from cornserve.task.forward import DataForward, Tensor


class Modality(enum.StrEnum):
    """Supported modalities for encoder tasks."""

    IMAGE = "image"
    VIDEO = "video"


class EncoderInput(TaskInput):
    """Input model for encoder tasks.

    Attributes:
        data_urls: The URLs of the data to encode.
    """

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
        model_id: The ID of the model to use for the task.
        modality: Modality of data this encoder can embed.
    """

    model_id: str
    modality: Modality

    def make_record_output(self, task_input: EncoderInput) -> EncoderOutput:
        """Create a task output for task invocation recording."""
        return EncoderOutput(embeddings=[DataForward[Tensor]() for _ in task_input.data_urls])

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        return f"encoder-{self.modality.lower()}-{self.model_id.split('/')[-1].lower()}"
