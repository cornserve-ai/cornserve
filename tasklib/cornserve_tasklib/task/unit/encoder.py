"""Build-in task for modality encoders."""

from __future__ import annotations

import enum

from cornserve.task.base import TaskInput, TaskOutput, UnitTask
from cornserve.task.forward import DataForward, Tensor
from pydantic import field_validator


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
        model_ids: The IDs of models to use for the task. When more than one is provided,
            the first model is used to load the base model weights, and the rest are used
            to load in extra adapters (e.g., Gemma 3 multimodal projectors).
        max_batch_size: Maximum batch size to use for the serving system.
    """

    modality: Modality
    model_ids: set[str]
    max_batch_size: int = 1

    @field_validator("model_ids")
    @classmethod
    def _validate_model_ids(cls, model_ids: set[str]) -> set[str]:
        """Ensure at least one model ID is provided."""
        if not model_ids:
            raise ValueError("At least one model ID must be provided.")
        return model_ids

    def make_record_output(self, task_input: EncoderInput) -> EncoderOutput:
        """Create a task output for task invocation recording."""
        return EncoderOutput(
            embeddings=[DataForward[Tensor]() for _ in task_input.data_urls]
        )

    def validate_input(self, task_input: EncoderInput) -> None:
        """Validate the input for the encoder task."""
        if not task_input.data_urls:
            raise ValueError("Data URLs cannot be empty.")

        if task_input.model_id not in self.model_ids:
            raise ValueError(
                f"Model ID {task_input.model_id} does not match any of the supported model IDs: {self.model_ids}."
            )

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        first_model_name = sorted(self.model_ids)[0].split("/")[-1].lower()
        return f"encoder-{self.modality.lower()}-{first_model_name}"


class DummyEncoderOutput(TaskOutput):
    """Dummpy Output model for encoder tasks."""


class DummyEncoderTask(UnitTask[EncoderInput, DummyEncoderOutput]):
    """A dummy task that invokes an encoder without dataforward.

    Attributes:
        modality: Modality of data this encoder can embed.
        model_ids: The IDs of models to use for the task. When more than one is provided,
            the first model is used to load the base model weights, and the rest are used
            to load in extra adapters (e.g., Gemma 3 multimodal projectors).
        max_batch_size: Maximum batch size to use for the serving system.
    """

    modality: Modality
    model_ids: set[str]
    max_batch_size: int = 1

    @field_validator("model_ids")
    @classmethod
    def _validate_model_ids(cls, model_ids: set[str]) -> set[str]:
        """Ensure at least one model ID is provided."""
        if not model_ids:
            raise ValueError("At least one model ID must be provided.")
        return model_ids

    def make_record_output(self, task_input: EncoderInput) -> DummyEncoderOutput:
        """Create a task output for task invocation recording."""
        return DummyEncoderOutput()

    def validate_input(self, task_input: EncoderInput) -> None:
        """Validate the input for the encoder task."""
        if not task_input.data_urls:
            raise ValueError("Data URLs cannot be empty.")

        if task_input.model_id not in self.model_ids:
            raise ValueError(
                f"Model ID {task_input.model_id} does not match any of the supported model IDs: {self.model_ids}."
            )

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        first_model_name = sorted(self.model_ids)[0].split("/")[-1].lower()
        return f"encoder-{self.modality.lower()}-{first_model_name}"
