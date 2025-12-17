"""Built-in task for multimodal content generation."""

from __future__ import annotations

import enum

from cornserve.task.base import Stream, TaskInput, TaskOutput, UnitTask
from cornserve.task.forward import DataForward, Tensor

from cornserve_tasklib.task.unit.llm import OpenAIChatCompletionChunk


class Modality(enum.StrEnum):
    """Supported modalities for generator tasks."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class GeneratorInput(TaskInput):
    """Input model for generator tasks.

    Attributes:
        height: Height of the generated content in pixels.
        width: Width of the generated content in pixels.
        num_inference_steps: Number of denoising steps to perform.
        embeddings: Text (e.g., prompt) embeddings from the encoder.
        skip_tokens: Number of initial tokens to skip from the embeddings.
    """

    height: int
    width: int
    num_inference_steps: int
    embeddings: DataForward[Tensor]
    skip_tokens: int = 0


class GeneratorOutput(TaskOutput):
    """Output model for generator tasks.

    Attributes:
        generated: Generated content as bytes, encoded in base64. PNG for images.
    """

    generated: str


class GeneratorTask(UnitTask[GeneratorInput, GeneratorOutput]):
    """A task that invokes a multimodal content generator.

    Attributes:
        modality: Modality of content this generator can create.
        model_id: The ID of the model to use for the task.
        max_batch_size: Maximum batch size to use for the serving system.
    """

    modality: Modality
    model_id: str
    max_batch_size: int = 1

    def make_record_output(self, task_input: GeneratorInput) -> GeneratorOutput:
        """Create a task output for task invocation recording."""
        return GeneratorOutput(generated="")

    def validate_input(self, task_input: GeneratorInput) -> None:
        """Validate the input for the generator task."""
        if task_input.height <= 0 or task_input.width <= 0:
            raise ValueError("Height and width must be positive integers.")

        if task_input.num_inference_steps <= 0:
            raise ValueError("Number of inference steps must be positive.")

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        model_name = self.model_id.split("/")[-1].lower()
        return f"generator-{self.modality.lower()}-{model_name}"


class AudioGeneratorInput(TaskInput):
    """Input model for generator tasks."""

    embeddings: DataForward[Tensor]
    chunk_size: int | None = None
    left_context_size: int | None = None


class AudioGeneratorTask(
    UnitTask[AudioGeneratorInput, Stream[OpenAIChatCompletionChunk]]
):
    """A task that invokes a multimodal content generator.
    Attributes:
        modality: Modality of content this generator can create.
        model_id: The ID of the model to use for the task.
        max_batch_size: Maximum batch size to use for the serving system.
    """

    modality: Modality
    model_id: str
    max_batch_size: int = 1

    def make_record_output(
        self, task_input: AudioGeneratorInput
    ) -> Stream[OpenAIChatCompletionChunk]:
        """Create a task output for task invocation recording."""
        return Stream[OpenAIChatCompletionChunk]()

    def validate_input(self, task_input: AudioGeneratorInput) -> None:
        """Validate the input for the generator task."""
        pass

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        model_name = self.model_id.split("/")[-1].lower()
        return f"generator-{self.modality.lower()}-{model_name}"
