"""Built-in task for multimodal content generators."""

from __future__ import annotations

import enum

from pydantic import field_validator

from cornserve.task.base import TaskInput, TaskOutput, UnitTask
from cornserve.task.forward import DataForward, Tensor


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
        embeddings: Text embeddings from the LLM encoder (received via sidecar).
    """

    height: int = 1328
    width: int = 1328
    num_inference_steps: int = 50
    embeddings: list[DataForward[Tensor]]


class GeneratorOutput(TaskOutput):
    """Output model for generator tasks.

    Attributes:
        content_url: URL to the generated content.
    """

    content_url: str


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

    @field_validator("model_id")
    @classmethod
    def _validate_model_id(cls, model_id: str) -> str:
        """Ensure model ID is provided."""
        if not model_id:
            raise ValueError("Model ID must be provided.")
        return model_id

    def make_record_output(self, task_input: GeneratorInput) -> GeneratorOutput:
        """Create a task output for task invocation recording."""
        return GeneratorOutput(content_url="")

    def validate_input(self, task_input: GeneratorInput) -> None:
        """Validate the input for the generator task."""
        if task_input.height <= 0 or task_input.width <= 0:
            raise ValueError("Height and width must be positive integers.")

        if task_input.num_inference_steps <= 0:
            raise ValueError("Number of inference steps must be positive.")

        if not task_input.embeddings:
            raise ValueError("Embeddings must be provided (received from LLM via sidecar).")

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        # Use the model name from the model_id (e.g., "Qwen/Qwen-Image" -> "qwen-image")
        model_name = self.model_id.split("/")[-1].lower().replace("-", "_")
        return f"generator-{self.modality.lower()}-{model_name}"
