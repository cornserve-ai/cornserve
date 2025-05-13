"""Build-in task for LLMs."""

from __future__ import annotations

from typing import Generic, TypeVar

from cornserve.task.base import TaskInput, TaskOutput, UnitTask
from cornserve.task.forward import DataForward, Tensor


class LLMInput(TaskInput):
    """Input model for LLM tasks.

    Attributes:
        prompt: The prompt to send to the LLM.
        multimodal_data: List of tuples (modality, data URL).
        embeddings: Multimodal embeddings to send to the LLM.
    """

    prompt: str
    multimodal_data: list[tuple[str, str]] = []
    embeddings: list[DataForward[Tensor]] = []


class LLMOutputBase(TaskOutput):
    """Base output model for LLM tasks."""


InputT = TypeVar("InputT", bound=TaskInput)
OutputT = TypeVar("OutputT", bound=TaskOutput)


class LLMBaseTask(UnitTask[InputT, OutputT], Generic[InputT, OutputT]):
    """A task that invokes an LLM.

    Attributes:
        model_id: The ID of the model to use for the task.
    """

    model_id: str

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        return f"llm-{self.model_id.split('/')[-1].lower()}"


class LLMOutput(LLMOutputBase):
    """Output model for LLM tasks.

    Attributes:
        response: The response from the LLM.
    """

    response: str


class LLMTask(LLMBaseTask[LLMInput, LLMOutput]):
    """A task that invokes an LLM and returns the response.

    Attributes:
        model_id: The ID of the model to use for the task.
    """

    model_id: str

    def make_record_output(self, task_input: LLMInput) -> LLMOutput:
        """Create a task output for task invocation recording."""
        return LLMOutput(response="")


class LLMForwardOutput(LLMOutputBase):
    """Output model for LLM tasks with the response forwarded.

    Attributes:
        response: The response from the LLM.
    """

    response: DataForward[str]


class LLMForwardOutputTask(LLMBaseTask[LLMInput, LLMForwardOutput]):
    """A task that invokes an LLM and forwards the response.

    Attributes:
        model_id: The ID of the model to use for the task.
    """

    model_id: str

    def make_record_output(self, task_input: LLMInput) -> LLMForwardOutput:
        """Create a task output for task invocation recording."""
        return LLMForwardOutput(response=DataForward[str]())
