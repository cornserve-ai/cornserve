"""Build-in task for LLMs."""

from __future__ import annotations

from typing import override

from cornserve.task.base import TaskInput, TaskOutput, UnitTask
from cornserve.task.forward import DataForward, Tensor


class LLMInput(TaskInput):
    """Input model for LLM tasks.

    Attributes:
        prompt: The prompt to send to the LLM.
        embeddings: Multimodal embeddings to send to the LLM.
    """

    prompt: str
    embeddings: list[DataForward[Tensor]] = []


class LLMOutput(TaskOutput):
    """Output model for LLM tasks.

    Attributes:
        response: The response from the LLM.
    """

    response: str


class LLMTask(UnitTask[LLMInput, LLMOutput]):
    """A task that invokes an LLM.

    Attributes:
        model_id: The ID of the model to use for the task.
    """

    model_id: str

    @override
    def make_record_output(self, task_input: LLMInput) -> LLMOutput:
        """Create a task output for task invocation recording."""
        return LLMOutput(response="")
