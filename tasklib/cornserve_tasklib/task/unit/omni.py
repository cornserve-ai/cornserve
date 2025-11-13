"""Built-in task for Qwen Omni Thinker and Talker."""

from __future__ import annotations

from typing import Literal

from cornserve.task.base import Stream, UnitTask
from cornserve.task.forward import DataForward, Tensor

from cornserve_tasklib.task.unit.llm import (
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequest,
)


class OmniTalkerVocoderInput(OpenAIChatCompletionRequest):
    """Input model for Qwen Omni Talker.

    Attributes:
        thinker_hidden_states: Thinker's hidden_states to send to the Talker.
    """

    thinker_hidden_states: DataForward[Tensor]


class OmniTalkerVocoderTask(
    UnitTask[OmniTalkerVocoderInput, Stream[OpenAIChatCompletionChunk]]
):
    """A task that represents the Qwen Omni Talker Vocoder.

    Attributes:
        model_id: The ID of the model to use for the task.
    """

    model_id: Literal["Qwen/Qwen3-Omni-30B-A3B-Instruct"] = (
        "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    )

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        return f"llm-{self.model_id.split('/')[-1].lower().replace('.', '-')}-talker"

    def make_record_output(
        self, task_input: OmniTalkerVocoderInput
    ) -> Stream[OpenAIChatCompletionChunk]:
        """Create a task output for task invocation recording."""
        return Stream[OpenAIChatCompletionChunk]()
