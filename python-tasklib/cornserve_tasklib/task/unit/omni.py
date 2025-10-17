"""Built-in task for Qwen Omni Thinker and Talker."""

from __future__ import annotations

from typing import Literal

from cornserve.task.base import Stream, TaskOutput, UnitTask
from cornserve.task.forward import DataForward, Tensor
from cornserve_tasklib.task.unit.llm import OpenAIChatCompletionChunk, OpenAIChatCompletionRequest


class OmniTalkerVocoderInput(OpenAIChatCompletionRequest):
    """Input model for Qwen Omni Talker.

    Attributes:
        multimodal_data: List of tuples (modality, data URL).
            "image", "audio", "video", etc. for modality.
        embeddings: Thinker embeddings to send to the Talker.
    """

    embeddings: DataForward[Tensor]


class OmniOutputChunk(TaskOutput):
    """Output chunk for Omni tasks.

    Either should be present but not both.

    Attributes:
        audio_chunk: Base64-encoded audio chunk of np.float32 raw waveform.
        text_chunk: Text chunk from the LLM.
    """

    audio_chunk: str | None = None
    text_chunk: OpenAIChatCompletionChunk | None = None


class OmniTalkerVocoderTask(UnitTask[OmniTalkerVocoderInput, Stream[OmniOutputChunk]]):
    """A task that represents the Qwen Omni Talker.

    Attributes:
        model_id: The ID of the model to use for the task.
    """

    model_id: Literal["Qwen/Qwen2.5-Omni-7B"] = "Qwen/Qwen2.5-Omni-7B"

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        return f"llm-{self.model_id.split('/')[-1].lower().replace('.', '-')}-talker"

    def make_record_output(self, task_input: OmniTalkerVocoderInput) -> Stream[OmniOutputChunk]:
        """Create a task output for task invocation recording."""
        return Stream[OmniOutputChunk]()






