"""Built-in task for Qwen Omni Thinker and Talker."""

from __future__ import annotations

from cornserve.task.base import Stream, TaskOutput, TaskProfileConfig, UnitTask
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
        max_num_seqs: Maximum batch size for the talker-vocoder serving system.
    """

    model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    max_num_seqs: int = 256

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        return f"llm-{self.model_id.split('/')[-1].lower().replace('.', '-')}-talker"

    def make_record_output(
        self, task_input: OmniTalkerVocoderInput
    ) -> Stream[OpenAIChatCompletionChunk]:
        """Create a task output for task invocation recording."""
        return Stream[OpenAIChatCompletionChunk]()


class OmniTalkerEmbeddingResponse(TaskOutput):
    """Output model for Talker embedding tasks."""

    embeddings: DataForward[Tensor]


class OmniTalkerEmbeddingTask(
    UnitTask[OmniTalkerVocoderInput, OmniTalkerEmbeddingResponse]
):
    """A task that represents the Qwen Omni Talker embedding operation.
    Attributes:
        model_id: The ID of the model to use for the task.
        max_num_seqs: Maximum batch size for the talker serving system.
    """

    model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    max_num_seqs: int = 256

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        return f"llm-{self.model_id.split('/')[-1].lower().replace('.', '-')}-talker"

    def make_record_output(
        self, task_input: OmniTalkerVocoderInput
    ) -> OmniTalkerEmbeddingResponse:
        """Create a mock task output for task invocation recording."""
        return OmniTalkerEmbeddingResponse(embeddings=DataForward[Tensor]())


class DummyOmniTalkerInput(OpenAIChatCompletionRequest):
    """Input for dummy Talker task.

    Attributes:
        thinker_hidden_states_len: Length of the Thinker hidden states tensor that
            Talker should simulate receiving.
        thinker_output_len: Number of Thinker output token IDs to simulate.
            Must be >= 1 and < thinker_hidden_states_len since Thinker hidden states
            include contributions from both the Thinker prompt and the Thinker
            output tokens.

    """

    thinker_hidden_states_len: int
    thinker_output_len: int


class TalkerProfileConfig(TaskProfileConfig):
    """Talker-specific profiling configuration fields.

    TP is not supported for the Talker (single-GPU only).
    """

    max_num_seqs: int = 256
    gpu_memory_utilization: float = 0.9

    def to_profile_str(self) -> str:
        """Return a string representation of the profile configuration."""
        return f"bs{self.max_num_seqs}+gpu{self.gpu_memory_utilization}"


class DummyOmniTalkerEmbeddingResponse(TaskOutput):
    """Output for dummy Talker task; discards all output."""

    pass


class DummyOmniTalkerEmbeddingTask(
    UnitTask[DummyOmniTalkerInput, DummyOmniTalkerEmbeddingResponse],
    TalkerProfileConfig,
):
    """Dummy task that runs the Talker in isolation with random hidden states."""

    model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        return f"dummy-llm-{self.model_id.split('/')[-1].lower().replace('.', '-')}-talker-{self.to_profile_str()}"

    def make_record_output(
        self, task_input: DummyOmniTalkerInput
    ) -> DummyOmniTalkerEmbeddingResponse:
        """Create a mock task output for task invocation recording."""
        return DummyOmniTalkerEmbeddingResponse()


class DummyOmniTalkerVocoderTask(
    UnitTask[DummyOmniTalkerInput, Stream[OpenAIChatCompletionChunk]],
    TalkerProfileConfig,
):
    """Dummy task that runs the Talker & Vocoder with random hidden states."""

    model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        return f"dummy-llm-{self.model_id.split('/')[-1].lower().replace('.', '-')}-talker-vocoder-{self.to_profile_str()}"

    def make_record_output(
        self, task_input: DummyOmniTalkerInput
    ) -> Stream[OpenAIChatCompletionChunk]:
        """Create a mock task output for task invocation recording."""
        return Stream[OpenAIChatCompletionChunk]()
