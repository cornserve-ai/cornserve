"""OpenAI-compatible LLM unit tasks.

All tasks are OpenAI compatible and output is always streamed.
"""

from __future__ import annotations

import uuid
from typing import Generic, Literal, TypeAlias, TypeVar

from cornserve.task.base import (
    Stream,
    TaskInput,
    TaskOutput,
    TaskProfileConfig,
    UnitTask,
)
from cornserve.task.forward import DataForward, Tensor
from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel, Field


class StreamOptions(BaseModel):
    """Streaming options for OpenAI Chat Completion.

    Attributes:
        include_usage: If set, the final chunk will include token usage statistics.
        continuous_usage_stats: If set, usage statistics are included on every
            streaming chunk (not just the final one).
    """

    include_usage: bool = True
    continuous_usage_stats: bool = False


class ChatCompletionContentPartTextParam(BaseModel):
    """Content part parameter for text content."""

    type: Literal["text"] = "text"
    text: str


class URL(BaseModel):
    """A URL."""

    url: str


class ChatCompletionContentPartAudioParam(BaseModel):
    """Content part parameter for audio content."""

    type: Literal["audio_url"] = "audio_url"
    audio_url: URL


class ChatCompletionContentPartImageParam(BaseModel):
    """Content part parameter for image content."""

    type: Literal["image_url"] = "image_url"
    image_url: URL


class ChatCompletionContentPartVideoParam(BaseModel):
    """Content part parameter for video content."""

    type: Literal["video_url"] = "video_url"
    video_url: URL


# Also supports ommitting the `type` field.
ChatCompletionContentPartMultimodalParam: TypeAlias = (
    ChatCompletionContentPartAudioParam
    | ChatCompletionContentPartImageParam
    | ChatCompletionContentPartVideoParam
)

ChatCompletionContentPartParam: TypeAlias = (
    ChatCompletionContentPartTextParam | ChatCompletionContentPartMultimodalParam
)


class ChatCompletionMessageParam(BaseModel):
    """Message parameter for OpenAI Chat Completion."""

    role: str
    content: str | list[ChatCompletionContentPartParam]


class OpenAIChatCompletionRequest(TaskInput):
    """Input model for OpenAI Chat Completion tasks."""

    messages: list[ChatCompletionMessageParam]
    model: str
    frequency_penalty: float | None = 0.0
    max_completion_tokens: int | None = None
    presence_penalty: float | None = 0.0
    seed: int | None = None
    stream_options: StreamOptions | None = None
    temperature: float | None = None
    top_p: float | None = None
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    ignore_eos: bool = False

    # Cornserve-specific fields
    cornserve_embeddings: list[DataForward[Tensor]] = []
    cornserve_kv_transfer_params: DataForward[dict] | None = None
    encoder_fission: bool = True  # not currently used


def extract_multimodal_content(
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionContentPartMultimodalParam]:
    """Extract multimodal contents from chat messages.

    Args:
        messages: List of chat messages.

    Returns:
        A list of tuples containing the modality and data URL.
    """
    multimodal_data: list[ChatCompletionContentPartMultimodalParam] = []
    for message in messages:
        for part in message.content:
            if isinstance(part, (ChatCompletionContentPartTextParam, str)):
                continue
            multimodal_data.append(part)

    return multimodal_data


InputT = TypeVar("InputT", bound=TaskInput)
OutputT = TypeVar("OutputT", bound=TaskOutput)


def llm_executor_name(
    prefix: str, model_id: str, receive_embeddings: bool, profile_str: str
) -> str:
    """Build the canonical executor name for an LLM task executor.

    Shared by :class:`TaskExecutionDescriptor` subclasses and the benchmark
    Prometheus helpers so that naming stays consistent.
    """
    re_suffix = "re1" if receive_embeddings else "re0"
    return "-".join([prefix, model_id.split("/")[-1], re_suffix, profile_str]).lower()


class LLMProfileConfig(TaskProfileConfig):
    """LLM-specific profiling configuration fields."""

    tp_size: int
    max_num_seqs: int
    gpu_memory_utilization: float

    def to_profile_str(self) -> str:
        """Return a string representation of the profile configuration."""
        return (
            f"tp{self.tp_size}+bs{self.max_num_seqs}+gpu{self.gpu_memory_utilization}"
        )


class LLMBaseUnitTask(
    UnitTask[InputT, OutputT], LLMProfileConfig, Generic[InputT, OutputT]
):
    """Base class for LLM unit tasks.

    Attributes:
        model_id: The ID of the model to use for the task.
        receive_embeddings: Whether to receive multimodal embeddings from
            a separate encoder task. If False, the task will compute them itself.
    """

    model_id: str
    receive_embeddings: bool = True
    enable_prefix_caching: bool = False

    def _re_suffix(self) -> str:
        return "re1" if self.receive_embeddings else "re0"

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        pc_suffix = "+pc1" if self.enable_prefix_caching else ""
        return f"llm-{self.model_id.split('/')[-1].lower()}-{self._re_suffix()}-{self.to_profile_str()}{pc_suffix}"


class OpenAIChatCompletionChunk(TaskOutput, ChatCompletionChunk):
    """Output model for streamed OpenAI Chat Completion tasks."""


class LLMUnitTask(
    LLMBaseUnitTask[OpenAIChatCompletionRequest, Stream[OpenAIChatCompletionChunk]]
):
    """A task that invokes an LLM and returns a stream of chat completion chunks.

    Attributes:
        model_id: The ID of the model to use for the task.
        receive_embeddings: Whether to receive multimodal embeddings from
            a separate encoder task. If False, the task will compute them itself.
        enable_prefix_caching: Whether to enable vLLM prefix caching at startup.
            Used for decode profiling to simulate a pre-populated KV cache.
    """

    def make_record_output(
        self,
        task_input: OpenAIChatCompletionRequest,
    ) -> Stream[OpenAIChatCompletionChunk]:
        """Create a mock task output object for invocation recording."""
        return Stream[OpenAIChatCompletionChunk]()


class DummyMLLMUnitTask(
    UnitTask[OpenAIChatCompletionRequest, Stream[OpenAIChatCompletionChunk]],
    LLMProfileConfig,
):
    """A task that invokes an MLLM and returns a stream of chat completion chunks.

    Attributes:
        model_id: The ID of the model to use for the task.
        receive_embeddings: Use random embeddings when set to True
    """

    model_id: str
    receive_embeddings: bool = True

    def make_record_output(
        self,
        task_input: OpenAIChatCompletionRequest,
    ) -> Stream[OpenAIChatCompletionChunk]:
        """Create a mock task output object for invocation recording."""
        return Stream[OpenAIChatCompletionChunk]()

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        re = "re1" if self.receive_embeddings else "re0"
        return f"dummy-mllm-{self.model_id.split('/')[-1].lower()}-{re}-{self.to_profile_str()}"


class OmniMLLMUnitTask(
    UnitTask[OpenAIChatCompletionRequest, Stream[OpenAIChatCompletionChunk]],
    LLMProfileConfig,
):
    """MLLM unit task with selective per-modality encoder disabling.

    When disable_*_enc is True for a modality, the descriptor rewrites
    that modality's data URLs so gpu_model_runner skips the real encoder
    and uses torch.randn via CORNSERVE_PROFILING instead.
    """

    model_id: str
    disable_audio_enc: bool = False
    disable_image_enc: bool = False
    disable_video_enc: bool = False

    def make_record_output(
        self,
        task_input: OpenAIChatCompletionRequest,
    ) -> Stream[OpenAIChatCompletionChunk]:
        """Create a mock task output object for invocation recording."""
        return Stream[OpenAIChatCompletionChunk]()

    def _enc_flags_str(self) -> str:
        """Short string encoding which encoders are disabled."""
        parts = [
            f"i{'0' if self.disable_image_enc else '1'}",
            f"a{'0' if self.disable_audio_enc else '1'}",
            f"v{'0' if self.disable_video_enc else '1'}",
        ]
        return "+".join(parts)

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        return f"omni-mllm-{self.model_id.split('/')[-1].lower()}-{self._enc_flags_str()}-{self.to_profile_str()}"


class LLMEmbeddingResponse(TaskOutput):
    """Output model for LLM embedding tasks."""

    embeddings: DataForward[Tensor]


class LLMEmbeddingUnitTask(
    LLMBaseUnitTask[OpenAIChatCompletionRequest, LLMEmbeddingResponse]
):
    """A task that invokes an LLM to compute multimodal embeddings.

    Attributes:
        model_id: The ID of the model to use for the task.
        receive_embeddings: Whether to receive multimodal embeddings from
            a separate encoder task. If False, the task will compute them itself.
    """

    def make_record_output(
        self, task_input: OpenAIChatCompletionRequest
    ) -> LLMEmbeddingResponse:
        """Create a mock task output object for invocation recording."""
        return LLMEmbeddingResponse(embeddings=DataForward[Tensor]())


class PrefillChatCompletionResponse(TaskOutput):
    """Output model for Prefill vLLM Chat Completion tasks."""

    kv_transfer_params: DataForward[dict] | None = None
    hidden_states: DataForward[Tensor] | None = None


class PrefillLLMUnitTask(
    UnitTask[OpenAIChatCompletionRequest, PrefillChatCompletionResponse],
    LLMProfileConfig,
):
    """A task that invokes a vLLM to perform prefill."""

    model_id: str
    receive_embeddings: bool = True
    send_hidden_states: bool = False

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        re = "re1" if self.receive_embeddings else "re0"
        return f"prefill-{self.model_id.split('/')[-1].lower()}-{re}-{self.to_profile_str()}"

    def make_record_output(
        self, task_input: OpenAIChatCompletionRequest
    ) -> PrefillChatCompletionResponse:
        """Create a mock task output object for invocation recording."""
        if self.send_hidden_states:
            return PrefillChatCompletionResponse(hidden_states=DataForward[Tensor]())
        return PrefillChatCompletionResponse(kv_transfer_params=DataForward[dict]())

    def validate_input(self, task_input: OpenAIChatCompletionRequest) -> None:
        """Validate the task input."""
        if task_input.model != self.model_id:
            raise ValueError(
                f"Model ID in task input ({task_input.model}) does not match the task model ID ({self.model_id})."
            )


class DecodeLLMUnitTask(
    UnitTask[OpenAIChatCompletionRequest, Stream[OpenAIChatCompletionChunk]],
    LLMProfileConfig,
):
    """A task that invokes a vLLM decoder and returns a stream of chat completion chunks."""

    model_id: str
    receive_embeddings: bool = True

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        re = "re1" if self.receive_embeddings else "re0"
        return f"decode-{self.model_id.split('/')[-1].lower()}-{re}-{self.to_profile_str()}"

    def make_record_output(
        self, task_input: OpenAIChatCompletionRequest
    ) -> Stream[OpenAIChatCompletionChunk]:
        """Create a mock task output object for invocation recording."""
        return Stream[OpenAIChatCompletionChunk]()

    def validate_input(self, task_input: OpenAIChatCompletionRequest) -> None:
        """Validate the task input."""
        if task_input.model != self.model_id:
            raise ValueError(
                f"Model ID in task input ({task_input.model}) does not match the task model ID ({self.model_id})."
            )
        if not task_input.cornserve_kv_transfer_params:
            raise ValueError(
                "KV transfer parameters must be specified in the task input."
            )
