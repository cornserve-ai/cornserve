"""OpenAI-compatible LLM tasks.

All tasks are OpenAI compatible and output is always streamed.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Generic, Literal, TypeAlias, TypeVar

from openai.types.chat import ChatCompletionChunk
from pydantic import BaseModel, Field

from cornserve.task.base import Stream, Task, TaskInput, TaskOutput, UnitTask
from cornserve.task.builtins.encoder import EncoderInput, EncoderOutput, EncoderTask, Modality
from cornserve.task.forward import DataForward, Tensor


class StreamOptions(BaseModel):
    """Streaming options for OpenAI Chat Completion.

    Attributes:
        include_usage: If set, the final chunk will include token usage statistics.
    """

    include_usage: bool = True


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
    ChatCompletionContentPartAudioParam | ChatCompletionContentPartImageParam | ChatCompletionContentPartVideoParam
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
            if isinstance(part, (str, ChatCompletionContentPartTextParam)):
                continue
            multimodal_data.append(part)

    return multimodal_data


InputT = TypeVar("InputT", bound=TaskInput)
OutputT = TypeVar("OutputT", bound=TaskOutput)


class LLMBaseUnitTask(UnitTask[InputT, OutputT], Generic[InputT, OutputT]):
    """Base class for LLM unit tasks.

    Attributes:
        model_id: The ID of the model to use for the task.
        receive_embeddings: Whether to receive multimodal embeddings from
            a separate encoder task. If False, the task will compute them itself.
    """

    model_id: str
    receive_embeddings: bool = True
    # receive_embeddings: bool = Field(
    #     True,
    #     json_schema_extra={"skip_comparison": False},
    #     # setting this will allowing sharing the vLLM instance
    #     # see is_equivalent_to in python/cornserve/task/base.py
    # )

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        return f"llm-{self.model_id.split('/')[-1].lower()}"


class OpenAIChatCompletionChunk(TaskOutput, ChatCompletionChunk):
    """Output model for streamed OpenAI Chat Completion tasks."""


class LLMUnitTask(LLMBaseUnitTask[OpenAIChatCompletionRequest, Stream[OpenAIChatCompletionChunk]]):
    """A task that invokes an LLM and returns a stream of chat completion chunks.

    Attributes:
        model_id: The ID of the model to use for the task.
        receive_embeddings: Whether to receive multimodal embeddings from
            a separate encoder task. If False, the task will compute them itself.
    """

    def make_record_output(
        self,
        task_input: OpenAIChatCompletionRequest,
    ) -> Stream[OpenAIChatCompletionChunk]:
        """Create a mock task output object for invocation recording."""
        return Stream[OpenAIChatCompletionChunk]()


class LLMEmbeddingResponse(TaskOutput):
    """Output model for LLM embedding tasks."""

    embeddings: DataForward[Tensor]


class LLMEmbeddingUnitTask(LLMBaseUnitTask[OpenAIChatCompletionRequest, LLMEmbeddingResponse]):
    """A task that invokes an LLM to compute multimodal embeddings.

    Attributes:
        model_id: The ID of the model to use for the task.
        receive_embeddings: Whether to receive multimodal embeddings from
            a separate encoder task. If False, the task will compute them itself.
    """

    def make_record_output(self, task_input: OpenAIChatCompletionRequest) -> LLMEmbeddingResponse:
        """Create a mock task output object for invocation recording."""
        return LLMEmbeddingResponse(embeddings=DataForward[Tensor]())


class MLLMTask(Task[OpenAIChatCompletionRequest, Stream[OpenAIChatCompletionChunk]]):
    """A task that invokes a Multimodal LLM.

    Attributes:
        model_id: The ID of the model to use for the task.
        modalities: List of input modalities other than text.
        encoder_fission: If True, the task will use separate encoder tasks for computing
            multimodal embeddings. If False, it will use the LLM server to compute them.
        coalesce_encoder_invocations: If True, the task will coalesce encoder invocations
            for the same modality into a single invocation. If False, it will invoke the
            encoder task for each data URL separately.
        encoder_model_ids: Encoders can take multiple model IDs when the architecture
            supports adapters (e.g., Gemma 3 multimodal projectors). Only used when
            `encoder_fission` is True.
    """

    model_id: str
    modalities: list[Modality] = []
    encoder_fission: bool = True
    coalesce_encoder_invocations: bool = False
    encoder_model_ids: set[str] | None = None

    def post_init(self) -> None:
        """Initialize subtasks."""
        if self.encoder_fission:
            self.encoders = {
                modality: EncoderTask(model_ids=self.encoder_model_ids or {self.model_id}, modality=modality)
                for modality in self.modalities
            }
        self.llm = LLMUnitTask(model_id=self.model_id, receive_embeddings=self.encoder_fission)

    def invoke(self, task_input: OpenAIChatCompletionRequest) -> Stream[OpenAIChatCompletionChunk]:
        """Invoke the task."""
        if self.encoder_fission:
            encoder_input_urls: dict[Modality, list[str]] = defaultdict(list)
            multimodal_contents = extract_multimodal_content(task_input.messages)
            for multimodal_content in multimodal_contents:
                modality = Modality(multimodal_content.type.split("_")[0])
                data_url: URL = getattr(multimodal_content, multimodal_content.type)
                encoder_input_urls[modality].append(data_url.url)

            # Check if modalities not specified in the task are present in the input.
            if diff := set(encoder_input_urls.keys()) - set(self.modalities):
                raise ValueError(
                    "The following modalities in the input are not specified in the task: "
                    f"{[mod.value for mod in diff]}",
                )

            # Invoke the encoder tasks
            if self.coalesce_encoder_invocations:
                # Coalesce encoder invocations: invoke once per modality with all URLs
                encoder_outputs: dict[Modality, EncoderOutput] = {}
                for modality, encoder_task in self.encoders.items():
                    if modality not in encoder_input_urls:
                        continue
                    encoder_input = EncoderInput(model_id=task_input.model, data_urls=encoder_input_urls[modality])
                    encoder_output = encoder_task.invoke(encoder_input)
                    encoder_outputs[modality] = encoder_output

                # Retain the order of multimodal data in the task input
                embeddings: list[DataForward[Tensor]] = []
                for multimodal_content in multimodal_contents:
                    modality = Modality(multimodal_content.type.split("_")[0])
                    embeddings.append(encoder_outputs[modality].embeddings.pop(0))
            else:
                # Separate encoder invocations: invoke encoder for each individual URL
                embeddings: list[DataForward[Tensor]] = []
                for multimodal_content in multimodal_contents:
                    modality = Modality(multimodal_content.type.split("_")[0])
                    data_url: URL = getattr(multimodal_content, multimodal_content.type)
                    encoder_input = EncoderInput(model_id=task_input.model, data_urls=[data_url.url])
                    encoder_output = self.encoders[modality].invoke(encoder_input)
                    embeddings.append(encoder_output.embeddings[0])

            # To be consumed by the LLM task.
            task_input.cornserve_embeddings = embeddings

        # Invoke the LLM task.
        return self.llm.invoke(task_input)


class PrefillChatCompletionResponse(TaskOutput):
    """Output model for Prefill vLLM Chat Completion tasks."""

    kv_transfer_params: DataForward[dict] | None = None
    hidden_states: DataForward[Tensor] | None = None


class PrefillLLMUnitTask(UnitTask[OpenAIChatCompletionRequest, PrefillChatCompletionResponse]):
    """A task that invokes a vLLM to perform prefill."""

    model_id: str
    receive_embeddings: bool = True
    send_hidden_states: bool = False

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        return f"prefill-{self.model_id.split('/')[-1].lower()}"

    def make_record_output(self, task_input: OpenAIChatCompletionRequest) -> PrefillChatCompletionResponse:
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


class DecodeLLMUnitTask(UnitTask[OpenAIChatCompletionRequest, Stream[OpenAIChatCompletionChunk]]):
    """A task that invokes a vLLM decoder and returns a stream of chat completion chunks."""

    model_id: str
    receive_embeddings: bool = True

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        return f"decode-{self.model_id.split('/')[-1].lower()}"

    def make_record_output(self, task_input: OpenAIChatCompletionRequest) -> Stream[OpenAIChatCompletionChunk]:
        """Create a mock task output object for invocation recording."""
        return Stream[OpenAIChatCompletionChunk]()

    def validate_input(self, task_input: OpenAIChatCompletionRequest) -> None:
        """Validate the task input."""
        if task_input.model != self.model_id:
            raise ValueError(
                f"Model ID in task input ({task_input.model}) does not match the task model ID ({self.model_id})."
            )
        if not task_input.cornserve_kv_transfer_params:
            raise ValueError("KV transfer parameters must be specified in the task input.")


class DisaggregatedMLLMTask(Task[OpenAIChatCompletionRequest, Stream[OpenAIChatCompletionChunk]]):
    """A task that invokes a Multimodal LLM, with disaggregated prefill and decode in LLM.

    Attributes:
        model_id: The ID of the model to use for the task.
        modalities: List of input modalities other than text.
        encoder_fission: If True, the task will use separate encoder tasks for computing
            multimodal embeddings. If False, it will use the LLM server to compute them.
        encoder_model_ids: Encoders can take multiple model IDs when the architecture
            supports adapters (e.g., Gemma 3 multimodal projectors). Only used when
            `encoder_fission` is True.
    """

    model_id: str
    modalities: list[Modality] = []
    encoder_fission: bool = True
    encoder_model_ids: set[str] | None = None

    def post_init(self) -> None:
        """Initialize subtasks."""
        if self.encoder_fission:
            self.encoders = {
                modality: EncoderTask(model_ids=self.encoder_model_ids or {self.model_id}, modality=modality)
                for modality in self.modalities
            }
        self.prefill = PrefillLLMUnitTask(model_id=self.model_id, receive_embeddings=self.encoder_fission)
        self.decode = DecodeLLMUnitTask(model_id=self.model_id, receive_embeddings=self.encoder_fission)

    def invoke(self, task_input: OpenAIChatCompletionRequest) -> Stream[OpenAIChatCompletionChunk]:
        """Invoke the task."""
        if self.encoder_fission:
            encoder_input_urls: dict[Modality, list[str]] = defaultdict(list)
            multimodal_contents = extract_multimodal_content(task_input.messages)
            for multimodal_content in multimodal_contents:
                modality = Modality(multimodal_content.type.split("_")[0])
                data_url: URL = getattr(multimodal_content, multimodal_content.type)
                encoder_input_urls[modality].append(data_url.url)

            # Check if modalities not specified in the task are present in the input.
            if diff := set(encoder_input_urls.keys()) - set(self.modalities):
                raise ValueError(
                    "The following modalities in the input are not specified in the task: "
                    f"{[mod.value for mod in diff]}",
                )

            # Invoke the encoder tasks for each modality
            encoder_outputs: dict[Modality, EncoderOutput] = {}
            for modality, encoder_task in self.encoders.items():
                if modality not in encoder_input_urls:
                    continue
                encoder_input = EncoderInput(model_id=task_input.model, data_urls=encoder_input_urls[modality])
                encoder_output = encoder_task.invoke(encoder_input)
                encoder_outputs[modality] = encoder_output

            # Retain the order of multimodal data in the task input
            embeddings: list[DataForward[Tensor]] = []
            for multimodal_content in multimodal_contents:
                modality = Modality(multimodal_content.type.split("_")[0])
                embeddings.append(encoder_outputs[modality].embeddings.pop(0))

            # To be consumed by the LLM task.
            task_input.cornserve_embeddings = embeddings

        prefill_output = self.prefill.invoke(task_input)
        # ideally we want to exclude and remove `cornserve_embeddings`
        # but sometimes the decode instance needs the image embeddings
        # due to a potential bug in vLLM
        decode_input = task_input.model_copy(deep=True)
        decode_input.cornserve_kv_transfer_params = prefill_output.kv_transfer_params

        # Invoke the LLM task.
        return self.decode.invoke(decode_input)
