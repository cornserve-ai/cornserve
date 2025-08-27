"""Built-in task for Qwen Omni Thinker and Talker."""

from __future__ import annotations

from typing import Any, Literal

from cornserve.task.base import Stream, Task, TaskOutput, UnitTask
from cornserve.task.builtins.encoder import Modality
from cornserve.task.builtins.llm import (
    MLLMEmbeddingTask,
    MLLMTask,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequest,
)
from cornserve.task.forward import DataForward, Tensor


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


class OmniInput(OpenAIChatCompletionRequest):
    """Input model for Qwen Omni tasks.

    Attributes:
        return_audio: Whether to return audio response.
    """

    return_audio: bool = True  # same as huggingface parameter
    # use_audio_in_video: not supported yet

    def model_post_init(self, context: Any, /) -> None:
        """Validate the model."""
        assert self.model == "Qwen/Qwen2.5-Omni-7B", "Only Qwen/Qwen2.5-Omni-7B is supported."


def text_stream_transformer(chunk: OpenAIChatCompletionChunk) -> OmniOutputChunk:
    """Transform a text chunk to an OmniOutputChunk."""
    return OmniOutputChunk(text_chunk=chunk)


class OmniTask(Task[OmniInput, Stream[OmniOutputChunk]]):
    """A task that invokes a Multimodal LLM.

    Attributes:
        model_id: The ID of the model to use for the task.
        modalities: List of input modalities other than text.
    """

    model_id: Literal["Qwen/Qwen2.5-Omni-7B"] = "Qwen/Qwen2.5-Omni-7B"
    modalities: list[Modality] = []
    encoder_fission: bool = True
    coalesce_encoder_invocations: bool = True

    def post_init(self) -> None:
        """Initialize subtasks."""
        self.thinker_text = MLLMTask(
            model_id=self.model_id,
            encoder_fission=self.encoder_fission,
            modalities=self.modalities,
            coalesce_encoder_invocations=self.coalesce_encoder_invocations,
        )
        self.thinker_embedding = MLLMEmbeddingTask(
            model_id=self.model_id,
            encoder_fission=self.encoder_fission,
            modalities=self.modalities,
            coalesce_encoder_invocations=self.coalesce_encoder_invocations,
        )
        self.talker_vocoder = OmniTalkerVocoderTask(model_id=self.model_id)
        self.system_prompt = (
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
            "perceiving auditory and visual inputs, as well as generating text and speech."
        )

    def invoke(self, task_input: OmniInput) -> Stream[OmniOutputChunk]:
        """Invoke the task.

        Given multimodal data and a text prompt, run the corresponding encoder
        for multimodal data and then pass the embeddings and text prompt to the LLM.
        """
        # add system prompt
        thinker_input = OpenAIChatCompletionRequest.model_validate(
            dict(
                model=task_input.model,
                messages=[{"role": "system", "content": self.system_prompt}] + task_input.messages,
            )
        )

        if not task_input.return_audio:
            text_stream = self.thinker_text.invoke(thinker_input)
            return text_stream.transform(text_stream_transformer)

        thinker_embedding_output = self.thinker_embedding.invoke(thinker_input)

        talker_input = OmniTalkerVocoderInput.model_validate(
            dict(
                **task_input.model_dump(exclude={"return_audio"}),
                embeddings=thinker_embedding_output.embeddings,
            )
        )
        return self.talker_vocoder.invoke(talker_input)
