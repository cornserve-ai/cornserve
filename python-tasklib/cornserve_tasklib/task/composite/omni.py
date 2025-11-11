"""Built-in task for Qwen Omni Thinker and Talker."""

from __future__ import annotations
from typing import Any, Literal

from cornserve.task.base import Stream, Task

from cornserve_tasklib.task.composite.llm import MLLMEmbeddingTask, MLLMTask
from cornserve_tasklib.task.unit.encoder import Modality
from cornserve_tasklib.task.unit.llm import (
    ChatCompletionMessageParam,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequest,
)
from cornserve_tasklib.task.unit.omni import (
    OmniTalkerVocoderInput,
    OmniTalkerVocoderTask,
)


class OmniInput(OpenAIChatCompletionRequest):
    """Input model for Qwen Omni tasks.

    Attributes:
        return_audio: Whether to return audio response.
    """

    return_audio: bool = True  # same as huggingface parameter
    # use_audio_in_video: not supported yet

    def model_post_init(self, context: Any, /) -> None:
        """Validate the model."""
        assert self.model in {
            "Qwen/Qwen3-Omni-30B-A3B-Instruct"
        }, f"Only Qwen/Qwen3-Omni-30B-A3B-Instruct is supported, got {self.model}"


class OmniTask(Task[OmniInput, Stream[OpenAIChatCompletionChunk]]):
    """A task that invokes a Multimodal LLM.

    Attributes:
        model_id: The ID of the model to use for the task.
        modalities: List of input modalities other than text.
    """

    model_id: Literal["Qwen/Qwen3-Omni-30B-A3B-Instruct"] = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
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

    def invoke(self, task_input: OmniInput) -> Stream[OpenAIChatCompletionChunk]:
        """Invoke the task.

        Given multimodal data and a text prompt, run the corresponding encoder
        for multimodal data and then pass the embeddings and text prompt to the LLM.
        """
        thinker_input = OpenAIChatCompletionRequest.model_validate(
            dict(
                **task_input.model_dump(exclude={"return_audio"}),
            )
        )
        # if only text response is needed, skip talker vocoder
        if not task_input.return_audio:
            return self.thinker_text.invoke(thinker_input)

        thinker_embedding_output = self.thinker_embedding.invoke(thinker_input)

        talker_input = OmniTalkerVocoderInput.model_validate(
            dict(
                **task_input.model_dump(exclude={"return_audio"}),
                thinker_hidden_states=thinker_embedding_output.embeddings,
            )
        )
        talker_input.messages.append(
            ChatCompletionMessageParam.model_validate(
                {"role": "assistant", "content": "<tts_pad>"}
            )
        )

        return self.talker_vocoder.invoke(talker_input)
