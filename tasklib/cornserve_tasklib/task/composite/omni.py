"""Built-in task for Qwen Omni Thinker and Talker."""

from __future__ import annotations

from typing import Any, Literal

from cornserve.task.base import Stream, Task

from cornserve_tasklib.task.composite.llm import MLLMEmbeddingTask, MLLMTask
from cornserve_tasklib.task.unit.encoder import Modality as EncoderModality
from cornserve_tasklib.task.unit.generator import (
    AudioGeneratorInput,
    AudioGeneratorTask,
)
from cornserve_tasklib.task.unit.generator import (
    Modality as GeneratorModality,
)
from cornserve_tasklib.task.unit.llm import (
    ChatCompletionMessageParam,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequest,
)
from cornserve_tasklib.task.unit.omni import (
    OmniTalkerEmbeddingTask,
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
        assert self.model in {"Qwen/Qwen3-Omni-30B-A3B-Instruct"}, (
            f"Only Qwen/Qwen3-Omni-30B-A3B-Instruct is supported, got {self.model}"
        )


class OmniTask(Task[OmniInput, Stream[OpenAIChatCompletionChunk]]):
    """A task that invokes a Multimodal LLM.

    Attributes:
        model_id: The ID of the model to use for the task.
        modalities: List of input modalities other than text.
    """

    model_id: Literal["Qwen/Qwen3-Omni-30B-A3B-Instruct"] = (
        "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    )
    modalities: list[EncoderModality] = []
    encoder_fission: bool = True
    vocoder_fission: bool = True
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
        if self.vocoder_fission:
            self.talker_vocoder = None
            self.talker_embedding = OmniTalkerEmbeddingTask(model_id=self.model_id)
            self.vocoder_geri = AudioGeneratorTask(
                modality=GeneratorModality.AUDIO, model_id=self.model_id
            )
        else:
            self.talker_vocoder = OmniTalkerVocoderTask(model_id=self.model_id)
            self.talker_embedding = None
            self.vocoder_geri = None

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

        if self.vocoder_fission:
            assert self.talker_embedding is not None
            assert self.vocoder_geri is not None
            talker_embedding_output = self.talker_embedding.invoke(talker_input)
            vocoder_input = AudioGeneratorInput(
                embeddings=talker_embedding_output.embeddings,
            )
            return self.vocoder_geri.invoke(vocoder_input)

        assert self.talker_vocoder is not None
        return self.talker_vocoder.invoke(talker_input)
