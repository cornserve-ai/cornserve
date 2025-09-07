"""OpenAI-compatible composite LLM tasks."""

from __future__ import annotations

from collections import defaultdict

from cornserve.task.base import Stream, Task
from cornserve.task.forward import DataForward, Tensor
from cornserve_tasklib.task.unit.encoder import EncoderInput, EncoderOutput, EncoderTask, Modality
from cornserve_tasklib.task.unit.llm import (
    LLMUnitTask,
    PrefillLLMUnitTask,
    DecodeLLMUnitTask,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionChunk,
    extract_multimodal_content,
    URL,
)


class MLLMTask(Task[OpenAIChatCompletionRequest, Stream[OpenAIChatCompletionChunk]]):
    """A task that invokes a Multimodal LLM.

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

        # Invoke the LLM task.
        return self.llm.invoke(task_input)


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
