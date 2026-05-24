"""OpenAI-compatible composite LLM tasks."""

from __future__ import annotations

from collections import defaultdict
from typing import cast

from cornserve.task.base import Stream, Task
from cornserve.task.forward import DataForward, Tensor
from pydantic import model_validator

from cornserve_tasklib.task.composite.router import RouterApp
from cornserve_tasklib.task.unit.encoder import (
    EncoderInput,
    EncoderOutput,
    EncoderTask,
    Modality,
)
from cornserve_tasklib.task.unit.llm import (
    URL,
    DecodeLLMUnitTask,
    LLMEmbeddingResponse,
    LLMEmbeddingUnitTask,
    LLMUnitTask,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequest,
    PrefillLLMUnitTask,
    extract_multimodal_content,
)


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
    macro_ut_deployment_id: str | None = None

    eric_max_batch_size: int
    llm_tp_size: int
    llm_max_num_seqs: int
    llm_gpu_memory_utilization: float

    def post_init(self) -> None:
        """Initialize subtasks."""
        if self.encoder_fission:
            self.encoders = {
                modality: EncoderTask(
                    model_ids=self.encoder_model_ids or {self.model_id},
                    modality=modality,
                    macro_ut_deployment_id=self.macro_ut_deployment_id,
                    max_batch_size=self.eric_max_batch_size,
                )
                for modality in self.modalities
            }
        self.llm = LLMUnitTask(
            model_id=self.model_id,
            receive_embeddings=self.encoder_fission,
            macro_ut_deployment_id=self.macro_ut_deployment_id,
            tp_size=self.llm_tp_size,
            max_num_seqs=self.llm_max_num_seqs,
            gpu_memory_utilization=self.llm_gpu_memory_utilization,
        )

    def invoke(
        self, task_input: OpenAIChatCompletionRequest
    ) -> Stream[OpenAIChatCompletionChunk]:
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
                    encoder_input = EncoderInput(
                        model_id=task_input.model,
                        data_urls=encoder_input_urls[modality],
                    )
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
                    encoder_input = EncoderInput(
                        model_id=task_input.model, data_urls=[data_url.url]
                    )
                    encoder_output = self.encoders[modality].invoke(encoder_input)
                    embeddings.append(encoder_output.embeddings[0])

            # To be consumed by the LLM task.
            task_input.cornserve_embeddings = embeddings

        # Invoke the LLM task.
        return self.llm.invoke(task_input)


class MLLMEmbeddingTask(Task[OpenAIChatCompletionRequest, LLMEmbeddingResponse]):
    """A task that invokes a Multimodal LLM.

    Note: this task only differs from MLLMTask in that it outputs embeddings instread of
    OpenAIChatCompletionChunk stream, which is intended to be chained to another UnitTask
    which needs the hidden states.
    TODO: update the task abstraction to allow multiple output types.

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
    macro_ut_deployment_id: str | None = None

    eric_max_batch_size: int
    llm_tp_size: int
    llm_max_num_seqs: int
    llm_gpu_memory_utilization: float

    def post_init(self) -> None:
        """Initialize subtasks."""
        if self.encoder_fission:
            self.encoders = {
                modality: EncoderTask(
                    model_ids=self.encoder_model_ids or {self.model_id},
                    modality=modality,
                    macro_ut_deployment_id=self.macro_ut_deployment_id,
                    max_batch_size=self.eric_max_batch_size,
                )
                for modality in self.modalities
            }
        self.llm = LLMEmbeddingUnitTask(
            model_id=self.model_id,
            receive_embeddings=self.encoder_fission,
            macro_ut_deployment_id=self.macro_ut_deployment_id,
            tp_size=self.llm_tp_size,
            max_num_seqs=self.llm_max_num_seqs,
            gpu_memory_utilization=self.llm_gpu_memory_utilization,
        )

    def invoke(self, task_input: OpenAIChatCompletionRequest) -> LLMEmbeddingResponse:
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
                    encoder_input = EncoderInput(
                        model_id=task_input.model,
                        data_urls=encoder_input_urls[modality],
                    )
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
                    encoder_input = EncoderInput(
                        model_id=task_input.model, data_urls=[data_url.url]
                    )
                    encoder_output = self.encoders[modality].invoke(encoder_input)
                    embeddings.append(encoder_output.embeddings[0])

            # To be consumed by the LLM task.
            task_input.cornserve_embeddings = embeddings

        # Invoke the LLM task.
        return self.llm.invoke(task_input)


class DisaggregatedMLLMTask(
    Task[OpenAIChatCompletionRequest, Stream[OpenAIChatCompletionChunk]]
):
    """A task that invokes a Multimodal LLM, with disaggregated prefill and decode in LLM.

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

    eric_max_batch_size: int
    prefill_tp_size: int
    prefill_max_num_seqs: int
    prefill_gpu_memory_utilization: float

    decode_tp_size: int
    decode_max_num_seqs: int
    decode_gpu_memory_utilization: float

    def post_init(self) -> None:
        """Initialize subtasks."""
        if self.encoder_fission:
            self.encoders = {
                modality: EncoderTask(
                    model_ids=self.encoder_model_ids or {self.model_id},
                    modality=modality,
                    max_batch_size=self.eric_max_batch_size,
                )
                for modality in self.modalities
            }
        self.prefill = PrefillLLMUnitTask(
            model_id=self.model_id,
            receive_embeddings=self.encoder_fission,
            tp_size=self.prefill_tp_size,
            max_num_seqs=self.prefill_max_num_seqs,
            gpu_memory_utilization=self.prefill_gpu_memory_utilization,
        )
        self.decode = DecodeLLMUnitTask(
            model_id=self.model_id,
            receive_embeddings=self.encoder_fission,
            tp_size=self.decode_tp_size,
            max_num_seqs=self.decode_max_num_seqs,
            gpu_memory_utilization=self.decode_gpu_memory_utilization,
        )

    def invoke(
        self, task_input: OpenAIChatCompletionRequest
    ) -> Stream[OpenAIChatCompletionChunk]:
        """Invoke the task."""
        # TODO: clean up repeated code with MLLMTask
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
                    encoder_input = EncoderInput(
                        model_id=task_input.model,
                        data_urls=encoder_input_urls[modality],
                    )
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
                    encoder_input = EncoderInput(
                        model_id=task_input.model, data_urls=[data_url.url]
                    )
                    encoder_output = self.encoders[modality].invoke(encoder_input)
                    embeddings.append(encoder_output.embeddings[0])

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


class TimeSharingMLLMTask(
    Task[OpenAIChatCompletionRequest, Stream[OpenAIChatCompletionChunk]]
):
    """A task that time-shares encoder processing between a separate encoder and the LLM.

    Unlike MLLMTask which statically routes all multimodal requests through either the
    separate encoder (encoder_fission=True) or the LLM's internal encoder
    (encoder_fission=False), this task makes a per-request routing decision.

    Both the separate encoders and the LLM are always launched. The LLM is launched with
    receive_embeddings=False so it can always process raw multimodal content internally.
    For each multimodal request, with probability ``encoder_fission_prob`` the request is
    routed through the separate encoder first; otherwise it goes directly to the LLM.

    The routing decision is derived deterministically from ``request_id`` so that it is
    identical across the record and replay phases of task dispatch.

    Attributes:
        model_id: The ID of the model to use for the task.
        modalities: List of input modalities other than text.
        encoder_fission_prob: Probability that a multimodal request is routed through
            the separate encoder. 0.0 means the LLM always encodes internally;
            1.0 means the separate encoder is always used.
        coalesce_encoder_invocations: If True, the task will coalesce encoder invocations
            for the same modality into a single invocation.
        encoder_model_ids: Encoders can take multiple model IDs when the architecture
            supports adapters. Only used for the separate encoder path.
    """

    model_id: str
    modalities: list[Modality] = []
    encoder_fission_prob: float
    coalesce_encoder_invocations: bool = False
    encoder_model_ids: set[str] | None = None
    macro_ut_deployment_id: str | None = None

    eric_max_batch_size: int
    llm_tp_size: int
    llm_max_num_seqs: int
    llm_gpu_memory_utilization: float

    def post_init(self) -> None:
        """Initialize subtasks."""
        # Always launch encoders.
        self.encoders = {
            modality: EncoderTask(
                model_ids=self.encoder_model_ids or {self.model_id},
                modality=modality,
                macro_ut_deployment_id=self.macro_ut_deployment_id,
                max_batch_size=self.eric_max_batch_size,
            )
            for modality in self.modalities
        }
        # LLM is always launched with receive_embeddings=False so it can handle
        # raw multimodal content when the request bypasses the encoder.
        self.llm = LLMUnitTask(
            model_id=self.model_id,
            receive_embeddings=False,
            macro_ut_deployment_id=self.macro_ut_deployment_id,
            tp_size=self.llm_tp_size,
            max_num_seqs=self.llm_max_num_seqs,
            gpu_memory_utilization=self.llm_gpu_memory_utilization,
        )

    def _use_encoder(self, request_id: str) -> bool:
        """Deterministically decide whether to route through the separate encoder.

        Derives the decision from the request_id so the same choice is made
        during both the record and replay phases of task dispatch.
        """
        return int(request_id, 16) % 1000 < int(self.encoder_fission_prob * 1000)

    def invoke(
        self, task_input: OpenAIChatCompletionRequest
    ) -> Stream[OpenAIChatCompletionChunk]:
        """Invoke the task."""
        multimodal_contents = extract_multimodal_content(task_input.messages)
        has_multimodal = len(multimodal_contents) > 0

        if has_multimodal and self._use_encoder(task_input.request_id):
            encoder_input_urls: dict[Modality, list[str]] = defaultdict(list)
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
                encoder_outputs: dict[Modality, EncoderOutput] = {}
                for modality, encoder_task in self.encoders.items():
                    if modality not in encoder_input_urls:
                        continue
                    encoder_input = EncoderInput(
                        model_id=task_input.model,
                        data_urls=encoder_input_urls[modality],
                    )
                    encoder_output = encoder_task.invoke(encoder_input)
                    encoder_outputs[modality] = encoder_output

                embeddings: list[DataForward[Tensor]] = []
                for multimodal_content in multimodal_contents:
                    modality = Modality(multimodal_content.type.split("_")[0])
                    embeddings.append(encoder_outputs[modality].embeddings.pop(0))
            else:
                embeddings: list[DataForward[Tensor]] = []
                for multimodal_content in multimodal_contents:
                    modality = Modality(multimodal_content.type.split("_")[0])
                    data_url: URL = getattr(multimodal_content, multimodal_content.type)
                    encoder_input = EncoderInput(
                        model_id=task_input.model, data_urls=[data_url.url]
                    )
                    encoder_output = self.encoders[modality].invoke(encoder_input)
                    embeddings.append(encoder_output.embeddings[0])

            task_input.cornserve_embeddings = embeddings

        # Invoke the LLM task.
        return self.llm.invoke(task_input)


class MixedMLLMTask(
    Task[OpenAIChatCompletionRequest, Stream[OpenAIChatCompletionChunk]]
):
    """A mixed MLLM app with a shared encoder pool and weighted LLM routing.

    This task always runs a separate encoder path and routes each request to one
    LLMUnitTask from ``llm_routing_tasks`` using ``routing_weights``.

    The encoder is shared across all LLM routes. Each routed LLM task must have
    ``receive_embeddings=True`` so that it consumes the shared embeddings.
    """

    model_id: str
    modalities: list[Modality] = []
    llm_routing_tasks: list[LLMUnitTask]
    routing_weights: list[float]

    coalesce_encoder_invocations: bool = False
    encoder_model_ids: set[str] | None = None
    macro_ut_deployment_id: str | None = None

    eric_max_batch_size: int

    @model_validator(mode="after")
    def _validate_routing_config(self) -> MixedMLLMTask:
        """Validate routing config shape and LLM compatibility."""
        if not self.llm_routing_tasks:
            raise ValueError("llm_routing_tasks must contain at least one task")

        if len(self.llm_routing_tasks) != len(self.routing_weights):
            raise ValueError(
                "routing_weights must have the same length as llm_routing_tasks"
            )

        if any(weight < 0 for weight in self.routing_weights):
            raise ValueError("routing_weights must be non-negative")

        if sum(self.routing_weights) <= 0:
            raise ValueError("routing_weights must sum to a positive value")

        for llm_task in self.llm_routing_tasks:
            if llm_task.model_id != self.model_id:
                raise ValueError(
                    "All llm_routing_tasks must use the same model_id as MixedMLLMTask"
                )
            if not llm_task.receive_embeddings:
                raise ValueError(
                    "All llm_routing_tasks must set receive_embeddings=True"
                )

        return self

    def post_init(self) -> None:
        """Initialize shared encoder subtasks and LLM router subtask."""
        self.encoders = {
            modality: EncoderTask(
                model_ids=self.encoder_model_ids or {self.model_id},
                modality=modality,
                macro_ut_deployment_id=self.macro_ut_deployment_id,
                max_batch_size=self.eric_max_batch_size,
            )
            for modality in self.modalities
        }
        self.llm_router = RouterApp[
            OpenAIChatCompletionRequest, Stream[OpenAIChatCompletionChunk]
        ](
            routing_tasks=cast(list[Task], self.llm_routing_tasks),
            routing_weights=self.routing_weights,
        )

    def invoke(
        self, task_input: OpenAIChatCompletionRequest
    ) -> Stream[OpenAIChatCompletionChunk]:
        """Invoke the shared encoder path, then route to one LLM route."""
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

        if multimodal_contents:
            # Invoke the shared encoder tasks.
            if self.coalesce_encoder_invocations:
                encoder_outputs: dict[Modality, EncoderOutput] = {}
                for modality, encoder_task in self.encoders.items():
                    if modality not in encoder_input_urls:
                        continue
                    encoder_input = EncoderInput(
                        model_id=task_input.model,
                        data_urls=encoder_input_urls[modality],
                    )
                    encoder_output = encoder_task.invoke(encoder_input)
                    encoder_outputs[modality] = encoder_output

                embeddings: list[DataForward[Tensor]] = []
                for multimodal_content in multimodal_contents:
                    modality = Modality(multimodal_content.type.split("_")[0])
                    embeddings.append(encoder_outputs[modality].embeddings.pop(0))
            else:
                embeddings: list[DataForward[Tensor]] = []
                for multimodal_content in multimodal_contents:
                    modality = Modality(multimodal_content.type.split("_")[0])
                    data_url: URL = getattr(multimodal_content, multimodal_content.type)
                    encoder_input = EncoderInput(
                        model_id=task_input.model,
                        data_urls=[data_url.url],
                    )
                    encoder_output = self.encoders[modality].invoke(encoder_input)
                    embeddings.append(encoder_output.embeddings[0])

            task_input.cornserve_embeddings = embeddings

        # Route to one LLM route and stream response.
        return self.llm_router.invoke(task_input)
