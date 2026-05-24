"""Built-in task for Qwen Omni Thinker and Talker."""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any

from cornserve.task.base import MacroUnitTask, Stream, Task
from cornserve.task.forward import DataForward, Tensor
from pydantic import BaseModel, Field, model_validator

from cornserve_tasklib.task.composite.llm import MLLMEmbeddingTask, MLLMTask
from cornserve_tasklib.task.unit.encoder import (
    EncoderInput,
    EncoderOutput,
    EncoderTask,
)
from cornserve_tasklib.task.unit.encoder import (
    Modality as EncoderModality,
)
from cornserve_tasklib.task.unit.generator import (
    AudioGeneratorInput,
    AudioGeneratorTask,
)
from cornserve_tasklib.task.unit.llm import (
    URL,
    ChatCompletionMessageParam,
    LLMEmbeddingUnitTask,
    LLMUnitTask,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequest,
    extract_multimodal_content,
)
from cornserve_tasklib.task.unit.omni import (
    OmniTalkerEmbeddingTask,
    OmniTalkerVocoderInput,
    OmniTalkerVocoderTask,
)

# Request type bit flags (matching omni_rate_matching.py)
BIT_IMG = 1
BIT_VID = 2
BIT_AUDIO = 4
BIT_RETURN_AUDIO = 8

# Map between full modality names (from message types) and short names (from planner)
_MODALITY_TO_SHORT: dict[str, str] = {"image": "img", "video": "vid", "audio": "audio"}
_SHORT_TO_MODALITY: dict[str, str] = {v: k for k, v in _MODALITY_TO_SHORT.items()}


class OmniInput(OpenAIChatCompletionRequest):
    """Input model for Qwen Omni tasks.

    Attributes:
        return_audio: Whether to return audio response.
    """

    return_audio: bool = True  # same as huggingface parameter
    # use_audio_in_video: not supported yet

    def model_post_init(self, context: Any, /) -> None:
        """Validate the model."""
        pass


class OmniTask(Task[OmniInput, Stream[OpenAIChatCompletionChunk]]):
    """A task that invokes a Multimodal LLM.

    Attributes:
        model_id: The ID of the model to use for the task.
        modalities: List of input modalities other than text.
    """

    model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    modalities: list[EncoderModality] = []
    encoder_fission: bool = True
    vocoder_fission: bool = True
    coalesce_encoder_invocations: bool = True
    macro_ut_deployment_id: str | None = None

    eric_max_batch_size: int
    llm_tp_size: int
    llm_max_num_seqs: int
    llm_gpu_memory_utilization: float

    def post_init(self) -> None:
        """Initialize subtasks."""
        self.thinker_text = MLLMTask(
            model_id=self.model_id,
            encoder_fission=self.encoder_fission,
            modalities=self.modalities,
            coalesce_encoder_invocations=self.coalesce_encoder_invocations,
            macro_ut_deployment_id=self.macro_ut_deployment_id,
            eric_max_batch_size=self.eric_max_batch_size,
            llm_tp_size=self.llm_tp_size,
            llm_max_num_seqs=self.llm_max_num_seqs,
            llm_gpu_memory_utilization=self.llm_gpu_memory_utilization,
        )
        self.thinker_embedding = MLLMEmbeddingTask(
            model_id=self.model_id,
            encoder_fission=self.encoder_fission,
            modalities=self.modalities,
            coalesce_encoder_invocations=self.coalesce_encoder_invocations,
            macro_ut_deployment_id=self.macro_ut_deployment_id,
            eric_max_batch_size=self.eric_max_batch_size,
            llm_tp_size=self.llm_tp_size,
            llm_max_num_seqs=self.llm_max_num_seqs,
            llm_gpu_memory_utilization=self.llm_gpu_memory_utilization,
        )
        if self.vocoder_fission:
            self.talker_vocoder = None
            self.talker_embedding = OmniTalkerEmbeddingTask(
                model_id=self.model_id,
                macro_ut_deployment_id=self.macro_ut_deployment_id,
            )
            self.vocoder_geri = AudioGeneratorTask(
                model_id=self.model_id,
                macro_ut_deployment_id=self.macro_ut_deployment_id,
            )
        else:
            self.talker_vocoder = OmniTalkerVocoderTask(
                model_id=self.model_id,
                macro_ut_deployment_id=self.macro_ut_deployment_id,
            )
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


class OmniMLLMTask(Task[OmniInput, Stream[OpenAIChatCompletionChunk]]):
    """Omni MLLM with audio output tail (TalkerVocoder).

    Supports mono (encoder_fission=False) and full disagg (encoder_fission=True).
    Pipeline: Encoders → Thinker → [TalkerVocoder or Talker→AudioGeri if return_audio] → text/audio.
    """

    model_id: str
    modalities: list[EncoderModality] = []
    encoder_fission: bool = True
    vocoder_fission: bool = False
    coalesce_encoder_invocations: bool = True
    macro_ut_deployment_id: str | None = None

    eric_max_batch_size: int
    llm_tp_size: int
    llm_max_num_seqs: int
    llm_gpu_memory_utilization: float

    def post_init(self) -> None:
        """Initialize subtasks."""
        self.thinker_text = MLLMTask(
            model_id=self.model_id,
            encoder_fission=self.encoder_fission,
            modalities=self.modalities,
            coalesce_encoder_invocations=self.coalesce_encoder_invocations,
            macro_ut_deployment_id=self.macro_ut_deployment_id,
            eric_max_batch_size=self.eric_max_batch_size,
            llm_tp_size=self.llm_tp_size,
            llm_max_num_seqs=self.llm_max_num_seqs,
            llm_gpu_memory_utilization=self.llm_gpu_memory_utilization,
        )
        self.thinker_embedding = MLLMEmbeddingTask(
            model_id=self.model_id,
            encoder_fission=self.encoder_fission,
            modalities=self.modalities,
            coalesce_encoder_invocations=self.coalesce_encoder_invocations,
            macro_ut_deployment_id=self.macro_ut_deployment_id,
            eric_max_batch_size=self.eric_max_batch_size,
            llm_tp_size=self.llm_tp_size,
            llm_max_num_seqs=self.llm_max_num_seqs,
            llm_gpu_memory_utilization=self.llm_gpu_memory_utilization,
        )
        if self.vocoder_fission:
            self.talker_vocoder = None
            self.talker_embedding = OmniTalkerEmbeddingTask(
                model_id=self.model_id,
                macro_ut_deployment_id=self.macro_ut_deployment_id,
            )
            self.vocoder_geri = AudioGeneratorTask(
                model_id=self.model_id,
                macro_ut_deployment_id=self.macro_ut_deployment_id,
            )
        else:
            self.talker_vocoder = OmniTalkerVocoderTask(
                model_id=self.model_id,
                macro_ut_deployment_id=self.macro_ut_deployment_id,
            )
            self.talker_embedding = None
            self.vocoder_geri = None

    def invoke(self, task_input: OmniInput) -> Stream[OpenAIChatCompletionChunk]:
        """Invoke the task."""
        if not task_input.return_audio:
            thinker_input = OpenAIChatCompletionRequest.model_validate(
                dict(**task_input.model_dump(exclude={"return_audio"}))
            )
            return self.thinker_text.invoke(thinker_input)

        # Save original input before thinker invocation — the thinker pipeline
        # may rewrite multimodal URLs in-place, and the talker needs originals.
        original_dump = task_input.model_dump(exclude={"return_audio"})

        # Embedding path: strip stream_options because vLLM rejects
        # stream_options without stream=True (non-streaming embedding request).
        thinker_input = OpenAIChatCompletionRequest.model_validate(
            dict(**task_input.model_dump(exclude={"return_audio", "stream_options"}))
        )
        thinker_embedding_output = self.thinker_embedding.invoke(thinker_input)
        talker_input = OmniTalkerVocoderInput.model_validate(
            dict(
                **original_dump,
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


def _compute_type_index(
    task_input: OmniInput,
) -> int:
    """Compute 4-bit type index from request content.

    bit 0 = has image, bit 1 = has video, bit 2 = has audio, bit 3 = return_audio.
    """
    bits = 0
    multimodal = extract_multimodal_content(task_input.messages)
    for mm in multimodal:
        modality = mm.type.split("_")[0]
        if modality == "image":
            bits |= BIT_IMG
        elif modality == "video":
            bits |= BIT_VID
        elif modality == "audio":
            bits |= BIT_AUDIO
    if getattr(task_input, "return_audio", False):
        bits |= BIT_RETURN_AUDIO
    return bits


def _select_mode(
    request_id: str,
    weights: dict[str, float],
) -> frozenset[str]:
    """Deterministically select an offloading mode from per-type weights.

    Mode keys are comma-joined sorted modality names (empty string = mono/∅).
    """
    items = sorted(weights.items())
    total = sum(w for _, w in items)
    if total <= 0:
        return frozenset()
    threshold = (int(request_id, 16) % 10000) / 10000
    cumulative = 0.0
    for mode_key, w in items:
        cumulative += w / total
        if threshold < cumulative:
            return frozenset(mode_key.split(",")) if mode_key else frozenset()
    # fallback to last
    last_key = items[-1][0]
    return frozenset(last_key.split(",")) if last_key else frozenset()


class OmniTimeSharingMLLMTask(Task[OmniInput, Stream[OpenAIChatCompletionChunk]]):
    """Omni MLLM with per-request-type multipath routing and audio output.

    For partial disagg: a subset of modality encoders are deployed. Each request
    type (defined by which modalities it carries) is routed to a mode (subset of
    deployed encoders to actually offload) based on per-type probability weights.

    The LLM runs with receive_embeddings=False so it can handle both offloaded
    (with cornserve_embeddings set) and non-offloaded requests.
    """

    model_id: str
    offloaded_modalities: list[EncoderModality] = []
    # Per-type routing weights: {type_index: {mode_key: weight}}
    # type_index: 4-bit int (bit0=img, bit1=vid, bit2=audio, bit3=return_audio)
    # mode_key: comma-joined sorted modality names, "" for mono/∅
    type_routing_weights: dict[int, dict[str, float]] = {}
    vocoder_fission: bool = False
    coalesce_encoder_invocations: bool = True
    macro_ut_deployment_id: str | None = None

    eric_max_batch_size: int
    llm_tp_size: int
    llm_max_num_seqs: int
    llm_gpu_memory_utilization: float

    def post_init(self) -> None:
        """Initialize subtasks."""
        self.encoders = {
            modality: EncoderTask(
                model_ids={self.model_id},
                modality=modality,
                macro_ut_deployment_id=self.macro_ut_deployment_id,
                max_batch_size=self.eric_max_batch_size,
            )
            for modality in self.offloaded_modalities
        }
        self.llm = LLMUnitTask(
            model_id=self.model_id,
            receive_embeddings=False,
            macro_ut_deployment_id=self.macro_ut_deployment_id,
            tp_size=self.llm_tp_size,
            max_num_seqs=self.llm_max_num_seqs,
            gpu_memory_utilization=self.llm_gpu_memory_utilization,
        )
        self.llm_embedding = LLMEmbeddingUnitTask(
            model_id=self.model_id,
            receive_embeddings=False,
            macro_ut_deployment_id=self.macro_ut_deployment_id,
            tp_size=self.llm_tp_size,
            max_num_seqs=self.llm_max_num_seqs,
            gpu_memory_utilization=self.llm_gpu_memory_utilization,
        )
        if self.vocoder_fission:
            self.talker_vocoder = None
            self.talker_embedding_task = OmniTalkerEmbeddingTask(
                model_id=self.model_id,
                macro_ut_deployment_id=self.macro_ut_deployment_id,
            )
            self.vocoder_geri = AudioGeneratorTask(
                model_id=self.model_id,
                macro_ut_deployment_id=self.macro_ut_deployment_id,
            )
        else:
            self.talker_vocoder = OmniTalkerVocoderTask(
                model_id=self.model_id,
                macro_ut_deployment_id=self.macro_ut_deployment_id,
            )
            self.talker_embedding_task = None
            self.vocoder_geri = None

    def invoke(self, task_input: OmniInput) -> Stream[OpenAIChatCompletionChunk]:
        """Invoke with per-type multipath routing."""
        type_idx = _compute_type_index(task_input)
        weights = self.type_routing_weights.get(type_idx, {"": 1.0})
        selected_mode = _select_mode(task_input.request_id, weights)

        # Offload encoders for selected modalities
        # selected_mode contains short names ('img', 'vid', 'audio')
        if selected_mode:
            multimodal_contents = extract_multimodal_content(task_input.messages)
            encoder_input_urls: dict[EncoderModality, list[str]] = defaultdict(list)
            for mm in multimodal_contents:
                modality = EncoderModality(mm.type.split("_")[0])
                short_name = _MODALITY_TO_SHORT.get(modality.value, modality.value)
                if short_name in selected_mode and modality in self.encoders:
                    data_url: URL = getattr(mm, mm.type)
                    encoder_input_urls[modality].append(data_url.url)

            if encoder_input_urls:
                if self.coalesce_encoder_invocations:
                    encoder_outputs: dict[EncoderModality, EncoderOutput] = {}
                    for modality, encoder_task in self.encoders.items():
                        if modality not in encoder_input_urls:
                            continue
                        encoder_output = encoder_task.invoke(
                            EncoderInput(
                                model_id=task_input.model,
                                data_urls=encoder_input_urls[modality],
                            )
                        )
                        encoder_outputs[modality] = encoder_output

                    embeddings: list[DataForward[Tensor]] = []
                    for mm in multimodal_contents:
                        modality = EncoderModality(mm.type.split("_")[0])
                        if modality in encoder_outputs:
                            embeddings.append(
                                encoder_outputs[modality].embeddings.pop(0)
                            )
                else:
                    embeddings: list[DataForward[Tensor]] = []
                    for mm in multimodal_contents:
                        modality = EncoderModality(mm.type.split("_")[0])
                        short_name = _MODALITY_TO_SHORT.get(
                            modality.value, modality.value
                        )
                        if short_name in selected_mode and modality in self.encoders:
                            data_url: URL = getattr(mm, mm.type)
                            encoder_output = self.encoders[modality].invoke(
                                EncoderInput(
                                    model_id=task_input.model,
                                    data_urls=[data_url.url],
                                )
                            )
                            embeddings.append(encoder_output.embeddings[0])

                task_input.cornserve_embeddings = embeddings

                # Rewrite offloaded modality URLs to sidecar format so vLLM
                # treats them as remote inputs (receive_embeddings=False means
                # the VLLMDescriptor won't rewrite them for us).
                embed_idx = 0
                for mm in multimodal_contents:
                    modality = EncoderModality(mm.type.split("_")[0])
                    short_name = _MODALITY_TO_SHORT.get(modality.value, modality.value)
                    if short_name in selected_mode and modality in self.encoders:
                        data_url: URL = getattr(mm, mm.type)
                        forward = embeddings[embed_idx]
                        data_url.url = f"data:{modality.value}/uuid;data_id={forward.id};url={data_url.url},"
                        embed_idx += 1

        # Save original input before thinker invocation — the thinker/encoder
        # pipeline may rewrite multimodal URLs in-place, and the talker needs originals.
        original_dump = task_input.model_dump(exclude={"return_audio"})

        # Route to LLM
        if not task_input.return_audio:
            thinker_input = OpenAIChatCompletionRequest.model_validate(
                dict(**task_input.model_dump(exclude={"return_audio"}))
            )
            return self.llm.invoke(thinker_input)

        # Audio path: strip stream_options for non-streaming embedding request
        # (vLLM rejects stream_options without stream=True)
        thinker_input = OpenAIChatCompletionRequest.model_validate(
            dict(**task_input.model_dump(exclude={"return_audio", "stream_options"}))
        )
        embedding_output = self.llm_embedding.invoke(thinker_input)
        talker_input = OmniTalkerVocoderInput.model_validate(
            dict(
                **original_dump,
                thinker_hidden_states=embedding_output.embeddings,
            )
        )
        talker_input.messages.append(
            ChatCompletionMessageParam.model_validate(
                {"role": "assistant", "content": "<tts_pad>"}
            )
        )

        if self.vocoder_fission:
            assert self.talker_embedding_task is not None
            assert self.vocoder_geri is not None
            talker_emb_output = self.talker_embedding_task.invoke(talker_input)
            vocoder_input = AudioGeneratorInput(
                embeddings=talker_emb_output.embeddings,
            )
            return self.vocoder_geri.invoke(vocoder_input)

        assert self.talker_vocoder is not None
        return self.talker_vocoder.invoke(talker_input)


class ThinkerLLMConfig(BaseModel):
    """One thinker LLM variant within a group."""

    tp_size: int
    max_num_seqs: int
    gpu_memory_utilization: float = 0.9
    weight: float = 1.0  # within-group routing weight


class OmniFlexGroupConfig(BaseModel):
    """One encoder+thinker group with its own encoder pool + LLM configs."""

    # Encoder config (per-modality, independent per group)
    offloaded_modalities: list[str] = []  # short names: "img", "vid", "audio"
    eric_max_batch_sizes: dict[str, int] = {}  # e.g. {"img": 4, "vid": 2}
    # Thinker LLM configs (multiple = weighted sub-routing within group)
    thinkers: list[ThinkerLLMConfig]


def _select_group(
    request_id: str,
    cdf: list[tuple[int, float]],
) -> int:
    """Deterministically select a group index from a pre-computed CDF.

    Uses the same modular hash as ``_select_mode`` for consistency.
    """
    threshold = (int(request_id, 16) % 10000) / 10000
    for gidx, cutoff in cdf:
        if threshold < cutoff:
            return gidx
    return cdf[-1][0]


def _select_thinker(
    request_id: str,
    cdf: list[float],
) -> int:
    """Deterministically select a thinker within a group.

    Uses the second half of ``request_id`` (hex) so the selection is
    independent of ``_select_group`` / ``_select_mode`` which use the
    full ``request_id``.  ``request_id`` is a ``uuid4().hex`` (32 hex
    chars = 128 random bits), so the two halves are independent.
    """
    threshold = (int(request_id[16:], 16) % 10000) / 10000
    for i, cutoff in enumerate(cdf):
        if threshold < cutoff:
            return i
    return len(cdf) - 1


class OmniFlexTask(Task[OmniInput, Stream[OpenAIChatCompletionChunk]]):
    """Flexible Omni composite task supporting all planner deployment topologies.

    Each **group** is a self-contained encoder+thinker pipeline (with its own
    encoder pool, encoder batch sizes, and one or more weighted LLM configs).
    All groups share a single audio output (TalkerVocoder or Talker+AudioGeri).

    Per-type routing selects which group handles each request.  Within a group,
    weighted sub-routing selects which thinker LLM config to use.

    Attributes:
        groups: List of encoder+thinker group configurations.
        type_routing_weights: Per-type routing to groups.
            ``{type_index: {group_index: weight}}``.
            Must be set for every active request type when ``len(groups) > 1``.
            May be empty when there is only one group (all types → group 0).
        vocoder_fission: Whether to split talker and vocoder.
    """

    model_id: str

    groups: list[OmniFlexGroupConfig]

    # Per-type routing to groups: {type_index: {group_index: weight}}
    type_routing_weights: dict[int, dict[int, float]] = {}

    # Shared audio output
    vocoder_fission: bool = False
    coalesce_encoder_invocations: bool = True
    macro_ut_deployment_id: str | None = None

    @model_validator(mode="after")
    def _validate_flex_config(self) -> OmniFlexTask:
        if not self.groups:
            raise ValueError("groups must contain at least one group")
        for gi, group in enumerate(self.groups):
            if not group.thinkers:
                raise ValueError(f"group {gi}: thinkers must be non-empty")
            for mod in group.offloaded_modalities:
                if mod not in group.eric_max_batch_sizes:
                    raise ValueError(
                        f"group {gi}: offloaded modality '{mod}' missing "
                        f"from eric_max_batch_sizes"
                    )
            for ti, cfg in enumerate(group.thinkers):
                if cfg.weight <= 0:
                    raise ValueError(
                        f"group {gi} thinker {ti}: weight must be positive"
                    )
        for type_idx, gweights in self.type_routing_weights.items():
            for gidx, w in gweights.items():
                if gidx < 0 or gidx >= len(self.groups):
                    raise ValueError(
                        f"type_routing_weights[{type_idx}]: "
                        f"group index {gidx} out of range"
                    )
                if w < 0:
                    raise ValueError(
                        f"type_routing_weights[{type_idx}][{gidx}]: "
                        f"weight must be non-negative"
                    )
        return self

    def post_init(self) -> None:
        """Initialize per-group encoders/LLMs and shared audio output."""
        base_id = self.macro_ut_deployment_id

        # ── Per-group encoders ───────────────────────────────────────
        self._group_encoders: list[dict[EncoderModality, EncoderTask]] = []
        for i, group in enumerate(self.groups):
            group_dep_id = f"{base_id}_g{i}" if base_id else None
            encoders: dict[EncoderModality, EncoderTask] = {}
            for mod_short in group.offloaded_modalities:
                modality = EncoderModality(_SHORT_TO_MODALITY[mod_short])
                encoders[modality] = EncoderTask(
                    model_ids={self.model_id},
                    modality=modality,
                    max_batch_size=group.eric_max_batch_sizes.get(mod_short, 1),
                    macro_ut_deployment_id=group_dep_id,
                )
            self._group_encoders.append(encoders)

        # ── Per-group LLM pairs (text + embedding) ──────────────────
        self._group_llm_pairs: list[list[tuple[LLMUnitTask, LLMEmbeddingUnitTask]]] = []
        self._group_llm_cdfs: list[list[float]] = []
        for i, group in enumerate(self.groups):
            group_dep_id = f"{base_id}_g{i}" if base_id else None
            pairs: list[tuple[LLMUnitTask, LLMEmbeddingUnitTask]] = []
            for cfg in group.thinkers:
                text_llm = LLMUnitTask(
                    model_id=self.model_id,
                    receive_embeddings=False,
                    macro_ut_deployment_id=group_dep_id,
                    tp_size=cfg.tp_size,
                    max_num_seqs=cfg.max_num_seqs,
                    gpu_memory_utilization=cfg.gpu_memory_utilization,
                )
                emb_llm = LLMEmbeddingUnitTask(
                    model_id=self.model_id,
                    receive_embeddings=False,
                    macro_ut_deployment_id=group_dep_id,
                    tp_size=cfg.tp_size,
                    max_num_seqs=cfg.max_num_seqs,
                    gpu_memory_utilization=cfg.gpu_memory_utilization,
                )
                pairs.append((text_llm, emb_llm))
            self._group_llm_pairs.append(pairs)

            # Pre-compute CDF for within-group thinker selection
            weights = [cfg.weight for cfg in group.thinkers]
            total = sum(weights) if weights else 1.0
            cdf: list[float] = []
            running = 0.0
            for w in weights:
                running += w / total
                cdf.append(running)
            if cdf:
                cdf[-1] = 1.0
            self._group_llm_cdfs.append(cdf)

        # ── Per-type group routing CDFs ──────────────────────────────
        self._type_group_cdfs: dict[int, list[tuple[int, float]]] = {}
        for type_idx, gweights in self.type_routing_weights.items():
            sorted_items = sorted(gweights.items())
            total = sum(w for _, w in sorted_items)
            if total <= 0:
                continue
            cdf_items: list[tuple[int, float]] = []
            running = 0.0
            for gidx, w in sorted_items:
                running += w / total
                cdf_items.append((gidx, running))
            cdf_items[-1] = (cdf_items[-1][0], 1.0)
            self._type_group_cdfs[type_idx] = cdf_items

        # ── Shared audio output (only Qwen3-Omni has talker/vocoder) ──
        _has_audio = self.model_id == "Qwen/Qwen3-Omni-30B-A3B-Instruct"
        ao_dep_id = f"{base_id}_ao" if base_id else None
        if not _has_audio:
            self.talker_vocoder = None
            self.flex_talker_embedding = None
            self.flex_vocoder_geri = None
        elif self.vocoder_fission:
            self.talker_vocoder = None
            self.flex_talker_embedding = OmniTalkerEmbeddingTask(
                model_id=self.model_id,
                macro_ut_deployment_id=ao_dep_id,
            )
            self.flex_vocoder_geri = AudioGeneratorTask(
                model_id=self.model_id,
                macro_ut_deployment_id=ao_dep_id,
            )
        else:
            self.talker_vocoder = OmniTalkerVocoderTask(
                model_id=self.model_id,
                macro_ut_deployment_id=ao_dep_id,
            )
            self.flex_talker_embedding = None
            self.flex_vocoder_geri = None

    def _get_group_idx(self, request_id: str, type_idx: int) -> int:
        """Select the group index for a request."""
        if len(self.groups) == 1:
            return 0
        cdf = self._type_group_cdfs[type_idx]
        return _select_group(request_id, cdf)

    def _get_thinker_idx(self, request_id: str, group_idx: int) -> int:
        """Select the thinker index within a group."""
        cdf = self._group_llm_cdfs[group_idx]
        if len(cdf) <= 1:
            return 0
        return _select_thinker(request_id, cdf)

    def _invoke_encoders(
        self,
        task_input: OmniInput,
        group_idx: int,
    ) -> None:
        """Invoke group's encoders for offloaded modalities and rewrite URLs.

        Modifies ``task_input`` in-place: sets ``cornserve_embeddings`` and
        rewrites multimodal URLs to sidecar data format.
        """
        encoders = self._group_encoders[group_idx]
        if not encoders:
            return

        group = self.groups[group_idx]
        offloaded_short = frozenset(group.offloaded_modalities)

        multimodal_contents = extract_multimodal_content(task_input.messages)

        # Collect URLs per modality
        encoder_input_urls: dict[EncoderModality, list[str]] = defaultdict(list)
        for mm in multimodal_contents:
            modality = EncoderModality(mm.type.split("_")[0])
            short_name = _MODALITY_TO_SHORT.get(modality.value, modality.value)
            if short_name in offloaded_short and modality in encoders:
                data_url: URL = getattr(mm, mm.type)
                encoder_input_urls[modality].append(data_url.url)

        if not encoder_input_urls:
            return

        # Invoke encoders
        if self.coalesce_encoder_invocations:
            encoder_outputs: dict[EncoderModality, EncoderOutput] = {}
            for modality, encoder_task in encoders.items():
                if modality not in encoder_input_urls:
                    continue
                encoder_output = encoder_task.invoke(
                    EncoderInput(
                        model_id=task_input.model,
                        data_urls=encoder_input_urls[modality],
                    )
                )
                encoder_outputs[modality] = encoder_output

            embeddings: list[DataForward[Tensor]] = []
            for mm in multimodal_contents:
                modality = EncoderModality(mm.type.split("_")[0])
                if modality in encoder_outputs:
                    embeddings.append(encoder_outputs[modality].embeddings.pop(0))
        else:
            embeddings = []
            for mm in multimodal_contents:
                modality = EncoderModality(mm.type.split("_")[0])
                short_name = _MODALITY_TO_SHORT.get(modality.value, modality.value)
                if short_name in offloaded_short and modality in encoders:
                    data_url = getattr(mm, mm.type)
                    encoder_output = encoders[modality].invoke(
                        EncoderInput(
                            model_id=task_input.model,
                            data_urls=[data_url.url],
                        )
                    )
                    embeddings.append(encoder_output.embeddings[0])

        task_input.cornserve_embeddings = embeddings

        # Rewrite offloaded modality URLs to sidecar data format
        embed_idx = 0
        for mm in multimodal_contents:
            modality = EncoderModality(mm.type.split("_")[0])
            short_name = _MODALITY_TO_SHORT.get(modality.value, modality.value)
            if short_name in offloaded_short and modality in encoders:
                data_url = getattr(mm, mm.type)
                forward = embeddings[embed_idx]
                data_url.url = (
                    f"data:{modality.value}/uuid;"
                    f"data_id={forward.id};url={data_url.url},"
                )
                embed_idx += 1

    def invoke(self, task_input: OmniInput) -> Stream[OpenAIChatCompletionChunk]:
        """Invoke with per-type group routing and within-group thinker selection."""
        # Non-Qwen models have no audio output; force return_audio off so
        # _compute_type_index doesn't set BIT_RETURN_AUDIO.
        if self.model_id != "Qwen/Qwen3-Omni-30B-A3B-Instruct":
            task_input.return_audio = False
        type_idx = _compute_type_index(task_input)
        group_idx = self._get_group_idx(task_input.request_id, type_idx)
        thinker_idx = self._get_thinker_idx(task_input.request_id, group_idx)
        text_llm, emb_llm = self._group_llm_pairs[group_idx][thinker_idx]

        # Invoke group's encoders (modifies task_input in-place)
        self._invoke_encoders(task_input, group_idx)

        # Save original input before thinker — talker needs original URLs
        _has_audio = self.model_id == "Qwen/Qwen3-Omni-30B-A3B-Instruct"
        _return_audio = _has_audio and getattr(task_input, "return_audio", False)
        original_dump = task_input.model_dump(exclude={"return_audio"})

        # Text-only path
        if not _return_audio:
            thinker_input = OpenAIChatCompletionRequest.model_validate(
                dict(**task_input.model_dump(exclude={"return_audio"}))
            )
            return text_llm.invoke(thinker_input)

        # Audio path: strip stream_options (vLLM rejects without stream=True)
        thinker_input = OpenAIChatCompletionRequest.model_validate(
            dict(**task_input.model_dump(exclude={"return_audio", "stream_options"}))
        )
        embedding_output = emb_llm.invoke(thinker_input)

        talker_input = OmniTalkerVocoderInput.model_validate(
            dict(
                **original_dump,
                thinker_hidden_states=embedding_output.embeddings,
            )
        )
        talker_input.messages.append(
            ChatCompletionMessageParam.model_validate(
                {"role": "assistant", "content": "<tts_pad>"}
            )
        )

        if self.vocoder_fission:
            assert self.flex_talker_embedding is not None
            assert self.flex_vocoder_geri is not None
            talker_emb_output = self.flex_talker_embedding.invoke(talker_input)
            vocoder_input = AudioGeneratorInput(
                embeddings=talker_emb_output.embeddings,
            )
            return self.flex_vocoder_geri.invoke(vocoder_input)

        assert self.talker_vocoder is not None
        return self.talker_vocoder.invoke(talker_input)


class Qwen3OmniMacroUnitTask(
    MacroUnitTask[OmniInput, Stream[OpenAIChatCompletionChunk]]
):
    model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    modalities: list[EncoderModality] = []
    encoder_fission: bool = True
    vocoder_fission: bool = True
    coalesce_encoder_invocations: bool = True
    macro_ut_deployment_id: str | None = Field(default_factory=lambda: uuid.uuid4().hex)

    eric_max_batch_size: int
    llm_tp_size: int
    llm_max_num_seqs: int
    llm_gpu_memory_utilization: float

    def post_init(self) -> None:
        self.omni = OmniTask(
            model_id=self.model_id,
            modalities=self.modalities,
            encoder_fission=self.encoder_fission,
            vocoder_fission=self.vocoder_fission,
            coalesce_encoder_invocations=self.coalesce_encoder_invocations,
            macro_ut_deployment_id=self.macro_ut_deployment_id,
            eric_max_batch_size=self.eric_max_batch_size,
            llm_tp_size=self.llm_tp_size,
            llm_max_num_seqs=self.llm_max_num_seqs,
            llm_gpu_memory_utilization=self.llm_gpu_memory_utilization,
        )

    def make_name(self) -> str:
        return f"macro-{self.model_id.split('/')[-1].lower().replace('.', '-')}"

    def make_record_output(
        self,
        task_input: OmniInput,
    ) -> Stream[OpenAIChatCompletionChunk]:
        return Stream[OpenAIChatCompletionChunk]()

    def expand(self, task_input: OmniInput) -> Stream[OpenAIChatCompletionChunk]:
        return self.omni.invoke(task_input)
