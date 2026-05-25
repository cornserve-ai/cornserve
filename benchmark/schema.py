"""Schema definitions for benchmark configurations and data handling."""

from __future__ import annotations
from abc import abstractmethod
import datetime
import json
from pathlib import Path
from typing import Any, Literal

from cornserve_tasklib.task.unit.encoder import EricProfileConfig
from cornserve_tasklib.task.unit.generator import (
    AudioGeriProfileConfig,
    ImageGeriProfileConfig,
)
from cornserve_tasklib.task.unit.llm import LLMProfileConfig
from cornserve_tasklib.task.unit.omni import TalkerProfileConfig
from pydantic import BaseModel, SerializeAsAny, model_validator

from benchmark_datasets import Dataset, ReplayedOmniServegen, ReplayedServegen, ServeGenDataset

DATA_ROOT = "data"


def _format_float(value: float) -> str:
    """Format float values for stable directory naming."""
    formatted = f"{value:.6f}".rstrip("0").rstrip(".")
    return formatted if formatted else "0"


class AppType(BaseModel):
    """Base class for a benchmark app."""

    model_id: str

    @abstractmethod
    def _to_app_dir(self) -> str:
        """Return the directory name for the app type."""
        pass

    @abstractmethod
    def _to_config_dir(self) -> str:
        """Return the directory name for the app config."""
        pass

    def to_dir(self) -> Path:
        return (
            Path(DATA_ROOT) / self._to_app_dir() / self.model_id / self._to_config_dir()
        )


# singleton (unit task)
class DummyEricApp(AppType, EricProfileConfig):
    """Eric standalone."""

    num_replicas: int
    modality: Literal["image", "video", "audio"] = "image"

    def _to_app_dir(self) -> str:
        """Return the directory name for the app type."""
        return "eric"

    def _to_config_dir(self) -> str:
        """Return the file name for the backend configuration."""
        subdir_name = f"{self.modality}+replicas{self.num_replicas}"
        subdir_name += f"+{self.to_profile_str()}"
        return subdir_name


class DummyLLMApp(AppType, LLMProfileConfig):
    """Dummy LLM standalone."""

    num_replicas: int

    def _to_app_dir(self) -> str:
        """Return the directory name for the app type."""
        return "llm"

    def _to_config_dir(self) -> str:
        """Return the file name for the backend configuration."""
        subdir_name = f"replicas{self.num_replicas}"
        subdir_name += f"+{self.to_profile_str()}"
        return subdir_name


class MonolithicLLMApp(AppType, LLMProfileConfig):
    """Monolithic LLM (encoder + LLM together)."""

    num_replicas: int

    def _to_app_dir(self) -> str:
        """Return the directory name for the app type."""
        return "monolithic-mllm"

    def _to_config_dir(self) -> str:
        """Return the file name for the backend configuration."""
        subdir_name = f"replicas{self.num_replicas}"
        subdir_name += f"+{self.to_profile_str()}"
        return subdir_name


class PrefillLLMApp(AppType, LLMProfileConfig):
    """Standalone LLM for prefill profiling (dataset uses output_len=1)."""

    num_replicas: int
    receive_embeddings: bool = False

    def _to_app_dir(self) -> str:
        return "prefill-llm"

    def _to_config_dir(self) -> str:
        re_str = "re1" if self.receive_embeddings else "re0"
        return f"replicas{self.num_replicas}+{re_str}+{self.to_profile_str()}"


class DecodeLLMApp(AppType, LLMProfileConfig):
    """LLM with prefix caching for decode profiling (all requests share a fixed prefix)."""

    num_replicas: int
    receive_embeddings: bool = True

    def _to_app_dir(self) -> str:
        return "decode-llm"

    def _to_config_dir(self) -> str:
        re_str = "re1" if self.receive_embeddings else "re0"
        return f"replicas{self.num_replicas}+{re_str}+{self.to_profile_str()}"


class DisaggregatedMLLMProfileConfig(BaseModel):
    """Shared prefill/decode fields and helpers for PD and EPD MLLM apps."""

    num_prefill_replicas: int
    prefill_tp_size: int
    prefill_max_num_seqs: int
    prefill_gpu_memory_utilization: float

    num_decode_replicas: int
    decode_tp_size: int
    decode_max_num_seqs: int
    decode_gpu_memory_utilization: float

    def prefill_profile_str(self) -> str:
        return f"tp{self.prefill_tp_size}+bs{self.prefill_max_num_seqs}+gpu{self.prefill_gpu_memory_utilization}"

    def decode_profile_str(self) -> str:
        return f"tp{self.decode_tp_size}+bs{self.decode_max_num_seqs}+gpu{self.decode_gpu_memory_utilization}"


class PDMLLMApp(AppType, DisaggregatedMLLMProfileConfig):
    """Disaggregated prefill+decode MLLM (no separate encoder; encoder_fission=False).

    PD topology: ``num_prefill_replicas`` prefill workers (each handles its own
    multimodal encoding) + ``num_decode_replicas`` decode workers.
    """

    def _to_app_dir(self) -> str:
        return "pd-mllm"

    def _to_config_dir(self) -> str:
        return (
            f"prefills{self.num_prefill_replicas}+{self.prefill_profile_str()}"
            f"+decodes{self.num_decode_replicas}+{self.decode_profile_str()}"
        )


class EPDMLLMApp(AppType, DisaggregatedMLLMProfileConfig):
    """Disaggregated encoder+prefill+decode MLLM (encoder_fission=True).

    EPD topology: ``num_eric_replicas`` Eric encoders + ``num_prefill_replicas``
    prefill workers (receive embeddings from Eric) + ``num_decode_replicas``
    decode workers.
    """

    num_eric_replicas: int
    eric_max_batch_size: int

    def _to_app_dir(self) -> str:
        return "epd-mllm"

    def _to_config_dir(self) -> str:
        return (
            f"erics{self.num_eric_replicas}+eric_bs{self.eric_max_batch_size}"
            f"+prefills{self.num_prefill_replicas}+{self.prefill_profile_str()}"
            f"+decodes{self.num_decode_replicas}+{self.decode_profile_str()}"
        )


class OmniMLLMApp(AppType, LLMProfileConfig):
    """Omni MLLM with selective encoder disabling for profiling."""

    num_replicas: int
    disable_audio_enc: bool = False
    disable_image_enc: bool = False
    disable_video_enc: bool = False

    def _to_app_dir(self) -> str:
        return "omni-mllm"

    def _enc_flags_str(self) -> str:
        """Short string: i0=image off, a0=audio off, v0=video off."""
        parts = [
            f"i{'0' if self.disable_image_enc else '1'}",
            f"a{'0' if self.disable_audio_enc else '1'}",
            f"v{'0' if self.disable_video_enc else '1'}",
        ]
        return "+".join(parts)

    def _to_config_dir(self) -> str:
        return f"replicas{self.num_replicas}+{self._enc_flags_str()}+{self.to_profile_str()}"


class OmniApp(AppType):
    """Single Omni deployment: encoders + thinker + audio output, no routing."""

    model_id: str

    # Encoder config
    encoder_fission: bool = True
    img_eric_num_replicas: int = 0
    vid_eric_num_replicas: int = 0
    audio_eric_num_replicas: int = 0
    eric_max_batch_size: int = 1

    # LLM (thinker) config
    llm_num_replicas: int = 1
    llm_tp_size: int = 1
    llm_max_num_seqs: int = 32
    llm_gpu_memory_utilization: float = 0.9

    # Audio output config
    vocoder_fission: bool = False
    talker_vocoder_num_replicas: int = 0  # for vocoder_fission=False
    talker_num_replicas: int = 0          # for vocoder_fission=True
    audio_geri_num_replicas: int = 0      # for vocoder_fission=True
    talker_max_num_seqs: int = 256
    audio_geri_max_batch_size: int = 1

    def _to_app_dir(self) -> str:
        return "omni"

    def _to_config_dir(self) -> str:
        enc = (
            f"ei{self.img_eric_num_replicas}"
            f"_ev{self.vid_eric_num_replicas}"
            f"_ea{self.audio_eric_num_replicas}"
        )
        llm = f"tp{self.llm_tp_size}x{self.llm_num_replicas}bs{self.llm_max_num_seqs}"
        if self.vocoder_fission:
            ao = f"tk{self.talker_num_replicas}_ag{self.audio_geri_num_replicas}"
        else:
            ao = f"tv{self.talker_vocoder_num_replicas}"
        if self.talker_max_num_seqs != 256:
            ao += f"_tkbs{self.talker_max_num_seqs}"
        if self.vocoder_fission and self.audio_geri_max_batch_size != 1:
            ao += f"_agbs{self.audio_geri_max_batch_size}"
        return f"{enc}_{llm}_{ao}"


class DummyImageGeriApp(AppType, ImageGeriProfileConfig):
    """Dummy Image Generator standalone."""

    num_replicas: int

    def _to_app_dir(self) -> str:
        """Return the directory name for the app type."""
        return "geri"

    def _to_config_dir(self) -> str:
        """Return the file name for the backend configuration."""
        subdir_name = f"image+replicas{self.num_replicas}"
        subdir_name += f"+{self.to_profile_str()}"
        return subdir_name


class QwenImageTextGeriApp(AppType, ImageGeriProfileConfig):
    """Direct-text Qwen-Image Geri standalone."""

    num_replicas: int

    def _to_app_dir(self) -> str:
        """Return the directory name for the app type."""
        return "geri"

    def _to_config_dir(self) -> str:
        """Return the file name for the backend configuration."""
        subdir_name = f"image-text+replicas{self.num_replicas}"
        subdir_name += f"+{self.to_profile_str()}"
        return subdir_name


class QwenImageDisaggApp(AppType):
    """Disaggregated Qwen-Image app (LLM encoder + Geri image generator)."""

    encoder_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    num_encoders: int
    encoder_tp_size: int = 1
    encoder_max_num_seqs: int = 256
    encoder_gpu_memory_utilization: float = 0.9

    num_generators: int = 1
    generator_sp_size: int = 2
    generator_max_batch_size: int = 1
    num_prefix_tokens_to_slice: int = 34

    def _to_app_dir(self) -> str:
        """Return the directory name for the app type."""
        return "qwen-image-disagg"

    def _to_config_dir(self) -> str:
        """Return the file name for the backend configuration."""
        return (
            f"enc{self.num_encoders}+tp{self.encoder_tp_size}+"
            f"encbs{self.encoder_max_num_seqs}+gpu{_format_float(self.encoder_gpu_memory_utilization)}+"
            f"geri{self.num_generators}+sp{self.generator_sp_size}+gbs{self.generator_max_batch_size}+"
            f"skip{self.num_prefix_tokens_to_slice}"
        )


class MixedImageGeneratorRouteConfig(BaseModel):
    """Per-route generator config for MixedQwenImageDisaggApp."""

    sp_size: int
    num_replicas: int
    max_batch_size: int = 1


class MixedQwenImageDisaggApp(AppType):
    """Disaggregated Qwen-Image with mixed SP generators and weighted routing.

    Shared LLM encoder feeds text embeddings to a RouterApp that dispatches
    to multiple ImageGeneratorTask instances (potentially different sp_size).
    """

    encoder_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    num_encoders: int
    encoder_tp_size: int = 1
    encoder_max_num_seqs: int = 256
    encoder_gpu_memory_utilization: float = 0.9

    generator_routes: list[MixedImageGeneratorRouteConfig]
    routing_weights: list[float]
    generator_max_batch_size: int = 1
    num_prefix_tokens_to_slice: int = 34

    def _to_app_dir(self) -> str:
        return "mixed-qwen-image-disagg"

    def _to_config_dir(self) -> str:
        route_parts = []
        for i, (route, weight) in enumerate(
            zip(self.generator_routes, self.routing_weights)
        ):
            route_parts.append(
                f"r{i}w{_format_float(weight)}sp{route.sp_size}n{route.num_replicas}"
            )
        routes_str = "+".join(route_parts)
        return (
            f"enc{self.num_encoders}+tp{self.encoder_tp_size}+"
            f"encbs{self.encoder_max_num_seqs}+"
            f"gpu{_format_float(self.encoder_gpu_memory_utilization)}+"
            f"{routes_str}"
        )


class MixedMonoQwenImageApp(AppType):
    """Monolithic Qwen-Image with mixed SP sizes and weighted routing.

    RouterApp dispatches requests across QwenImageTextGeneratorTask
    instances with different sp_size.  No separate LLM encoder — each
    mono generator handles text encoding internally.
    """

    mono_routes: list[MixedImageGeneratorRouteConfig]
    routing_weights: list[float]

    def _to_app_dir(self) -> str:
        return "mixed-mono-qwen-image"

    def _to_config_dir(self) -> str:
        route_parts = []
        for i, (route, weight) in enumerate(
            zip(self.mono_routes, self.routing_weights)
        ):
            route_parts.append(
                f"r{i}w{_format_float(weight)}sp{route.sp_size}n{route.num_replicas}"
            )
        return "+".join(route_parts)


class SuperGroupQwenImageConfig(BaseModel):
    """One group within a SuperGroupQwenImageApp."""

    weight: float
    # The inner app for this group (serialized via model_dump_json).
    inner_app_json: str


class SuperGroupQwenImageApp(AppType):
    """SuperGroup: RouterApp over independently-configured sub-groups.

    Each group is a complete Qwen-Image app (mono, disagg, or mixed).
    The top-level RouterApp distributes requests by group weight.
    """

    groups: list[SuperGroupQwenImageConfig]

    def _to_app_dir(self) -> str:
        return "supergroup-qwen-image"

    def _to_config_dir(self) -> str:
        parts = []
        for i, g in enumerate(self.groups):
            inner = json.loads(g.inner_app_json)
            inner_type = inner.get("_app_type", "?")
            parts.append(f"g{i}w{_format_float(g.weight)}{inner_type}")
        return "+".join(parts) if parts else "empty"

    def inner_apps(self) -> list[AppType]:
        """Deserialize and return the inner apps."""
        apps: list[AppType] = []
        for g in self.groups:
            d = json.loads(g.inner_app_json)
            app_type = d.pop("_app_type")
            if app_type == "QwenImageTextGeriApp":
                apps.append(QwenImageTextGeriApp(**d))
            elif app_type == "QwenImageDisaggApp":
                apps.append(QwenImageDisaggApp(**d))
            elif app_type == "MixedQwenImageDisaggApp":
                apps.append(MixedQwenImageDisaggApp.model_validate(d))
            elif app_type == "MixedMonoQwenImageApp":
                apps.append(MixedMonoQwenImageApp.model_validate(d))
            else:
                raise ValueError(f"Unknown inner app type: {app_type}")
        return apps


class ImageRoute(BaseModel):
    """One route in the image generation plan."""

    route_type: Literal["mono", "disagg"] = "disagg"
    weight: float = 1.0
    group_id: int | None = None
    # Mono: QwenImageTextGeriApp
    mono_sp_size: int = 0
    mono_num_replicas: int = 0
    # Disagg: MonolithicLLMApp + DummyImageGeriApp
    llm_tp_size: int = 1
    llm_max_num_seqs: int = 256
    llm_gpu_memory_utilization: float = 0.9
    llm_num_replicas: int = 0
    geri_allocations: dict[int, int] = {}  # {sp_size: n_replicas}
    geri_routing_weights: dict[int, float] = {}  # {sp_size: weight}


class ImageVerifyPayload(BaseModel):
    """Planner output for image generation deployment."""

    resolutions: list[tuple[int, int]] = [(1024, 1024)]
    resolution_weights: list[float] = [1.0]
    routes: list[ImageRoute] = []

    def to_app(
        self,
        model_id: str = "Qwen/Qwen-Image",
        encoder_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    ) -> AppType:
        """Convert to a deployable app.

        Supports:
          - Single mono route → QwenImageTextGeriApp
          - Multi mono routes (mixed SP) → MixedMonoQwenImageApp
          - Single disagg route, uniform SP → QwenImageDisaggApp
          - Single disagg route, mixed SP → MixedQwenImageDisaggApp
        SuperGroup (multi-group with different route_types) is not yet
        supported.
        """
        # Filter to non-group routes (single-route plans)
        routes = [r for r in self.routes if r.group_id is None]
        if not routes:
            routes = self.routes

        # Multi-group → SuperGroup (RouterApp over sub-apps)
        group_ids = {r.group_id for r in routes}
        if len(group_ids) > 1 and group_ids != {None}:
            return self._to_supergroup_app(routes, model_id, encoder_model_id)

        # ── Mono routes ──────────────────────────────────────────
        mono_routes = [r for r in routes if r.route_type == "mono"]
        if mono_routes and len(mono_routes) == len(routes):
            if len(mono_routes) == 1:
                r = mono_routes[0]
                return QwenImageTextGeriApp(
                    model_id=model_id,
                    num_replicas=r.mono_num_replicas,
                    sp_size=r.mono_sp_size,
                    max_batch_size=1,
                )
            # Mixed mono: RouterApp over different SP mono generators
            gen_routes = []
            weights = []
            for r in mono_routes:
                gen_routes.append(MixedImageGeneratorRouteConfig(
                    sp_size=r.mono_sp_size,
                    num_replicas=r.mono_num_replicas,
                ))
                weights.append(r.weight)
            total_w = sum(weights)
            if total_w > 0:
                weights = [w / total_w for w in weights]
            return MixedMonoQwenImageApp(
                model_id=model_id,
                mono_routes=gen_routes,
                routing_weights=weights,
            )

        # ── Disagg route (single) ────────────────────────────────
        disagg_routes = [r for r in routes if r.route_type == "disagg"]
        if len(disagg_routes) == 1:
            r = disagg_routes[0]
            allocs = r.geri_allocations
            if len(allocs) == 1:
                sp, n = next(iter(allocs.items()))
                return QwenImageDisaggApp(
                    model_id=model_id,
                    encoder_model_id=encoder_model_id,
                    num_encoders=r.llm_num_replicas,
                    encoder_tp_size=r.llm_tp_size,
                    encoder_max_num_seqs=r.llm_max_num_seqs,
                    encoder_gpu_memory_utilization=r.llm_gpu_memory_utilization,
                    num_generators=n,
                    generator_sp_size=int(sp),
                )
            # Mixed SP disagg
            gen_routes = []
            weights = []
            for sp_str, n in allocs.items():
                sp = int(sp_str)
                gen_routes.append(
                    MixedImageGeneratorRouteConfig(sp_size=sp, num_replicas=n)
                )
                weights.append(r.geri_routing_weights.get(sp, 1.0))
            total_w = sum(weights)
            if total_w > 0:
                weights = [w / total_w for w in weights]
            return MixedQwenImageDisaggApp(
                model_id=model_id,
                encoder_model_id=encoder_model_id,
                num_encoders=r.llm_num_replicas,
                encoder_tp_size=r.llm_tp_size,
                encoder_max_num_seqs=r.llm_max_num_seqs,
                encoder_gpu_memory_utilization=r.llm_gpu_memory_utilization,
                generator_routes=gen_routes,
                routing_weights=weights,
            )

        raise ValueError(
            f"Unsupported route configuration "
            f"({len(routes)} routes, types={[r.route_type for r in routes]})"
        )

    def _to_supergroup_app(
        self,
        routes: list[ImageRoute],
        model_id: str,
        encoder_model_id: str,
    ) -> SuperGroupQwenImageApp:
        """Build a SuperGroup by converting each group to an inner app."""
        by_group: dict[int, list[ImageRoute]] = {}
        for r in routes:
            gid = r.group_id if r.group_id is not None else 0
            by_group.setdefault(gid, []).append(r)

        groups: list[SuperGroupQwenImageConfig] = []
        for gid in sorted(by_group):
            group_routes = by_group[gid]
            # Sum weights for this group
            group_weight = sum(r.weight for r in group_routes)

            # Build a sub-payload with just this group's routes (clear group_id)
            clean_routes = [
                r.model_copy(update={"group_id": None})
                for r in group_routes
            ]
            sub_payload = ImageVerifyPayload(
                resolutions=self.resolutions,
                resolution_weights=self.resolution_weights,
                routes=clean_routes,
            )
            inner_app = sub_payload.to_app(model_id, encoder_model_id)
            # Serialize with a type tag
            d = inner_app.model_dump()
            d["_app_type"] = type(inner_app).__name__
            groups.append(SuperGroupQwenImageConfig(
                weight=group_weight,
                inner_app_json=json.dumps(d),
            ))

        return SuperGroupQwenImageApp(
            model_id=model_id,
            groups=groups,
        )


class DummyAudioGeriApp(AppType, AudioGeriProfileConfig):
    """Dummy Audio Generator standalone."""

    num_replicas: int

    def _to_app_dir(self) -> str:
        """Return the directory name for the app type."""
        return "geri"

    def _to_config_dir(self) -> str:
        """Return the file name for the backend configuration."""
        subdir_name = f"audio+replicas{self.num_replicas}"
        subdir_name += f"+{self.to_profile_str()}"
        return subdir_name


class DummyTalkerApp(AppType, TalkerProfileConfig):
    """Dummy Omni Talker Embedding."""

    num_replicas: int

    def _to_app_dir(self) -> str:
        """Return the directory name for the app type."""
        return "talker"

    def _to_config_dir(self) -> str:
        """Return the file name for the backend configuration."""
        subdir_name = f"replicas{self.num_replicas}"
        subdir_name += f"+{self.to_profile_str()}"
        return subdir_name


class ModServeApp(AppType):
    """ModServe."""

    num_erics: int
    num_llms: int

    eric_max_batch_size: int
    llm_tp_size: int
    llm_max_num_seqs: int
    llm_gpu_memory_utilization: float

    def _to_app_dir(self) -> str:
        """Return the directory name for the app type."""
        return "modserve"

    def _to_config_dir(self) -> str:
        """Return the file name for the backend configuration."""
        return (
            f"erics{self.num_erics}+llms{self.num_llms}"
            f"+eric_bs{self.eric_max_batch_size}"
            f"+llm_tp{self.llm_tp_size}+llm_bs{self.llm_max_num_seqs}+llm_gpu{self.llm_gpu_memory_utilization}"
        )


class TimeSharingApp(AppType):
    """Time-sharing MLLM (encoder + LLM with probabilistic fission)."""

    num_erics: int
    num_llms: int

    encoder_fission_prob: float
    eric_max_batch_size: int
    llm_tp_size: int
    llm_max_num_seqs: int
    llm_gpu_memory_utilization: float

    def _to_app_dir(self) -> str:
        """Return the directory name for the app type."""
        return "time-sharing-mllm"

    def _to_config_dir(self) -> str:
        """Return the file name for the backend configuration."""
        return (
            f"erics{self.num_erics}+llms{self.num_llms}"
            f"+fission{self.encoder_fission_prob}"
            f"+eric_bs{self.eric_max_batch_size}"
            f"+llm_tp{self.llm_tp_size}+llm_bs{self.llm_max_num_seqs}+llm_gpu{self.llm_gpu_memory_utilization}"
        )


class MixedMLLMRouteConfig(BaseModel):
    """Per-route LLM config for :class:`MixedMLLMApp`."""

    llm_num_replicas: int
    llm_tp_size: int
    llm_max_num_seqs: int
    llm_gpu_memory_utilization: float

    @model_validator(mode="after")
    def _validate_route(self) -> MixedMLLMRouteConfig:
        """Validate route-level replica and profile constraints."""
        if self.llm_num_replicas <= 0:
            raise ValueError("llm_num_replicas must be positive")
        if self.llm_tp_size <= 0:
            raise ValueError("llm_tp_size must be positive")
        if self.llm_max_num_seqs <= 0:
            raise ValueError("llm_max_num_seqs must be positive")
        if not (0 < self.llm_gpu_memory_utilization <= 1):
            raise ValueError("llm_gpu_memory_utilization must be in (0, 1]")
        return self


class MixedMLLMApp(AppType):
    """Mixed MLLM app with a shared encoder pool and weighted LLM routes."""

    num_erics: int
    eric_max_batch_size: int

    routes: list[MixedMLLMRouteConfig]
    routing_weights: list[float]

    @model_validator(mode="after")
    def _validate_app(self) -> MixedMLLMApp:
        """Validate shared-encoder mixed routing config."""
        if self.num_erics <= 0:
            raise ValueError("num_erics must be positive")
        if self.eric_max_batch_size <= 0:
            raise ValueError("eric_max_batch_size must be positive")
        if not self.routes:
            raise ValueError("routes must contain at least one route")
        if len(self.routes) != len(self.routing_weights):
            raise ValueError("routing_weights must have the same length as routes")
        if any(weight <= 0 for weight in self.routing_weights):
            raise ValueError("routing_weights must be positive")
        if sum(self.routing_weights) <= 0:
            raise ValueError("routing_weights must sum to a positive value")
        return self

    def _to_app_dir(self) -> str:
        """Return the directory name for the app type."""
        return "mixed-mllm"

    def _to_config_dir(self) -> str:
        """Return the file name for the backend configuration."""
        route_parts: list[str] = []
        for index, (route, weight) in enumerate(
            zip(self.routes, self.routing_weights, strict=True)
        ):
            route_parts.append(
                (
                    f"r{index}"
                    f"+w{_format_float(weight)}"
                    f"+llms{route.llm_num_replicas}"
                    f"+llm_tp{route.llm_tp_size}"
                    f"+llm_bs{route.llm_max_num_seqs}"
                    f"+llm_gpu{_format_float(route.llm_gpu_memory_utilization)}"
                )
            )

        return (
            f"erics{self.num_erics}+eric_bs{self.eric_max_batch_size}"
            f"+routes{len(self.routes)}+" + "+".join(route_parts)
        )


class GroupedMixedMLLMGroupConfig(BaseModel):
    """One shared-encoder MixedMLLM group."""

    num_erics: int
    eric_max_batch_size: int

    routes: list[MixedMLLMRouteConfig]
    routing_weights: list[float]
    macro_ut_deployment_id: str | None = None

    @model_validator(mode="after")
    def _validate_group(self) -> GroupedMixedMLLMGroupConfig:
        """Validate one group's shared-encoder mixed routing config."""
        if self.num_erics <= 0:
            raise ValueError("num_erics must be positive")
        if self.eric_max_batch_size <= 0:
            raise ValueError("eric_max_batch_size must be positive")
        if not self.routes:
            raise ValueError("routes must contain at least one route")
        if len(self.routes) != len(self.routing_weights):
            raise ValueError("routing_weights must have the same length as routes")
        if any(weight <= 0 for weight in self.routing_weights):
            raise ValueError("routing_weights must be positive")
        if sum(self.routing_weights) <= 0:
            raise ValueError("routing_weights must sum to a positive value")
        return self


class GroupedMixedMLLMApp(AppType):
    """Top-level weighted router over multiple MixedMLLM groups."""

    groups: list[GroupedMixedMLLMGroupConfig]
    group_routing_weights: list[float]

    @model_validator(mode="after")
    def _validate_app(self) -> GroupedMixedMLLMApp:
        """Validate grouped mixed routing config."""
        if not self.groups:
            raise ValueError("groups must contain at least one group")
        if len(self.groups) != len(self.group_routing_weights):
            raise ValueError(
                "group_routing_weights must have the same length as groups"
            )
        if any(weight <= 0 for weight in self.group_routing_weights):
            raise ValueError("group_routing_weights must be positive")
        if sum(self.group_routing_weights) <= 0:
            raise ValueError("group_routing_weights must sum to a positive value")
        return self

    def _to_app_dir(self) -> str:
        """Return the directory name for the app type."""
        return "grouped-mixed-mllm"

    def _to_config_dir(self) -> str:
        """Return the file name for the grouped backend configuration."""
        group_parts: list[str] = []
        for g_idx, (group, g_weight) in enumerate(
            zip(self.groups, self.group_routing_weights, strict=True)
        ):
            route_parts: list[str] = []
            for r_idx, (route, r_weight) in enumerate(
                zip(group.routes, group.routing_weights, strict=True)
            ):
                route_parts.append(
                    (
                        f"r{r_idx}"
                        f"+w{_format_float(r_weight)}"
                        f"+llms{route.llm_num_replicas}"
                        f"+llm_tp{route.llm_tp_size}"
                        f"+llm_bs{route.llm_max_num_seqs}"
                        f"+llm_gpu{_format_float(route.llm_gpu_memory_utilization)}"
                    )
                )
            group_parts.append(
                (
                    f"g{g_idx}"
                    f"+gw{_format_float(g_weight)}"
                    f"+erics{group.num_erics}"
                    f"+eric_bs{group.eric_max_batch_size}"
                    f"+routes{len(group.routes)}"
                    f"+{'+'.join(route_parts)}"
                )
            )
        return f"groups{len(self.groups)}+" + "+".join(group_parts)


class MLLMRouterRouteConfig(BaseModel):
    """Per-route MLLM configuration for :class:`MLLMRouterApp`."""

    route_type: Literal["mllm", "time_sharing"] = "mllm"

    encoder_fission: bool = False
    encoder_fission_prob: float | None = None

    eric_num_replicas: int = 0
    eric_max_batch_size: int = 1

    llm_num_replicas: int
    llm_tp_size: int
    llm_max_num_seqs: int
    llm_gpu_memory_utilization: float

    @model_validator(mode="after")
    def _validate_route(self) -> MLLMRouterRouteConfig:
        """Validate route-level replica and profile constraints."""
        if self.llm_num_replicas <= 0:
            raise ValueError("llm_num_replicas must be positive")
        if self.llm_tp_size <= 0:
            raise ValueError("llm_tp_size must be positive")
        if self.llm_max_num_seqs <= 0:
            raise ValueError("llm_max_num_seqs must be positive")
        if not (0 < self.llm_gpu_memory_utilization <= 1):
            raise ValueError("llm_gpu_memory_utilization must be in (0, 1]")

        if self.eric_num_replicas < 0:
            raise ValueError("eric_num_replicas must be non-negative")
        if self.eric_max_batch_size <= 0:
            raise ValueError("eric_max_batch_size must be positive")

        if self.route_type == "mllm":
            if self.encoder_fission_prob is not None:
                raise ValueError(
                    "encoder_fission_prob is only valid for route_type='time_sharing'"
                )

            if self.encoder_fission:
                if self.eric_num_replicas <= 0:
                    raise ValueError(
                        "eric_num_replicas must be positive when encoder_fission=True"
                    )
            elif self.eric_num_replicas != 0:
                raise ValueError(
                    "eric_num_replicas must be 0 when encoder_fission=False"
                )

        elif self.route_type == "time_sharing":
            if self.encoder_fission_prob is None:
                raise ValueError(
                    "encoder_fission_prob must be set when route_type='time_sharing'"
                )
            if not (0 <= self.encoder_fission_prob <= 1):
                raise ValueError("encoder_fission_prob must be in [0, 1]")
            if self.eric_num_replicas <= 0:
                raise ValueError(
                    "eric_num_replicas must be positive for route_type='time_sharing'"
                )
        else:
            raise ValueError(f"Unsupported route_type: {self.route_type}")

        return self

    def is_time_sharing(self) -> bool:
        """Return whether this route uses time-sharing encoder routing."""
        return self.route_type == "time_sharing"

    def has_encoder_task(self) -> bool:
        """Return whether this route deploys a separate encoder task."""
        return self.is_time_sharing() or self.encoder_fission

    def llm_receive_embeddings(self) -> bool:
        """Return the LLMUnitTask receive_embeddings flag for this route."""
        if self.is_time_sharing():
            return False
        return self.encoder_fission


class MLLMRouterMixedLLMConfig(BaseModel):
    """One LLM backend within a mixed route."""

    llm_num_replicas: int
    llm_tp_size: int
    llm_max_num_seqs: int
    llm_gpu_memory_utilization: float = 0.9

    @model_validator(mode="after")
    def _validate(self) -> MLLMRouterMixedLLMConfig:
        if self.llm_num_replicas <= 0:
            raise ValueError("llm_num_replicas must be positive")
        if self.llm_tp_size <= 0:
            raise ValueError("llm_tp_size must be positive")
        if self.llm_max_num_seqs <= 0:
            raise ValueError("llm_max_num_seqs must be positive")
        if not (0 < self.llm_gpu_memory_utilization <= 1):
            raise ValueError("llm_gpu_memory_utilization must be in (0, 1]")
        return self


class MLLMRouterMixedRouteConfig(BaseModel):
    """A mixed route: shared encoder pool + N weighted LLM backends.

    This is MixedMLLM expressed as a route within :class:`MLLMRouterApp`.
    Each mixed route has its own dedicated encoder pool that is shared
    across its LLM backends, with traffic split by ``llm_routing_weights``.
    """

    eric_num_replicas: int
    eric_max_batch_size: int = 1

    llm_configs: list[MLLMRouterMixedLLMConfig]
    llm_routing_weights: list[float]

    @model_validator(mode="after")
    def _validate_mixed_route(self) -> MLLMRouterMixedRouteConfig:
        if self.eric_num_replicas <= 0:
            raise ValueError("eric_num_replicas must be positive for mixed routes")
        if self.eric_max_batch_size <= 0:
            raise ValueError("eric_max_batch_size must be positive")
        if not self.llm_configs:
            raise ValueError("llm_configs must contain at least one LLM config")
        if len(self.llm_configs) != len(self.llm_routing_weights):
            raise ValueError("llm_routing_weights must match llm_configs length")
        if any(w <= 0 for w in self.llm_routing_weights):
            raise ValueError("llm_routing_weights must be positive")
        return self

    def has_encoder_task(self) -> bool:
        """Mixed routes always have an encoder task."""
        return True


class MLLMRouterApp(AppType):
    """Weighted router app over multiple MLLM routes.

    Routes can be either :class:`MLLMRouterRouteConfig` (single encoder+LLM
    or time-sharing) or :class:`MLLMRouterMixedRouteConfig` (shared encoder
    pool with multiple LLM backends).  This single app type can express all
    deployment topologies: monolithic, disaggregated, mixed, time-sharing,
    and grouped configurations.
    """

    routes: list[MLLMRouterRouteConfig | MLLMRouterMixedRouteConfig]
    routing_weights: list[float]

    @model_validator(mode="after")
    def _validate_router(self) -> MLLMRouterApp:
        """Validate route and weight configuration."""
        if not self.routes:
            raise ValueError("routes must contain at least one route")
        if len(self.routes) != len(self.routing_weights):
            raise ValueError("routing_weights must have the same length as routes")
        if any(weight <= 0 for weight in self.routing_weights):
            raise ValueError("routing_weights must be positive")
        if sum(self.routing_weights) <= 0:
            raise ValueError("routing_weights must sum to a positive value")
        return self

    def _to_app_dir(self) -> str:
        """Return the directory name for the app type."""
        return "mllm-router"

    def _to_config_dir(self) -> str:
        """Return the file name for the backend configuration."""
        route_parts: list[str] = []
        for index, (route, weight) in enumerate(
            zip(self.routes, self.routing_weights, strict=True)
        ):
            if isinstance(route, MLLMRouterMixedRouteConfig):
                sub_parts = []
                for si, (lc, lw) in enumerate(
                    zip(route.llm_configs, route.llm_routing_weights, strict=True)
                ):
                    sub_parts.append(
                        f"s{si}+sw{_format_float(lw)}"
                        f"+llms{lc.llm_num_replicas}"
                        f"+llm_tp{lc.llm_tp_size}"
                        f"+llm_bs{lc.llm_max_num_seqs}"
                        f"+llm_gpu{_format_float(lc.llm_gpu_memory_utilization)}"
                    )
                route_parts.append(
                    f"r{index}"
                    f"+w{_format_float(weight)}"
                    f"+mixed"
                    f"+erics{route.eric_num_replicas}"
                    f"+eric_bs{route.eric_max_batch_size}"
                    f"+{'_'.join(sub_parts)}"
                )
            elif route.is_time_sharing():
                route_parts.append(
                    (
                        f"r{index}"
                        f"+w{_format_float(weight)}"
                        f"+tsprob{_format_float(route.encoder_fission_prob or 0.0)}"
                        f"+erics{route.eric_num_replicas}"
                        f"+eric_bs{route.eric_max_batch_size}"
                        f"+llms{route.llm_num_replicas}"
                        f"+llm_tp{route.llm_tp_size}"
                        f"+llm_bs{route.llm_max_num_seqs}"
                        f"+llm_gpu{_format_float(route.llm_gpu_memory_utilization)}"
                    )
                )
            else:
                route_parts.append(
                    (
                        f"r{index}"
                        f"+w{_format_float(weight)}"
                        f"+fission{int(route.encoder_fission)}"
                        f"+erics{route.eric_num_replicas}"
                        f"+eric_bs{route.eric_max_batch_size}"
                        f"+llms{route.llm_num_replicas}"
                        f"+llm_tp{route.llm_tp_size}"
                        f"+llm_bs{route.llm_max_num_seqs}"
                        f"+llm_gpu{_format_float(route.llm_gpu_memory_utilization)}"
                    )
                )
        return f"routes{len(self.routes)}+" + "+".join(route_parts)


class MLLMRouterVerifyRoute(BaseModel):
    """Serializable route knobs for planner -> verifier handoff."""

    route_type: Literal["mllm", "time_sharing"] = "mllm"
    weight: float

    encoder_fission: bool = False
    encoder_fission_prob: float | None = None

    eric_num_replicas: int = 0
    eric_max_batch_size: int = 1

    llm_num_replicas: int
    llm_tp_size: int
    llm_max_num_seqs: int

    @model_validator(mode="after")
    def _validate_route(self) -> MLLMRouterVerifyRoute:
        """Validate route knobs before serialization."""
        if self.weight <= 0:
            raise ValueError("weight must be positive")

        if self.llm_num_replicas <= 0:
            raise ValueError("llm_num_replicas must be positive")
        if self.llm_tp_size <= 0:
            raise ValueError("llm_tp_size must be positive")
        if self.llm_max_num_seqs <= 0:
            raise ValueError("llm_max_num_seqs must be positive")

        if self.eric_num_replicas < 0:
            raise ValueError("eric_num_replicas must be non-negative")
        if self.eric_max_batch_size <= 0:
            raise ValueError("eric_max_batch_size must be positive")

        if self.route_type == "mllm":
            if self.encoder_fission_prob is not None:
                raise ValueError(
                    "encoder_fission_prob is only valid for route_type='time_sharing'"
                )
            if self.encoder_fission and self.eric_num_replicas <= 0:
                raise ValueError(
                    "eric_num_replicas must be positive when encoder_fission=True"
                )
            if not self.encoder_fission and self.eric_num_replicas != 0:
                raise ValueError(
                    "eric_num_replicas must be 0 when encoder_fission=False"
                )

        elif self.route_type == "time_sharing":
            if self.encoder_fission_prob is None:
                raise ValueError(
                    "encoder_fission_prob must be set when route_type='time_sharing'"
                )
            if not (0 <= self.encoder_fission_prob <= 1):
                raise ValueError("encoder_fission_prob must be in [0, 1]")
            if self.eric_num_replicas <= 0:
                raise ValueError(
                    "eric_num_replicas must be positive for route_type='time_sharing'"
                )

        return self


class MLLMRouterVerifyGroup(BaseModel):
    """Serializable group knobs for grouped mixed planner payloads."""

    weight: float
    routes: list[MLLMRouterVerifyRoute]
    type_routing_weights: dict[str, float] = {}

    @model_validator(mode="after")
    def _validate_group(self) -> MLLMRouterVerifyGroup:
        """Validate grouped mixed route constraints."""
        if self.weight <= 0:
            raise ValueError("group weight must be positive")
        if not self.routes:
            raise ValueError("group routes must be non-empty")
        if sum(route.weight for route in self.routes) <= 0:
            raise ValueError("group routes must have a positive total weight")
        if any(route.route_type != "mllm" for route in self.routes):
            raise ValueError("group routes only support route_type='mllm'")

        # Mono group: single route with encoder_fission=False
        is_mono = (
            len(self.routes) == 1
            and not self.routes[0].encoder_fission
            and self.routes[0].eric_num_replicas == 0
        )
        if not is_mono:
            # Disagg group: all routes share encoders
            if any(not route.encoder_fission for route in self.routes):
                raise ValueError("disagg group routes require encoder_fission=True")
            eric_counts = {route.eric_num_replicas for route in self.routes}
            eric_batch_sizes = {route.eric_max_batch_size for route in self.routes}
            if len(eric_counts) != 1:
                raise ValueError("group routes require a single shared eric_num_replicas")
            if len(eric_batch_sizes) != 1:
                raise ValueError("group routes require a single shared eric_max_batch_size")
            shared_erics = next(iter(eric_counts))
            if shared_erics <= 0:
                raise ValueError("group routes require eric_num_replicas > 0")

        return self


class MLLMRouterVerifyPayload(BaseModel):
    """Structured planner output consumed by verification scripts."""

    schema_version: Literal[1] = 1
    app_type: Literal["MLLMRouterApp", "MixedMLLMApp", "GroupedMixedMLLMApp"] = (
        "MLLMRouterApp"
    )

    verify_supported: bool = True
    unsupported_reason: str | None = None
    routes: list[MLLMRouterVerifyRoute] = []
    groups: list[MLLMRouterVerifyGroup] = []

    @model_validator(mode="after")
    def _validate_payload(self) -> MLLMRouterVerifyPayload:
        """Validate support flags and route list shape."""
        if self.verify_supported:
            if self.unsupported_reason is not None:
                raise ValueError(
                    "unsupported_reason must be None when verify_supported=True"
                )
            if self.app_type == "GroupedMixedMLLMApp":
                if self.routes:
                    raise ValueError(
                        "routes must be empty when app_type='GroupedMixedMLLMApp'"
                    )
                if not self.groups:
                    raise ValueError(
                        "groups must be non-empty when app_type='GroupedMixedMLLMApp'"
                    )
                if sum(group.weight for group in self.groups) <= 0:
                    raise ValueError("groups must have a positive total weight")
            else:
                if self.groups:
                    raise ValueError(
                        "groups must be empty unless app_type='GroupedMixedMLLMApp'"
                    )
                if not self.routes:
                    raise ValueError(
                        "routes must be non-empty when verify_supported=True"
                    )
                if sum(route.weight for route in self.routes) <= 0:
                    raise ValueError("routes must have a positive total weight")

                if self.app_type == "MixedMLLMApp":
                    if any(route.route_type != "mllm" for route in self.routes):
                        raise ValueError(
                            "MixedMLLMApp payload only supports route_type='mllm'"
                        )
                    if any(not route.encoder_fission for route in self.routes):
                        raise ValueError(
                            "MixedMLLMApp payload requires encoder_fission=True for all routes"
                        )

                    eric_counts = {route.eric_num_replicas for route in self.routes}
                    eric_batch_sizes = {
                        route.eric_max_batch_size for route in self.routes
                    }
                    if len(eric_counts) != 1:
                        raise ValueError(
                            "MixedMLLMApp payload requires a single shared eric_num_replicas"
                        )
                    if len(eric_batch_sizes) != 1:
                        raise ValueError(
                            "MixedMLLMApp payload requires a single shared eric_max_batch_size"
                        )

                    shared_erics = next(iter(eric_counts))
                    if shared_erics <= 0:
                        raise ValueError(
                            "MixedMLLMApp payload requires eric_num_replicas > 0"
                        )
        else:
            if not self.unsupported_reason:
                raise ValueError(
                    "unsupported_reason must be set when verify_supported=False"
                )
            if self.routes:
                raise ValueError("routes must be empty when verify_supported=False")
            if self.groups:
                raise ValueError("groups must be empty when verify_supported=False")

        return self

    @classmethod
    def unsupported(cls, reason: str) -> MLLMRouterVerifyPayload:
        """Create an unsupported payload with a reason."""
        return cls(verify_supported=False, unsupported_reason=reason, routes=[])

    def to_router_app(
        self,
        *,
        model_id: str,
        llm_gpu_memory_utilization: float,
    ) -> MLLMRouterApp:
        """Convert verified knobs to an :class:`MLLMRouterApp`."""
        if self.app_type != "MLLMRouterApp":
            raise ValueError(f"Payload app_type is not MLLMRouterApp: {self.app_type}")
        if not self.verify_supported:
            raise ValueError(
                "Cannot convert unsupported verify payload to MLLMRouterApp: "
                f"{self.unsupported_reason}"
            )

        total_weight = sum(route.weight for route in self.routes)
        if total_weight <= 0:
            raise ValueError("routes must have positive total weight")

        routes = [
            MLLMRouterRouteConfig(
                route_type=route.route_type,
                encoder_fission=route.encoder_fission,
                encoder_fission_prob=route.encoder_fission_prob,
                eric_num_replicas=route.eric_num_replicas,
                eric_max_batch_size=route.eric_max_batch_size,
                llm_num_replicas=route.llm_num_replicas,
                llm_tp_size=route.llm_tp_size,
                llm_max_num_seqs=route.llm_max_num_seqs,
                llm_gpu_memory_utilization=llm_gpu_memory_utilization,
            )
            for route in self.routes
        ]
        weights = [route.weight / total_weight for route in self.routes]
        return MLLMRouterApp(model_id=model_id, routes=routes, routing_weights=weights)

    def to_mixed_app(
        self,
        *,
        model_id: str,
        llm_gpu_memory_utilization: float,
    ) -> MixedMLLMApp:
        """Convert verified knobs to a :class:`MixedMLLMApp`."""
        if self.app_type != "MixedMLLMApp":
            raise ValueError(f"Payload app_type is not MixedMLLMApp: {self.app_type}")
        if not self.verify_supported:
            raise ValueError(
                "Cannot convert unsupported verify payload to MixedMLLMApp: "
                f"{self.unsupported_reason}"
            )

        total_weight = sum(route.weight for route in self.routes)
        if total_weight <= 0:
            raise ValueError("routes must have positive total weight")

        num_erics = self.routes[0].eric_num_replicas
        eric_max_batch_size = self.routes[0].eric_max_batch_size
        routes = [
            MixedMLLMRouteConfig(
                llm_num_replicas=route.llm_num_replicas,
                llm_tp_size=route.llm_tp_size,
                llm_max_num_seqs=route.llm_max_num_seqs,
                llm_gpu_memory_utilization=llm_gpu_memory_utilization,
            )
            for route in self.routes
        ]
        weights = [route.weight / total_weight for route in self.routes]
        return MixedMLLMApp(
            model_id=model_id,
            num_erics=num_erics,
            eric_max_batch_size=eric_max_batch_size,
            routes=routes,
            routing_weights=weights,
        )

    def to_grouped_mixed_app(
        self,
        *,
        model_id: str,
        llm_gpu_memory_utilization: float,
    ) -> GroupedMixedMLLMApp:
        """Convert verified knobs to a :class:`GroupedMixedMLLMApp`."""
        if self.app_type != "GroupedMixedMLLMApp":
            raise ValueError(
                f"Payload app_type is not GroupedMixedMLLMApp: {self.app_type}"
            )
        if not self.verify_supported:
            raise ValueError(
                "Cannot convert unsupported verify payload to GroupedMixedMLLMApp: "
                f"{self.unsupported_reason}"
            )

        total_group_weight = sum(group.weight for group in self.groups)
        if total_group_weight <= 0:
            raise ValueError("groups must have positive total weight")

        groups: list[GroupedMixedMLLMGroupConfig] = []
        for index, group in enumerate(self.groups):
            total_route_weight = sum(route.weight for route in group.routes)
            if total_route_weight <= 0:
                raise ValueError("group routes must have positive total weight")

            num_erics = group.routes[0].eric_num_replicas
            eric_max_batch_size = group.routes[0].eric_max_batch_size
            routes = [
                MixedMLLMRouteConfig(
                    llm_num_replicas=route.llm_num_replicas,
                    llm_tp_size=route.llm_tp_size,
                    llm_max_num_seqs=route.llm_max_num_seqs,
                    llm_gpu_memory_utilization=llm_gpu_memory_utilization,
                )
                for route in group.routes
            ]
            weights = [route.weight / total_route_weight for route in group.routes]
            groups.append(
                GroupedMixedMLLMGroupConfig(
                    num_erics=num_erics,
                    eric_max_batch_size=eric_max_batch_size,
                    routes=routes,
                    routing_weights=weights,
                    macro_ut_deployment_id=f"group-{index}",
                )
            )

        group_weights = [group.weight / total_group_weight for group in self.groups]
        return GroupedMixedMLLMApp(
            model_id=model_id,
            groups=groups,
            group_routing_weights=group_weights,
        )

    def to_app(
        self,
        *,
        model_id: str,
        llm_gpu_memory_utilization: float,
    ) -> MLLMRouterApp | MixedMLLMApp | GroupedMixedMLLMApp:
        """Convert verified knobs to the benchmark app selected by ``app_type``."""
        if self.app_type == "MLLMRouterApp":
            return self.to_router_app(
                model_id=model_id,
                llm_gpu_memory_utilization=llm_gpu_memory_utilization,
            )
        if self.app_type == "MixedMLLMApp":
            return self.to_mixed_app(
                model_id=model_id,
                llm_gpu_memory_utilization=llm_gpu_memory_utilization,
            )
        return self.to_grouped_mixed_app(
            model_id=model_id,
            llm_gpu_memory_utilization=llm_gpu_memory_utilization,
        )


class OmniRouterRouteConfig(BaseModel):
    """Per-route config for OmniRouterApp."""

    route_type: Literal["omni_mllm", "omni_time_sharing"]

    # Encoder config
    encoder_fission: bool = True  # omni_mllm only
    img_eric_num_replicas: int = 0
    vid_eric_num_replicas: int = 0
    audio_eric_num_replicas: int = 0
    eric_max_batch_size: int = 1

    # Time-sharing specific (omni_time_sharing only)
    offloaded_modalities: list[str] = []
    type_routing_weights: dict[int, dict[str, float]] = {}

    # LLM (thinker) config
    llm_num_replicas: int = 1
    llm_tp_size: int = 1
    llm_max_num_seqs: int = 32
    llm_gpu_memory_utilization: float = 0.9

    # Audio output config
    vocoder_fission: bool = False
    talker_vocoder_num_replicas: int = 0  # for vocoder_fission=False
    talker_num_replicas: int = 0          # for vocoder_fission=True
    audio_geri_num_replicas: int = 0      # for vocoder_fission=True
    talker_max_num_seqs: int = 256        # TalkerProfileConfig default
    audio_geri_max_batch_size: int = 1    # AudioGeriProfileConfig default

    def total_eric_replicas(self) -> int:
        return (
            self.img_eric_num_replicas
            + self.vid_eric_num_replicas
            + self.audio_eric_num_replicas
        )


class OmniRouterApp(AppType):
    """Router across Omni deployment topologies (mono, full disagg, partial disagg)."""

    routes: list[OmniRouterRouteConfig]
    routing_weights: list[float]

    def _to_app_dir(self) -> str:
        return "omni-router"

    def _to_config_dir(self) -> str:
        parts = []
        for i, route in enumerate(self.routes):
            w = self.routing_weights[i]
            rt = route.route_type.replace("omni_", "")
            enc = (
                f"ei{route.img_eric_num_replicas}"
                f"_ev{route.vid_eric_num_replicas}"
                f"_ea{route.audio_eric_num_replicas}"
            )
            llm = f"tp{route.llm_tp_size}x{route.llm_num_replicas}bs{route.llm_max_num_seqs}"
            # Audio output: either fused TalkerVocoder or split Talker + AudioGeri
            if route.vocoder_fission:
                ao = f"tk{route.talker_num_replicas}_ag{route.audio_geri_num_replicas}"
            else:
                ao = f"tv{route.talker_vocoder_num_replicas}"
            if route.talker_max_num_seqs != 256:
                ao += f"_tkbs{route.talker_max_num_seqs}"
            if route.vocoder_fission and route.audio_geri_max_batch_size != 1:
                ao += f"_agbs{route.audio_geri_max_batch_size}"
            parts.append(f"r{i}_{rt}_{enc}_{llm}_{ao}_w{_format_float(w)}")
        return "+".join(parts)


class OmniRouterVerifyRoute(BaseModel):
    """Serializable route config for planner → deployment conversion."""

    route_type: Literal["omni_mllm", "omni_time_sharing"]
    weight: float = 1.0

    encoder_fission: bool = True
    img_eric_num_replicas: int = 0
    vid_eric_num_replicas: int = 0
    audio_eric_num_replicas: int = 0
    eric_max_batch_size: int = 1

    offloaded_modalities: list[str] = []
    type_routing_weights: dict[int, dict[str, float]] = {}

    llm_num_replicas: int = 1
    llm_tp_size: int = 1
    llm_max_num_seqs: int = 32
    llm_gpu_memory_utilization: float = 0.9

    vocoder_fission: bool = False
    talker_vocoder_num_replicas: int = 0
    talker_num_replicas: int = 0
    audio_geri_num_replicas: int = 0
    talker_max_num_seqs: int = 256
    audio_geri_max_batch_size: int = 1

    # Optional group ID to prevent merging routes with same encoder signature.
    # Routes with different group_id are never merged into the same group.
    # When None, uses signature-based merging (legacy behavior).
    group_id: int | None = None


class OmniFlexThinkerConfig(BaseModel):
    """One thinker LLM variant within a group (benchmark schema)."""

    llm_num_replicas: int
    llm_tp_size: int
    llm_max_num_seqs: int
    llm_gpu_memory_utilization: float = 0.9
    weight: float = 1.0


class OmniFlexGroupSchema(BaseModel):
    """One encoder+thinker group (benchmark schema)."""

    offloaded_modalities: list[str] = []
    eric_max_batch_sizes: dict[str, int] = {}
    img_eric_num_replicas: int = 0
    vid_eric_num_replicas: int = 0
    audio_eric_num_replicas: int = 0
    thinkers: list[OmniFlexThinkerConfig]


class OmniFlexApp(AppType):
    """Flexible Omni deployment: per-group encoders+thinkers, shared audio output."""

    groups: list[OmniFlexGroupSchema]
    type_routing_weights: dict[int, dict[int, float]] = {}

    vocoder_fission: bool = False
    talker_vocoder_num_replicas: int = 0
    talker_num_replicas: int = 0
    audio_geri_num_replicas: int = 0
    talker_max_num_seqs: int = 256
    audio_geri_max_batch_size: int = 1

    def _to_app_dir(self) -> str:
        return "omni-flex"

    def _to_config_dir(self) -> str:
        parts = []
        for i, group in enumerate(self.groups):
            mods = "+".join(sorted(group.offloaded_modalities)) or "mono"
            enc_bs = "_".join(
                f"{m}{bs}"
                for m, bs in sorted(group.eric_max_batch_sizes.items())
            )
            enc_reps = (
                f"ei{group.img_eric_num_replicas}"
                f"_ev{group.vid_eric_num_replicas}"
                f"_ea{group.audio_eric_num_replicas}"
            )
            thinker_parts = []
            for t in group.thinkers:
                thinker_parts.append(
                    f"tp{t.llm_tp_size}x{t.llm_num_replicas}"
                    f"bs{t.llm_max_num_seqs}w{_format_float(t.weight)}"
                )
            parts.append(
                f"g{i}_{mods}_{enc_bs}_{enc_reps}_{'_'.join(thinker_parts)}"
            )
        ao = (
            f"tk{self.talker_num_replicas}_ag{self.audio_geri_num_replicas}"
            if self.vocoder_fission
            else f"tv{self.talker_vocoder_num_replicas}"
        )
        if self.talker_max_num_seqs != 256:
            ao += f"_tkbs{self.talker_max_num_seqs}"
        if self.vocoder_fission and self.audio_geri_max_batch_size != 1:
            ao += f"_agbs{self.audio_geri_max_batch_size}"
        return "+".join(parts) + f"_{ao}"


class OmniRouterVerifyPayload(BaseModel):
    """Planner output payload that can be converted to OmniRouterApp or OmniFlexApp."""

    verify_supported: bool = True
    unsupported_reason: str | None = None
    routes: list[OmniRouterVerifyRoute] = []

    def to_app(self, model_id: str) -> OmniRouterApp:
        """Convert to deployable OmniRouterApp."""
        route_configs = []
        weights = []
        for route in self.routes:
            route_configs.append(
                OmniRouterRouteConfig(
                    route_type=route.route_type,
                    encoder_fission=route.encoder_fission,
                    img_eric_num_replicas=route.img_eric_num_replicas,
                    vid_eric_num_replicas=route.vid_eric_num_replicas,
                    audio_eric_num_replicas=route.audio_eric_num_replicas,
                    eric_max_batch_size=route.eric_max_batch_size,
                    offloaded_modalities=route.offloaded_modalities,
                    type_routing_weights=route.type_routing_weights,
                    llm_num_replicas=route.llm_num_replicas,
                    llm_tp_size=route.llm_tp_size,
                    llm_max_num_seqs=route.llm_max_num_seqs,
                    llm_gpu_memory_utilization=route.llm_gpu_memory_utilization,
                    vocoder_fission=route.vocoder_fission,
                    talker_vocoder_num_replicas=route.talker_vocoder_num_replicas,
                    talker_num_replicas=route.talker_num_replicas,
                    audio_geri_num_replicas=route.audio_geri_num_replicas,
                    talker_max_num_seqs=route.talker_max_num_seqs,
                    audio_geri_max_batch_size=route.audio_geri_max_batch_size,
                )
            )
            weights.append(route.weight)
        return OmniRouterApp(
            model_id=model_id,
            routes=route_configs,
            routing_weights=weights,
        )

    def to_flex_app(self, model_id: str) -> OmniFlexApp:
        """Convert routes to a single OmniFlexApp with shared audio output.

        Routes with the same encoder signature (offloaded_modalities, batch size,
        encoder_fission) are merged into one group with multiple thinker configs.
        Routes with different encoder signatures become separate groups.
        """
        # Group routes by encoder signature.  When group_id is set, it is
        # included in the key so routes from independent pools are never merged.
        sig_to_routes: dict[
            tuple[tuple[str, ...], int, bool, int | None], list[OmniRouterVerifyRoute]
        ] = {}
        for route in self.routes:
            sig = (
                tuple(sorted(route.offloaded_modalities)),
                route.eric_max_batch_size,
                route.encoder_fission,
                route.group_id,
            )
            sig_to_routes.setdefault(sig, []).append(route)

        groups: list[OmniFlexGroupSchema] = []
        # Map from route → group index for type_routing_weights conversion
        route_to_group: dict[int, int] = {}

        for sig, sig_routes in sig_to_routes.items():
            offloaded_mods, enc_bs, encoder_fission, _gid = sig
            group_idx = len(groups)

            # All routes in this sig share encoder config
            first = sig_routes[0]
            # Determine offloaded modalities for group
            if encoder_fission and not offloaded_mods:
                # omni_mllm with encoder_fission=True → full disagg
                group_offloaded = ["audio", "img", "vid"]
            else:
                group_offloaded = list(offloaded_mods)

            eric_max_batch_sizes = {m: enc_bs for m in group_offloaded}

            thinkers = []
            for route in sig_routes:
                # Record mapping for this route
                route_idx = self.routes.index(route)
                route_to_group[route_idx] = group_idx

                thinkers.append(
                    OmniFlexThinkerConfig(
                        llm_num_replicas=route.llm_num_replicas,
                        llm_tp_size=route.llm_tp_size,
                        llm_max_num_seqs=route.llm_max_num_seqs,
                        llm_gpu_memory_utilization=route.llm_gpu_memory_utilization,
                        weight=route.weight,
                    )
                )

            groups.append(
                OmniFlexGroupSchema(
                    offloaded_modalities=group_offloaded,
                    eric_max_batch_sizes=eric_max_batch_sizes,
                    img_eric_num_replicas=first.img_eric_num_replicas,
                    vid_eric_num_replicas=first.vid_eric_num_replicas,
                    audio_eric_num_replicas=first.audio_eric_num_replicas,
                    thinkers=thinkers,
                )
            )

        # Convert type_routing_weights from per-route mode-key format
        # to per-group index format.
        # Each route's type_routing_weights maps type_idx → {mode: weight}.
        # Use the actual per-type weight (sum of mode weights) rather than
        # the route's aggregate weight, so per-type fractional routing is preserved.
        flex_type_weights: dict[int, dict[int, float]] = {}
        for ri, route in enumerate(self.routes):
            gidx = route_to_group[ri]
            for type_idx, mode_weights in route.type_routing_weights.items():
                if type_idx not in flex_type_weights:
                    flex_type_weights[type_idx] = {}
                # Use the per-type weight (sum of mode fractions for this route)
                per_type_w = sum(mode_weights.values()) if mode_weights else route.weight
                flex_type_weights[type_idx][gidx] = (
                    flex_type_weights[type_idx].get(gidx, 0.0) + per_type_w
                )

        # For routes without explicit type_routing_weights (e.g., omni_mllm),
        # add all active types pointing to the route's group with its weight
        for ri, route in enumerate(self.routes):
            if not route.type_routing_weights:
                gidx = route_to_group[ri]
                # Add to all existing type entries or create for all types 0-15
                for t in range(16):
                    if t not in flex_type_weights:
                        flex_type_weights[t] = {}
                    flex_type_weights[t][gidx] = (
                        flex_type_weights[t].get(gidx, 0.0) + route.weight
                    )

        # Audio output: take max across routes (shared pool)
        first_route = self.routes[0]
        vocoder_fission = first_route.vocoder_fission
        tv_reps = max(r.talker_vocoder_num_replicas for r in self.routes)
        tk_reps = max(r.talker_num_replicas for r in self.routes)
        ag_reps = max(r.audio_geri_num_replicas for r in self.routes)
        tk_bs = max(r.talker_max_num_seqs for r in self.routes)
        ag_bs = max(r.audio_geri_max_batch_size for r in self.routes)

        return OmniFlexApp(
            model_id=model_id,
            groups=groups,
            type_routing_weights=flex_type_weights,
            vocoder_fission=vocoder_fission,
            talker_vocoder_num_replicas=tv_reps,
            talker_num_replicas=tk_reps,
            audio_geri_num_replicas=ag_reps,
            talker_max_num_seqs=tk_bs,
            audio_geri_max_batch_size=ag_bs,
        )


class DummyTalkerVocoderApp(AppType, TalkerProfileConfig):
    """Dummy Omni Talker Vocoder."""

    num_replicas: int

    def _to_app_dir(self) -> str:
        """Return the directory name for the app type."""
        return "talker-vocoder"

    def _to_config_dir(self) -> str:
        """Return the file name for the backend configuration."""
        subdir_name = f"replicas{self.num_replicas}"
        subdir_name += f"+{self.to_profile_str()}"
        return subdir_name


class Experiment(BaseModel):
    """Experiment configuration containing the app and profile information."""

    app: SerializeAsAny[AppType]
    dataset: Dataset  # ControlledImageChat | ControlledDiT | ServeGenDataset
    gpu_type: Literal["A40", "A100", "H100"]
    seed: int = 48105
    request_rate: float = float("inf")
    max_output_len: int | None = None  # clamp output_len on sampled requests

    def _to_dirname(self) -> str:
        """Convert the experiment config to a directory name."""
        dirname = f"{self.gpu_type}+{self.dataset.to_suffix()}"
        # Append rate unless the dataset already encodes it.
        if not isinstance(self.dataset, (ServeGenDataset, ReplayedServegen, ReplayedOmniServegen)) and self.request_rate != float(
            "inf"
        ):
            dirname += f"+rate{self.request_rate}"
        if self.max_output_len is not None:
            dirname += f"+maxout{self.max_output_len}"
        return dirname

    def to_dir(self) -> Path:
        """Return the experiment directory for this experiment."""
        return self.app.to_dir() / self._to_dirname()

    def to_path(self) -> Path:
        """Return the path where benchmark results should be saved."""
        return self.to_dir() / "result.json"

    def exists(self) -> bool:
        """Check if the benchmark results file exists."""
        return self.to_path().exists()

    def save(self, data: dict[str, Any]) -> None:
        """Save the benchmark data to a file."""
        config = self.model_dump()
        config["timestamp"] = datetime.datetime.now().isoformat()
        data["config"] = config
        path = self.to_path()
        if path.exists():
            print(f"Warning: Data file {path} already exists. Overwriting it.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved benchmark results to: {path}")

    def load(self) -> dict[str, Any]:
        """Load the benchmark data."""
        path = self.to_path()
        if not path.exists():
            raise FileNotFoundError(f"Benchmark results file {path} does not exist.")
        with path.open("r") as f:
            return json.load(f)
