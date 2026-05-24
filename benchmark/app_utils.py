"""Utility functions to create and load cornserve app source code."""

import importlib.util
from pathlib import Path
from string import Template
from types import ModuleType

from cornserve.task.base import UnitTask, discover_unit_tasks

from schema import (
    AppType,
    DecodeLLMApp,
    DummyAudioGeriApp,
    DummyEricApp,
    DummyImageGeriApp,
    DummyLLMApp,
    DummyTalkerApp,
    DummyTalkerVocoderApp,
    EPDMLLMApp,
    GroupedMixedMLLMApp,
    MixedMLLMApp,
    MLLMRouterApp,
    MLLMRouterMixedRouteConfig,
    MLLMRouterRouteConfig,
    ModServeApp,
    MonolithicLLMApp,
    OmniFlexApp,
    OmniMLLMApp,
    OmniRouterApp,
    MixedMonoQwenImageApp,
    MixedQwenImageDisaggApp,
    QwenImageDisaggApp,
    SuperGroupQwenImageApp,
    QwenImageTextGeriApp,
    PDMLLMApp,
    PrefillLLMApp,
    TimeSharingApp,
)

# Template paths relative to benchmark directory
BENCHMARK_ROOT = Path(__file__).parent.parent / "benchmark"
DUMMY_ERIC_TEMPLATE_PATH = BENCHMARK_ROOT / "apps" / "eric.py.tmpl"
DUMMY_LLM_TEMPLATE_PATH = BENCHMARK_ROOT / "apps" / "dummy_llm.py.tmpl"
MONOLITHIC_LLM_TEMPLATE_PATH = BENCHMARK_ROOT / "apps" / "monolithic_llm.py.tmpl"
DUMMY_IMAGE_GERI_TEMPLATE_PATH = BENCHMARK_ROOT / "apps" / "dummy_image_geri.py.tmpl"
DUMMY_AUDIO_GERI_TEMPLATE_PATH = BENCHMARK_ROOT / "apps" / "dummy_audio_geri.py.tmpl"
QWEN_IMAGE_TEXT_GERI_TEMPLATE_PATH = BENCHMARK_ROOT / "apps" / "qwen_image_text_geri.py.tmpl"
QWEN_IMAGE_DISAGG_TEMPLATE_PATH = BENCHMARK_ROOT / "apps" / "qwen_image_disagg.py.tmpl"
MIXED_QWEN_IMAGE_DISAGG_TEMPLATE_PATH = (
    BENCHMARK_ROOT / "apps" / "mixed_qwen_image_disagg.py.tmpl"
)
MIXED_MONO_QWEN_IMAGE_TEMPLATE_PATH = (
    BENCHMARK_ROOT / "apps" / "mixed_mono_qwen_image.py.tmpl"
)
SUPERGROUP_QWEN_IMAGE_TEMPLATE_PATH = (
    BENCHMARK_ROOT / "apps" / "supergroup_qwen_image.py.tmpl"
)
DUMMY_TALKER_TEMPLATE_PATH = BENCHMARK_ROOT / "apps" / "dummy_talker.py.tmpl"
DUMMY_TALKER_VOCODER_TEMPLATE_PATH = (
    BENCHMARK_ROOT / "apps" / "dummy_talker_vocoder.py.tmpl"
)
MLLM_TEMPLATE_PATH = BENCHMARK_ROOT / "apps" / "mllm.py.tmpl"
TIME_SHARING_MLLM_TEMPLATE_PATH = BENCHMARK_ROOT / "apps" / "time_sharing_mllm.py.tmpl"
MIXED_MLLM_TEMPLATE_PATH = BENCHMARK_ROOT / "apps" / "mixed_mllm.py.tmpl"
GROUPED_MIXED_MLLM_TEMPLATE_PATH = (
    BENCHMARK_ROOT / "apps" / "grouped_mixed_mllm.py.tmpl"
)
OMNI_MLLM_TEMPLATE_PATH = BENCHMARK_ROOT / "apps" / "omni_mllm.py.tmpl"
OMNI_ROUTER_TEMPLATE_PATH = BENCHMARK_ROOT / "apps" / "omni_router.py.tmpl"
OMNI_FLEX_TEMPLATE_PATH = BENCHMARK_ROOT / "apps" / "omni_flex.py.tmpl"
MLLM_ROUTER_TEMPLATE_PATH = BENCHMARK_ROOT / "apps" / "mllm_router.py.tmpl"
PREFILL_LLM_TEMPLATE_PATH = BENCHMARK_ROOT / "apps" / "prefill_llm.py.tmpl"
DECODE_LLM_TEMPLATE_PATH = BENCHMARK_ROOT / "apps" / "decode_llm.py.tmpl"
PD_MLLM_TEMPLATE_PATH = BENCHMARK_ROOT / "apps" / "pd_mllm.py.tmpl"
EPD_MLLM_TEMPLATE_PATH = BENCHMARK_ROOT / "apps" / "epd_mllm.py.tmpl"


def _format_float(value: float) -> str:
    """Format a float for Python source/template substitution."""
    formatted = f"{value:.6f}".rstrip("0").rstrip(".")
    return formatted if formatted else "0"


def create_app_source(app: AppType) -> str:
    """Create app source code from template based on app type.

    Args:
        app: The app configuration

    Returns:
        Generated source code as a string
    """
    if isinstance(app, DummyEricApp):
        src = DUMMY_ERIC_TEMPLATE_PATH.read_text()
        rendered = Template(src).substitute(
            MODEL_ID=app.model_id,
            MODALITY=app.modality.upper(),
            MAX_BATCH_SIZE=app.max_batch_size,
            TP_SIZE=app.tp_size,
        )
    elif isinstance(app, DummyLLMApp):
        src = DUMMY_LLM_TEMPLATE_PATH.read_text()
        rendered = Template(src).substitute(
            MODEL_ID=app.model_id,
            TP_SIZE=app.tp_size,
            MAX_NUM_SEQS=app.max_num_seqs,
            GPU_MEMORY_UTILIZATION=app.gpu_memory_utilization,
        )
    elif isinstance(app, MonolithicLLMApp):
        src = MONOLITHIC_LLM_TEMPLATE_PATH.read_text()
        rendered = Template(src).substitute(
            MODEL_ID=app.model_id,
            TP_SIZE=app.tp_size,
            MAX_NUM_SEQS=app.max_num_seqs,
            GPU_MEMORY_UTILIZATION=app.gpu_memory_utilization,
        )
    elif isinstance(app, PrefillLLMApp):
        src = PREFILL_LLM_TEMPLATE_PATH.read_text()
        rendered = Template(src).substitute(
            MODEL_ID=app.model_id,
            RECEIVE_EMBEDDINGS=app.receive_embeddings,
            TP_SIZE=app.tp_size,
            MAX_NUM_SEQS=app.max_num_seqs,
            GPU_MEMORY_UTILIZATION=_format_float(app.gpu_memory_utilization),
        )
    elif isinstance(app, DecodeLLMApp):
        src = DECODE_LLM_TEMPLATE_PATH.read_text()
        rendered = Template(src).substitute(
            MODEL_ID=app.model_id,
            RECEIVE_EMBEDDINGS=app.receive_embeddings,
            TP_SIZE=app.tp_size,
            MAX_NUM_SEQS=app.max_num_seqs,
            GPU_MEMORY_UTILIZATION=_format_float(app.gpu_memory_utilization),
        )
    elif isinstance(app, EPDMLLMApp):
        # HACK: Qwen3-VL OOMs at 0.9 gpu_memory_utilization on prefill
        prefill_gpu_mem = app.prefill_gpu_memory_utilization
        if "Qwen3-VL" in app.model_id:
            prefill_gpu_mem = min(prefill_gpu_mem, 0.85)
        src = EPD_MLLM_TEMPLATE_PATH.read_text()
        rendered = Template(src).substitute(
            MODEL_ID=app.model_id,
            ERIC_MAX_BATCH_SIZE=app.eric_max_batch_size,
            PREFILL_TP_SIZE=app.prefill_tp_size,
            PREFILL_MAX_NUM_SEQS=app.prefill_max_num_seqs,
            PREFILL_GPU_MEMORY_UTILIZATION=_format_float(prefill_gpu_mem),
            DECODE_TP_SIZE=app.decode_tp_size,
            DECODE_MAX_NUM_SEQS=app.decode_max_num_seqs,
            DECODE_GPU_MEMORY_UTILIZATION=_format_float(app.decode_gpu_memory_utilization),
        )
    elif isinstance(app, PDMLLMApp):
        src = PD_MLLM_TEMPLATE_PATH.read_text()
        rendered = Template(src).substitute(
            MODEL_ID=app.model_id,
            PREFILL_TP_SIZE=app.prefill_tp_size,
            PREFILL_MAX_NUM_SEQS=app.prefill_max_num_seqs,
            PREFILL_GPU_MEMORY_UTILIZATION=_format_float(app.prefill_gpu_memory_utilization),
            DECODE_TP_SIZE=app.decode_tp_size,
            DECODE_MAX_NUM_SEQS=app.decode_max_num_seqs,
            DECODE_GPU_MEMORY_UTILIZATION=_format_float(app.decode_gpu_memory_utilization),
        )
    elif isinstance(app, OmniMLLMApp):
        src = OMNI_MLLM_TEMPLATE_PATH.read_text()
        rendered = Template(src).substitute(
            MODEL_ID=app.model_id,
            TP_SIZE=app.tp_size,
            MAX_NUM_SEQS=app.max_num_seqs,
            GPU_MEMORY_UTILIZATION=app.gpu_memory_utilization,
            DISABLE_AUDIO_ENC=app.disable_audio_enc,
            DISABLE_IMAGE_ENC=app.disable_image_enc,
            DISABLE_VIDEO_ENC=app.disable_video_enc,
        )
    elif isinstance(app, DummyImageGeriApp):
        src = DUMMY_IMAGE_GERI_TEMPLATE_PATH.read_text()
        rendered = Template(src).substitute(
            MODEL_ID=app.model_id,
            MAX_BATCH_SIZE=app.max_batch_size,
            SP_SIZE=app.sp_size,
        )
    elif isinstance(app, QwenImageTextGeriApp):
        src = QWEN_IMAGE_TEXT_GERI_TEMPLATE_PATH.read_text()
        rendered = Template(src).substitute(
            MODEL_ID=app.model_id,
            MAX_BATCH_SIZE=app.max_batch_size,
            SP_SIZE=app.sp_size,
        )
    elif isinstance(app, SuperGroupQwenImageApp):
        src = SUPERGROUP_QWEN_IMAGE_TEMPLATE_PATH.read_text()
        sub_defs: list[str] = []
        sub_names: list[str] = []
        inner_apps = app.inner_apps()
        for gi, (group, inner) in enumerate(zip(app.groups, inner_apps)):
            sub_name = f"sub_group_{gi}"
            sub_names.append(sub_name)
            if isinstance(inner, QwenImageTextGeriApp):
                sub_defs.append("\n".join([
                    "from cornserve_tasklib.task.unit.generator import QwenImageTextGeneratorTask",
                    f'{sub_name} = QwenImageTextGeneratorTask(',
                    f'    model_id="{inner.model_id}",',
                    f'    max_batch_size={inner.max_batch_size},',
                    f'    sp_size={inner.sp_size},',
                    ')',
                ]))
            elif isinstance(inner, QwenImageDisaggApp):
                sub_defs.append("\n".join([
                    "from cornserve_tasklib.task.unit.generator import ImageGeneratorTask",
                    "from cornserve_tasklib.task.unit.llm import LLMEmbeddingUnitTask",
                    "from cornserve_tasklib.task.composite.image_gen import MixedQwenImageTask",
                    f'{sub_name} = MixedQwenImageTask(',
                    f'    model_id="{inner.model_id}",',
                    f'    encoder_model_id="{inner.encoder_model_id}",',
                    f'    generator_routing_tasks=[ImageGeneratorTask(',
                    f'        model_id="{inner.model_id}",',
                    f'        max_batch_size={inner.generator_max_batch_size},',
                    f'        sp_size={inner.generator_sp_size},',
                    f'    )],',
                    f'    routing_weights=[1.0],',
                    f'    encoder_tp_size={inner.encoder_tp_size},',
                    f'    encoder_max_num_seqs={inner.encoder_max_num_seqs},',
                    f'    encoder_gpu_memory_utilization={inner.encoder_gpu_memory_utilization},',
                    f'    num_prefix_tokens_to_slice={inner.num_prefix_tokens_to_slice},',
                    ')',
                ]))
            elif isinstance(inner, MixedQwenImageDisaggApp):
                gen_lines = []
                for ri, route in enumerate(inner.generator_routes):
                    gen_lines.append(
                        f'    ImageGeneratorTask(model_id="{inner.model_id}", '
                        f'max_batch_size={route.max_batch_size}, sp_size={route.sp_size}),'
                    )
                wt_str = ", ".join(_format_float(w) for w in inner.routing_weights)
                sub_defs.append("\n".join([
                    "from cornserve_tasklib.task.unit.generator import ImageGeneratorTask",
                    "from cornserve_tasklib.task.unit.llm import LLMEmbeddingUnitTask",
                    "from cornserve_tasklib.task.composite.image_gen import MixedQwenImageTask",
                    f'{sub_name} = MixedQwenImageTask(',
                    f'    model_id="{inner.model_id}",',
                    f'    encoder_model_id="{inner.encoder_model_id}",',
                    f'    generator_routing_tasks=[',
                    *gen_lines,
                    f'    ],',
                    f'    routing_weights=[{wt_str}],',
                    f'    encoder_tp_size={inner.encoder_tp_size},',
                    f'    encoder_max_num_seqs={inner.encoder_max_num_seqs},',
                    f'    encoder_gpu_memory_utilization={inner.encoder_gpu_memory_utilization},',
                    f'    num_prefix_tokens_to_slice={inner.num_prefix_tokens_to_slice},',
                    ')',
                ]))
            elif isinstance(inner, MixedMonoQwenImageApp):
                gen_lines = []
                for ri, route in enumerate(inner.mono_routes):
                    gen_lines.append(
                        f'    QwenImageTextGeneratorTask(model_id="{inner.model_id}", '
                        f'max_batch_size={route.max_batch_size}, sp_size={route.sp_size}),'
                    )
                wt_str = ", ".join(_format_float(w) for w in inner.routing_weights)
                sub_defs.append("\n".join([
                    "from cornserve_tasklib.task.unit.generator import QwenImageTextGeneratorTask",
                    "from cornserve_tasklib.task.composite.image_gen import MixedMonoQwenImageTask",
                    f'{sub_name} = MixedMonoQwenImageTask(',
                    f'    mono_routing_tasks=[',
                    *gen_lines,
                    f'    ],',
                    f'    routing_weights=[{wt_str}],',
                    ')',
                ]))
            else:
                raise ValueError(f"Unsupported inner app type: {type(inner).__name__}")
        rendered = Template(src).substitute(
            SUB_TASK_DEFINITIONS="\n\n".join(sub_defs),
            SUB_TASK_NAMES=", ".join(sub_names),
            ROUTING_WEIGHTS=", ".join(
                _format_float(g.weight) for g in app.groups
            ),
        )
    elif isinstance(app, MixedMonoQwenImageApp):
        src = MIXED_MONO_QWEN_IMAGE_TEMPLATE_PATH.read_text()
        route_defs: list[str] = []
        route_names: list[str] = []
        for ri, route in enumerate(app.mono_routes):
            name = f"mono_route_{ri}"
            route_names.append(name)
            route_defs.append(
                "\n".join([
                    f"{name} = QwenImageTextGeneratorTask(",
                    f'    model_id="{app.model_id}",',
                    f"    max_batch_size={route.max_batch_size},",
                    f"    sp_size={route.sp_size},",
                    ")",
                ])
            )
        rendered = Template(src).substitute(
            MONO_ROUTING_TASK_DEFINITIONS="\n\n".join(route_defs),
            ROUTING_TASK_NAMES=", ".join(route_names),
            ROUTING_WEIGHTS=", ".join(
                _format_float(w) for w in app.routing_weights
            ),
        )
    elif isinstance(app, MixedQwenImageDisaggApp):
        src = MIXED_QWEN_IMAGE_DISAGG_TEMPLATE_PATH.read_text()
        route_defs: list[str] = []
        route_names: list[str] = []
        for ri, route in enumerate(app.generator_routes):
            name = f"gen_route_{ri}"
            route_names.append(name)
            route_defs.append(
                "\n".join([
                    f"{name} = ImageGeneratorTask(",
                    f'    model_id="{app.model_id}",',
                    f"    max_batch_size={route.max_batch_size},",
                    f"    sp_size={route.sp_size},",
                    ")",
                ])
            )
        rendered = Template(src).substitute(
            MODEL_ID=app.model_id,
            ENCODER_MODEL_ID=app.encoder_model_id,
            ENCODER_TP_SIZE=app.encoder_tp_size,
            ENCODER_MAX_NUM_SEQS=app.encoder_max_num_seqs,
            ENCODER_GPU_MEMORY_UTILIZATION=app.encoder_gpu_memory_utilization,
            GENERATOR_MAX_BATCH_SIZE=app.generator_max_batch_size,
            NUM_PREFIX_TOKENS_TO_SLICE=app.num_prefix_tokens_to_slice,
            GENERATOR_ROUTING_TASK_DEFINITIONS="\n\n".join(route_defs),
            ROUTING_TASK_NAMES=", ".join(route_names),
            ROUTING_WEIGHTS=", ".join(
                _format_float(w) for w in app.routing_weights
            ),
        )
    elif isinstance(app, QwenImageDisaggApp):
        src = QWEN_IMAGE_DISAGG_TEMPLATE_PATH.read_text()
        rendered = Template(src).substitute(
            MODEL_ID=app.model_id,
            ENCODER_MODEL_ID=app.encoder_model_id,
            ENCODER_TP_SIZE=app.encoder_tp_size,
            ENCODER_MAX_NUM_SEQS=app.encoder_max_num_seqs,
            ENCODER_GPU_MEMORY_UTILIZATION=app.encoder_gpu_memory_utilization,
            GENERATOR_SP_SIZE=app.generator_sp_size,
            GENERATOR_MAX_BATCH_SIZE=app.generator_max_batch_size,
            NUM_PREFIX_TOKENS_TO_SLICE=app.num_prefix_tokens_to_slice,
        )
    elif isinstance(app, DummyAudioGeriApp):
        src = DUMMY_AUDIO_GERI_TEMPLATE_PATH.read_text()
        rendered = Template(src).substitute(
            MODEL_ID=app.model_id,
            MAX_BATCH_SIZE=app.max_batch_size,
        )
    elif isinstance(app, DummyTalkerApp):
        src = DUMMY_TALKER_TEMPLATE_PATH.read_text()
        rendered = Template(src).substitute(
            MODEL_ID=app.model_id,
            MAX_NUM_SEQS=app.max_num_seqs,
            GPU_MEMORY_UTILIZATION=app.gpu_memory_utilization,
        )
    elif isinstance(app, DummyTalkerVocoderApp):
        src = DUMMY_TALKER_VOCODER_TEMPLATE_PATH.read_text()
        rendered = Template(src).substitute(
            MODEL_ID=app.model_id,
            MAX_NUM_SEQS=app.max_num_seqs,
            GPU_MEMORY_UTILIZATION=app.gpu_memory_utilization,
        )
    elif isinstance(app, ModServeApp):
        src = MLLM_TEMPLATE_PATH.read_text()
        rendered = Template(src).substitute(
            MODEL_ID=app.model_id,
            TASK_CLASS="MLLMTask",
            ENCODER_FISSION="True",
            ERIC_MAX_BATCH_SIZE=app.eric_max_batch_size,
            LLM_TP_SIZE=app.llm_tp_size,
            LLM_MAX_NUM_SEQS=app.llm_max_num_seqs,
            LLM_GPU_MEMORY_UTILIZATION=app.llm_gpu_memory_utilization,
        )
    elif isinstance(app, TimeSharingApp):
        src = TIME_SHARING_MLLM_TEMPLATE_PATH.read_text()
        rendered = Template(src).substitute(
            MODEL_ID=app.model_id,
            ENCODER_FISSION_PROB=app.encoder_fission_prob,
            ERIC_MAX_BATCH_SIZE=app.eric_max_batch_size,
            LLM_TP_SIZE=app.llm_tp_size,
            LLM_MAX_NUM_SEQS=app.llm_max_num_seqs,
            LLM_GPU_MEMORY_UTILIZATION=app.llm_gpu_memory_utilization,
        )
    elif isinstance(app, MixedMLLMApp):
        src = MIXED_MLLM_TEMPLATE_PATH.read_text()
        route_defs: list[str] = []
        route_names: list[str] = []
        for route_index, route in enumerate(app.routes):
            route_name = f"llm_route_{route_index}"
            route_names.append(route_name)
            route_defs.append(
                "\n".join(
                    [
                        f"{route_name} = LLMUnitTask(",
                        f'    model_id="{app.model_id}",',
                        "    receive_embeddings=True,",
                        f"    tp_size={route.llm_tp_size},",
                        f"    max_num_seqs={route.llm_max_num_seqs},",
                        (
                            "    gpu_memory_utilization="
                            f"{_format_float(route.llm_gpu_memory_utilization)},"
                        ),
                        ")",
                    ]
                )
            )

        rendered = Template(src).substitute(
            MODEL_ID=app.model_id,
            ERIC_MAX_BATCH_SIZE=app.eric_max_batch_size,
            LLM_ROUTING_TASK_DEFINITIONS="\n\n".join(route_defs),
            ROUTING_TASK_NAMES=", ".join(route_names),
            ROUTING_WEIGHTS=", ".join(
                _format_float(weight) for weight in app.routing_weights
            ),
        )
    elif isinstance(app, GroupedMixedMLLMApp):
        src = GROUPED_MIXED_MLLM_TEMPLATE_PATH.read_text()
        group_defs: list[str] = []
        group_task_names: list[str] = []

        for group_index, group in enumerate(app.groups):
            macro_literal = (
                f'"{group.macro_ut_deployment_id}"'
                if group.macro_ut_deployment_id is not None
                else "None"
            )
            route_defs: list[str] = []
            route_names: list[str] = []
            for route_index, route in enumerate(group.routes):
                route_name = f"group_{group_index}_llm_route_{route_index}"
                route_names.append(route_name)
                route_defs.append(
                    "\n".join(
                        [
                            f"{route_name} = LLMUnitTask(",
                            f'    model_id="{app.model_id}",',
                            "    receive_embeddings=True,",
                            f"    tp_size={route.llm_tp_size},",
                            f"    max_num_seqs={route.llm_max_num_seqs},",
                            (
                                "    gpu_memory_utilization="
                                f"{_format_float(route.llm_gpu_memory_utilization)},"
                            ),
                            f"    macro_ut_deployment_id={macro_literal},",
                            ")",
                        ]
                    )
                )

            group_task_name = f"mixed_group_{group_index}"
            group_task_names.append(group_task_name)
            group_defs.append(
                "\n".join(
                    [
                        *route_defs,
                        "",
                        f"{group_task_name} = MixedMLLMTask(",
                        f'    model_id="{app.model_id}",',
                        "    modalities=[Modality.IMAGE],",
                        f"    eric_max_batch_size={group.eric_max_batch_size},",
                        f"    llm_routing_tasks=[{', '.join(route_names)}],",
                        (
                            "    routing_weights=["
                            f"{', '.join(_format_float(weight) for weight in group.routing_weights)}],"
                        ),
                        f"    macro_ut_deployment_id={macro_literal},",
                        ")",
                    ]
                )
            )

        rendered = Template(src).substitute(
            GROUP_DEFINITIONS="\n\n".join(group_defs),
            GROUP_TASK_NAMES=", ".join(group_task_names),
            GROUP_ROUTING_WEIGHTS=", ".join(
                _format_float(weight) for weight in app.group_routing_weights
            ),
        )
    elif isinstance(app, MLLMRouterApp):
        src = MLLM_ROUTER_TEMPLATE_PATH.read_text()
        route_defs: list[str] = []
        route_names: list[str] = []
        for route_index, route in enumerate(app.routes):
            route_name = f"mllm_route_{route_index}"
            route_names.append(route_name)
            if isinstance(route, MLLMRouterMixedRouteConfig):
                # Generate LLMUnitTask definitions for each sub-route
                sub_defs: list[str] = []
                sub_names: list[str] = []
                for si, lc in enumerate(route.llm_configs):
                    sub_name = f"llm_sub_{route_index}_{si}"
                    sub_names.append(sub_name)
                    sub_defs.append(
                        "\n".join(
                            [
                                f"{sub_name} = LLMUnitTask(",
                                f'    model_id="{app.model_id}",',
                                f"    tp_size={lc.llm_tp_size},",
                                f"    max_num_seqs={lc.llm_max_num_seqs},",
                                (
                                    "    gpu_memory_utilization="
                                    f"{_format_float(lc.llm_gpu_memory_utilization)},"
                                ),
                                "    receive_embeddings=True,",
                                ")",
                            ]
                        )
                    )
                # Generate MixedMLLMTask wrapping the sub-routes
                sub_weights_str = ", ".join(
                    _format_float(w) for w in route.llm_routing_weights
                )
                route_defs.append(
                    "\n".join(sub_defs)
                    + "\n\n"
                    + "\n".join(
                        [
                            f"{route_name} = MixedMLLMTask(",
                            f'    model_id="{app.model_id}",',
                            "    modalities=[Modality.IMAGE],",
                            f"    eric_max_batch_size={route.eric_max_batch_size},",
                            f"    llm_routing_tasks=[{', '.join(sub_names)}],",
                            f"    routing_weights=[{sub_weights_str}],",
                            ")",
                        ]
                    )
                )
            elif isinstance(route, MLLMRouterRouteConfig) and route.is_time_sharing():
                route_defs.append(
                    "\n".join(
                        [
                            f"{route_name} = TimeSharingMLLMTask(",
                            f'    model_id="{app.model_id}",',
                            "    modalities=[Modality.IMAGE],",
                            (
                                "    encoder_fission_prob="
                                f"{_format_float(route.encoder_fission_prob or 0.0)},"
                            ),
                            f"    eric_max_batch_size={route.eric_max_batch_size},",
                            f"    llm_tp_size={route.llm_tp_size},",
                            f"    llm_max_num_seqs={route.llm_max_num_seqs},",
                            (
                                "    llm_gpu_memory_utilization="
                                f"{_format_float(route.llm_gpu_memory_utilization)},"
                            ),
                            ")",
                        ]
                    )
                )
            else:
                assert isinstance(route, MLLMRouterRouteConfig)
                route_defs.append(
                    "\n".join(
                        [
                            f"{route_name} = MLLMTask(",
                            f'    model_id="{app.model_id}",',
                            "    modalities=[Modality.IMAGE],",
                            f"    encoder_fission={route.encoder_fission},",
                            f"    eric_max_batch_size={route.eric_max_batch_size},",
                            f"    llm_tp_size={route.llm_tp_size},",
                            f"    llm_max_num_seqs={route.llm_max_num_seqs},",
                            (
                                "    llm_gpu_memory_utilization="
                                f"{_format_float(route.llm_gpu_memory_utilization)},"
                            ),
                            ")",
                        ]
                    )
                )

        rendered = Template(src).substitute(
            ROUTE_DEFINITIONS="\n\n".join(route_defs),
            ROUTING_TASK_NAMES=", ".join(route_names),
            ROUTING_WEIGHTS=", ".join(
                _format_float(weight) for weight in app.routing_weights
            ),
        )
    elif isinstance(app, OmniRouterApp):
        src = OMNI_ROUTER_TEMPLATE_PATH.read_text()
        route_defs: list[str] = []
        route_names: list[str] = []

        modalities_str = "[Modality.IMAGE, Modality.VIDEO, Modality.AUDIO]"

        for route_index, route in enumerate(app.routes):
            route_name = f"omni_route_{route_index}"
            route_names.append(route_name)

            if route.route_type == "omni_mllm":
                route_defs.append(
                    "\n".join(
                        [
                            f"{route_name} = OmniMLLMTask(",
                            f'    model_id="{app.model_id}",',
                            f"    modalities={modalities_str},",
                            f"    encoder_fission={route.encoder_fission},",
                            f"    vocoder_fission={route.vocoder_fission},",
                            "    coalesce_encoder_invocations=True,",
                            f"    eric_max_batch_size={route.eric_max_batch_size},",
                            f"    llm_tp_size={route.llm_tp_size},",
                            f"    llm_max_num_seqs={route.llm_max_num_seqs},",
                            (
                                "    llm_gpu_memory_utilization="
                                f"{_format_float(route.llm_gpu_memory_utilization)},"
                            ),
                            ")",
                        ]
                    )
                )
            elif route.route_type == "omni_time_sharing":
                trw_parts = []
                for type_idx, mode_weights in sorted(route.type_routing_weights.items()):
                    mw_str = ", ".join(
                        f'"{k}": {_format_float(v)}'
                        for k, v in sorted(mode_weights.items())
                    )
                    trw_parts.append(f"        {type_idx}: {{{mw_str}}},")
                trw_str = "{\n" + "\n".join(trw_parts) + "\n    }"

                # Map planner short names (img/vid/audio) to Modality enum names
                _modality_enum = {"img": "IMAGE", "vid": "VIDEO", "audio": "AUDIO"}
                offloaded_str = ", ".join(
                    f"Modality.{_modality_enum.get(m, m.upper())}"
                    for m in route.offloaded_modalities
                )
                route_defs.append(
                    "\n".join(
                        [
                            f"{route_name} = OmniTimeSharingMLLMTask(",
                            f'    model_id="{app.model_id}",',
                            f"    offloaded_modalities=[{offloaded_str}],",
                            f"    type_routing_weights={trw_str},",
                            f"    vocoder_fission={route.vocoder_fission},",
                            "    coalesce_encoder_invocations=True,",
                            f"    eric_max_batch_size={route.eric_max_batch_size},",
                            f"    llm_tp_size={route.llm_tp_size},",
                            f"    llm_max_num_seqs={route.llm_max_num_seqs},",
                            (
                                "    llm_gpu_memory_utilization="
                                f"{_format_float(route.llm_gpu_memory_utilization)},"
                            ),
                            ")",
                        ]
                    )
                )

        rendered = Template(src).substitute(
            ROUTE_DEFINITIONS="\n\n".join(route_defs),
            ROUTING_TASK_NAMES=", ".join(route_names),
            ROUTING_WEIGHTS=", ".join(
                _format_float(weight) for weight in app.routing_weights
            ),
        )
    elif isinstance(app, OmniFlexApp):
        src = OMNI_FLEX_TEMPLATE_PATH.read_text()

        _modality_enum = {"img": "IMAGE", "vid": "VIDEO", "audio": "AUDIO"}

        # Build groups list as Python code
        group_code_parts: list[str] = []
        for gi, group in enumerate(app.groups):
            offloaded_str = ", ".join(
                f'"{m}"' for m in group.offloaded_modalities
            )
            bs_str = ", ".join(
                f'"{m}": {bs}'
                for m, bs in sorted(group.eric_max_batch_sizes.items())
            )
            thinker_parts = []
            for t in group.thinkers:
                thinker_parts.append(
                    f"ThinkerLLMConfig("
                    f"tp_size={t.llm_tp_size}, "
                    f"max_num_seqs={t.llm_max_num_seqs}, "
                    f"gpu_memory_utilization="
                    f"{_format_float(t.llm_gpu_memory_utilization)}, "
                    f"weight={_format_float(t.weight)})"
                )
            thinkers_str = ", ".join(thinker_parts)
            group_code_parts.append(
                f"        OmniFlexGroupConfig(\n"
                f"            offloaded_modalities=[{offloaded_str}],\n"
                f"            eric_max_batch_sizes={{{bs_str}}},\n"
                f"            thinkers=[{thinkers_str}],\n"
                f"        )"
            )

        groups_str = "[\n" + ",\n".join(group_code_parts) + ",\n    ]"

        # Build type_routing_weights as Python code
        trw_parts = []
        for type_idx, gweights in sorted(app.type_routing_weights.items()):
            gw_str = ", ".join(
                f"{gidx}: {_format_float(w)}"
                for gidx, w in sorted(gweights.items())
            )
            trw_parts.append(f"        {type_idx}: {{{gw_str}}},")
        trw_str = "{\n" + "\n".join(trw_parts) + "\n    }" if trw_parts else "{}"

        task_def = (
            f'omni_flex = OmniFlexTask(\n'
            f'    model_id="{app.model_id}",\n'
            f'    groups={groups_str},\n'
            f'    type_routing_weights={trw_str},\n'
            f'    vocoder_fission={app.vocoder_fission},\n'
            f'    coalesce_encoder_invocations=True,\n'
            f')'
        )

        rendered = Template(src).substitute(TASK_DEFINITION=task_def)
    else:
        raise NotImplementedError(f"App type {type(app).__name__} is not supported.")

    return rendered.strip()


def app_to_expected_task(app: AppType):
    """Create the expected UnitTask from an app configuration.

    This creates the task definition that we expect the template-generated
    source code to produce.

    Args:
        app: The app configuration

    Returns:
        The expected UnitTask instance
    """
    from cornserve_tasklib.task.unit.encoder import (
        DummyEncoderTask,
        Modality as EncoderModality,
    )
    from cornserve_tasklib.task.unit.generator import (
        DummyAudioGeneratorTask,
        DummyImageGeneratorTask,
        Modality as GeneratorModality,
        QwenImageTextGeneratorTask,
    )
    from cornserve_tasklib.task.unit.llm import DummyMLLMUnitTask
    from cornserve_tasklib.task.unit.omni import (
        DummyOmniTalkerEmbeddingTask,
        DummyOmniTalkerVocoderTask,
    )

    if isinstance(app, DummyEricApp):
        return DummyEncoderTask(
            model_ids={app.model_id},
            modality=EncoderModality(app.modality),
            max_batch_size=app.max_batch_size,
            tp_size=app.tp_size,
        )
    elif isinstance(app, DummyLLMApp):
        return DummyMLLMUnitTask(
            model_id=app.model_id,
            receive_embeddings=True,
            tp_size=app.tp_size,
            max_num_seqs=app.max_num_seqs,
            gpu_memory_utilization=app.gpu_memory_utilization,
        )
    elif isinstance(app, MonolithicLLMApp):
        return DummyMLLMUnitTask(
            model_id=app.model_id,
            receive_embeddings=False,
            tp_size=app.tp_size,
            max_num_seqs=app.max_num_seqs,
            gpu_memory_utilization=app.gpu_memory_utilization,
        )
    elif isinstance(app, PrefillLLMApp):
        from cornserve_tasklib.task.unit.llm import LLMUnitTask

        return LLMUnitTask(
            model_id=app.model_id,
            receive_embeddings=app.receive_embeddings,
            tp_size=app.tp_size,
            max_num_seqs=app.max_num_seqs,
            gpu_memory_utilization=app.gpu_memory_utilization,
        )
    elif isinstance(app, DecodeLLMApp):
        from cornserve_tasklib.task.unit.llm import LLMUnitTask

        return LLMUnitTask(
            model_id=app.model_id,
            receive_embeddings=app.receive_embeddings,
            enable_prefix_caching=True,
            tp_size=app.tp_size,
            max_num_seqs=app.max_num_seqs,
            gpu_memory_utilization=app.gpu_memory_utilization,
        )
    elif isinstance(app, OmniMLLMApp):
        from cornserve_tasklib.task.unit.llm import OmniMLLMUnitTask

        return OmniMLLMUnitTask(
            model_id=app.model_id,
            tp_size=app.tp_size,
            max_num_seqs=app.max_num_seqs,
            gpu_memory_utilization=app.gpu_memory_utilization,
            disable_audio_enc=app.disable_audio_enc,
            disable_image_enc=app.disable_image_enc,
            disable_video_enc=app.disable_video_enc,
        )
    elif isinstance(app, DummyImageGeriApp):
        return DummyImageGeneratorTask(
            model_id=app.model_id,
            modality=GeneratorModality.IMAGE,
            max_batch_size=app.max_batch_size,
            sp_size=app.sp_size,
        )
    elif isinstance(app, DummyAudioGeriApp):
        return DummyAudioGeneratorTask(
            model_id=app.model_id,  # type: ignore[arg-type]
            modality=GeneratorModality.AUDIO,
            max_batch_size=app.max_batch_size,
        )
    elif isinstance(app, QwenImageTextGeriApp):
        return QwenImageTextGeneratorTask(
            model_id=app.model_id,
            max_batch_size=app.max_batch_size,
            sp_size=app.sp_size,
        )
    elif isinstance(app, SuperGroupQwenImageApp):
        raise NotImplementedError(
            "SuperGroupQwenImageApp has multiple sub-tasks. Use app_to_expected_tasks() instead."
        )
    elif isinstance(app, MixedMonoQwenImageApp):
        raise NotImplementedError(
            "MixedMonoQwenImageApp has multiple sub-tasks. Use app_to_expected_tasks() instead."
        )
    elif isinstance(app, MixedQwenImageDisaggApp):
        raise NotImplementedError(
            "MixedQwenImageDisaggApp has multiple sub-tasks. Use app_to_expected_tasks() instead."
        )
    elif isinstance(app, QwenImageDisaggApp):
        raise NotImplementedError(
            "QwenImageDisaggApp has multiple sub-tasks. Use app_to_expected_tasks() instead."
        )
    elif isinstance(app, DummyTalkerApp):
        return DummyOmniTalkerEmbeddingTask(
            model_id=app.model_id,  # type: ignore[arg-type]
            max_num_seqs=app.max_num_seqs,
            gpu_memory_utilization=app.gpu_memory_utilization,
        )
    elif isinstance(app, DummyTalkerVocoderApp):
        return DummyOmniTalkerVocoderTask(
            model_id=app.model_id,  # type: ignore[arg-type]
            max_num_seqs=app.max_num_seqs,
            gpu_memory_utilization=app.gpu_memory_utilization,
        )
    elif isinstance(app, ModServeApp):
        raise NotImplementedError(
            "ModServeApp has multiple sub-tasks. Use app_to_expected_tasks() instead."
        )
    elif isinstance(app, TimeSharingApp):
        raise NotImplementedError(
            "TimeSharingApp has multiple sub-tasks. Use app_to_expected_tasks() instead."
        )
    elif isinstance(app, MixedMLLMApp):
        raise NotImplementedError(
            "MixedMLLMApp has multiple sub-tasks. Use app_to_expected_tasks() instead."
        )
    elif isinstance(app, GroupedMixedMLLMApp):
        raise NotImplementedError(
            "GroupedMixedMLLMApp has multiple sub-tasks. "
            "Use app_to_expected_tasks() instead."
        )
    elif isinstance(app, MLLMRouterApp):
        raise NotImplementedError(
            "MLLMRouterApp has multiple sub-tasks. Use app_to_expected_tasks() instead."
        )
    elif isinstance(app, OmniRouterApp):
        raise NotImplementedError(
            "OmniRouterApp has multiple sub-tasks. Use app_to_expected_tasks() instead."
        )
    elif isinstance(app, OmniFlexApp):
        raise NotImplementedError(
            "OmniFlexApp has multiple sub-tasks. Use app_to_expected_tasks() instead."
        )
    else:
        raise NotImplementedError(f"App type {type(app).__name__} is not supported.")


def _tasks_match(task1: UnitTask, task2: UnitTask) -> bool:
    """Check if two tasks are equivalent by root class fields."""
    if task1.__class__.__name__ != task2.__class__.__name__:
        return False
    root_cls = task1.root_unit_task_cls
    for field_name in root_cls.model_fields:
        if field_name == "id":
            continue
        if getattr(task1, field_name, None) != getattr(task2, field_name, None):
            return False
    return True


def _merge_task_replica_specs(
    task_replica_specs: list[tuple[UnitTask, int]],
) -> list[tuple[UnitTask, int]]:
    """Merge equivalent task specs by summing replica counts."""
    merged: list[tuple[UnitTask, int]] = []
    for task, num_replicas in task_replica_specs:
        if num_replicas <= 0:
            continue

        for index, (existing_task, existing_replicas) in enumerate(merged):
            if _tasks_match(existing_task, task):
                merged[index] = (existing_task, existing_replicas + num_replicas)
                break
        else:
            merged.append((task, num_replicas))

    return merged


def app_to_expected_task_replica_specs(app: AppType) -> list[tuple[UnitTask, int]]:
    """Return expected unit tasks with desired replica counts.

    Equivalent tasks are merged by summing their replica counts.
    """
    if isinstance(app, SuperGroupQwenImageApp):
        # Recurse into each inner app and merge all specs
        all_specs: list[tuple[UnitTask, int]] = []
        for inner in app.inner_apps():
            all_specs.extend(app_to_expected_task_replica_specs(inner))
        return _merge_task_replica_specs(all_specs)

    if isinstance(app, MixedMonoQwenImageApp):
        from cornserve_tasklib.task.unit.generator import QwenImageTextGeneratorTask

        specs: list[tuple[UnitTask, int]] = []
        for route in app.mono_routes:
            specs.append(
                (
                    QwenImageTextGeneratorTask(
                        model_id=app.model_id,
                        max_batch_size=route.max_batch_size,
                        sp_size=route.sp_size,
                    ),
                    route.num_replicas,
                )
            )
        return _merge_task_replica_specs(specs)

    if isinstance(app, MixedQwenImageDisaggApp):
        from cornserve_tasklib.task.unit.generator import ImageGeneratorTask
        from cornserve_tasklib.task.unit.llm import LLMEmbeddingUnitTask

        specs: list[tuple[UnitTask, int]] = [
            (
                LLMEmbeddingUnitTask(
                    model_id=app.encoder_model_id,
                    receive_embeddings=False,
                    tp_size=app.encoder_tp_size,
                    max_num_seqs=app.encoder_max_num_seqs,
                    gpu_memory_utilization=app.encoder_gpu_memory_utilization,
                ),
                app.num_encoders,
            ),
        ]
        for route in app.generator_routes:
            specs.append(
                (
                    ImageGeneratorTask(
                        model_id=app.model_id,
                        max_batch_size=route.max_batch_size,
                        sp_size=route.sp_size,
                    ),
                    route.num_replicas,
                )
            )
        return _merge_task_replica_specs(specs)

    if isinstance(app, QwenImageDisaggApp):
        from cornserve_tasklib.task.unit.generator import ImageGeneratorTask
        from cornserve_tasklib.task.unit.llm import LLMEmbeddingUnitTask

        return _merge_task_replica_specs(
            [
                (
                    LLMEmbeddingUnitTask(
                        model_id=app.encoder_model_id,
                        receive_embeddings=False,
                        tp_size=app.encoder_tp_size,
                        max_num_seqs=app.encoder_max_num_seqs,
                        gpu_memory_utilization=app.encoder_gpu_memory_utilization,
                    ),
                    app.num_encoders,
                ),
                (
                    ImageGeneratorTask(
                        model_id=app.model_id,
                        max_batch_size=app.generator_max_batch_size,
                        sp_size=app.generator_sp_size,
                    ),
                    app.num_generators,
                ),
            ]
        )

    if isinstance(app, ModServeApp):
        from cornserve_tasklib.task.unit.encoder import (
            EncoderTask,
            Modality as EncoderModality,
        )
        from cornserve_tasklib.task.unit.llm import LLMUnitTask

        return _merge_task_replica_specs(
            [
                (
                    EncoderTask(
                        model_ids={app.model_id},
                        modality=EncoderModality.IMAGE,
                        max_batch_size=app.eric_max_batch_size,
                    ),
                    app.num_erics,
                ),
                (
                    LLMUnitTask(
                        model_id=app.model_id,
                        receive_embeddings=True,
                        tp_size=app.llm_tp_size,
                        max_num_seqs=app.llm_max_num_seqs,
                        gpu_memory_utilization=app.llm_gpu_memory_utilization,
                    ),
                    app.num_llms,
                ),
            ]
        )

    if isinstance(app, TimeSharingApp):
        from cornserve_tasklib.task.unit.encoder import (
            EncoderTask,
            Modality as EncoderModality,
        )
        from cornserve_tasklib.task.unit.llm import LLMUnitTask

        return _merge_task_replica_specs(
            [
                (
                    EncoderTask(
                        model_ids={app.model_id},
                        modality=EncoderModality.IMAGE,
                        max_batch_size=app.eric_max_batch_size,
                    ),
                    app.num_erics,
                ),
                (
                    LLMUnitTask(
                        model_id=app.model_id,
                        receive_embeddings=False,
                        tp_size=app.llm_tp_size,
                        max_num_seqs=app.llm_max_num_seqs,
                        gpu_memory_utilization=app.llm_gpu_memory_utilization,
                    ),
                    app.num_llms,
                ),
            ]
        )

    if isinstance(app, MixedMLLMApp):
        from cornserve_tasklib.task.unit.encoder import (
            EncoderTask,
            Modality as EncoderModality,
        )
        from cornserve_tasklib.task.unit.llm import LLMUnitTask

        specs: list[tuple[UnitTask, int]] = [
            (
                EncoderTask(
                    model_ids={app.model_id},
                    modality=EncoderModality.IMAGE,
                    max_batch_size=app.eric_max_batch_size,
                ),
                app.num_erics,
            )
        ]
        for route in app.routes:
            specs.append(
                (
                    LLMUnitTask(
                        model_id=app.model_id,
                        receive_embeddings=True,
                        tp_size=route.llm_tp_size,
                        max_num_seqs=route.llm_max_num_seqs,
                        gpu_memory_utilization=route.llm_gpu_memory_utilization,
                    ),
                    route.llm_num_replicas,
                )
            )

        return _merge_task_replica_specs(specs)

    if isinstance(app, GroupedMixedMLLMApp):
        from cornserve_tasklib.task.unit.encoder import (
            EncoderTask,
            Modality as EncoderModality,
        )
        from cornserve_tasklib.task.unit.llm import LLMUnitTask

        specs: list[tuple[UnitTask, int]] = []
        for group in app.groups:
            specs.append(
                (
                    EncoderTask(
                        model_ids={app.model_id},
                        modality=EncoderModality.IMAGE,
                        max_batch_size=group.eric_max_batch_size,
                        macro_ut_deployment_id=group.macro_ut_deployment_id,
                    ),
                    group.num_erics,
                )
            )

            for route in group.routes:
                specs.append(
                    (
                        LLMUnitTask(
                            model_id=app.model_id,
                            receive_embeddings=True,
                            tp_size=route.llm_tp_size,
                            max_num_seqs=route.llm_max_num_seqs,
                            gpu_memory_utilization=route.llm_gpu_memory_utilization,
                            macro_ut_deployment_id=group.macro_ut_deployment_id,
                        ),
                        route.llm_num_replicas,
                    )
                )

        return _merge_task_replica_specs(specs)

    if isinstance(app, MLLMRouterApp):
        from cornserve_tasklib.task.unit.encoder import (
            EncoderTask,
            Modality as EncoderModality,
        )
        from cornserve_tasklib.task.unit.llm import LLMUnitTask

        specs: list[tuple[UnitTask, int]] = []
        for route in app.routes:
            if isinstance(route, MLLMRouterMixedRouteConfig):
                # Mixed route: shared encoder + N LLM sub-routes
                specs.append(
                    (
                        EncoderTask(
                            model_ids={app.model_id},
                            modality=EncoderModality.IMAGE,
                            max_batch_size=route.eric_max_batch_size,
                        ),
                        route.eric_num_replicas,
                    )
                )
                for lc in route.llm_configs:
                    specs.append(
                        (
                            LLMUnitTask(
                                model_id=app.model_id,
                                receive_embeddings=True,
                                tp_size=lc.llm_tp_size,
                                max_num_seqs=lc.llm_max_num_seqs,
                                gpu_memory_utilization=lc.llm_gpu_memory_utilization,
                            ),
                            lc.llm_num_replicas,
                        )
                    )
            else:
                assert isinstance(route, MLLMRouterRouteConfig)
                if route.has_encoder_task():
                    specs.append(
                        (
                            EncoderTask(
                                model_ids={app.model_id},
                                modality=EncoderModality.IMAGE,
                                max_batch_size=route.eric_max_batch_size,
                            ),
                            route.eric_num_replicas,
                        )
                    )

                specs.append(
                    (
                        LLMUnitTask(
                            model_id=app.model_id,
                            receive_embeddings=route.llm_receive_embeddings(),
                            tp_size=route.llm_tp_size,
                            max_num_seqs=route.llm_max_num_seqs,
                            gpu_memory_utilization=route.llm_gpu_memory_utilization,
                        ),
                        route.llm_num_replicas,
                    )
                )

        return _merge_task_replica_specs(specs)

    if isinstance(app, EPDMLLMApp):
        from cornserve_tasklib.task.unit.encoder import (
            EncoderTask,
            Modality as EncoderModality,
        )
        from cornserve_tasklib.task.unit.llm import DecodeLLMUnitTask, PrefillLLMUnitTask

        return _merge_task_replica_specs(
            [
                (
                    EncoderTask(
                        model_ids={app.model_id},
                        modality=EncoderModality.IMAGE,
                        max_batch_size=app.eric_max_batch_size,
                    ),
                    app.num_eric_replicas,
                ),
                (
                    PrefillLLMUnitTask(
                        model_id=app.model_id,
                        receive_embeddings=True,
                        tp_size=app.prefill_tp_size,
                        max_num_seqs=app.prefill_max_num_seqs,
                        # HACK: Qwen3-VL OOMs at 0.9 on prefill
                        gpu_memory_utilization=min(app.prefill_gpu_memory_utilization, 0.85)
                        if "Qwen3-VL" in app.model_id
                        else app.prefill_gpu_memory_utilization,
                    ),
                    app.num_prefill_replicas,
                ),
                (
                    DecodeLLMUnitTask(
                        model_id=app.model_id,
                        receive_embeddings=True,
                        tp_size=app.decode_tp_size,
                        max_num_seqs=app.decode_max_num_seqs,
                        gpu_memory_utilization=app.decode_gpu_memory_utilization,
                    ),
                    app.num_decode_replicas,
                ),
            ]
        )

    if isinstance(app, PDMLLMApp):
        from cornserve_tasklib.task.unit.llm import DecodeLLMUnitTask, PrefillLLMUnitTask

        return _merge_task_replica_specs(
            [
                (
                    PrefillLLMUnitTask(
                        model_id=app.model_id,
                        receive_embeddings=False,
                        tp_size=app.prefill_tp_size,
                        max_num_seqs=app.prefill_max_num_seqs,
                        gpu_memory_utilization=app.prefill_gpu_memory_utilization,
                    ),
                    app.num_prefill_replicas,
                ),
                (
                    DecodeLLMUnitTask(
                        model_id=app.model_id,
                        receive_embeddings=False,
                        tp_size=app.decode_tp_size,
                        max_num_seqs=app.decode_max_num_seqs,
                        gpu_memory_utilization=app.decode_gpu_memory_utilization,
                    ),
                    app.num_decode_replicas,
                ),
            ]
        )

    if isinstance(app, OmniRouterApp):
        from cornserve_tasklib.task.unit.encoder import (
            EncoderTask,
            Modality as EncoderModality,
        )
        from cornserve_tasklib.task.unit.llm import LLMUnitTask
        from cornserve_tasklib.task.unit.omni import OmniTalkerVocoderTask

        specs: list[tuple[UnitTask, int]] = []
        for route in app.routes:
            receive_embeddings = (
                route.route_type == "omni_mllm" and route.encoder_fission
            )

            # Per-modality encoders
            for modality, n_replicas in [
                ("image", route.img_eric_num_replicas),
                ("video", route.vid_eric_num_replicas),
                ("audio", route.audio_eric_num_replicas),
            ]:
                if n_replicas > 0:
                    specs.append(
                        (
                            EncoderTask(
                                model_ids={app.model_id},
                                modality=EncoderModality(modality),
                                max_batch_size=route.eric_max_batch_size,
                            ),
                            n_replicas,
                        )
                    )

            # LLM
            specs.append(
                (
                    LLMUnitTask(
                        model_id=app.model_id,
                        receive_embeddings=receive_embeddings,
                        tp_size=route.llm_tp_size,
                        max_num_seqs=route.llm_max_num_seqs,
                        gpu_memory_utilization=route.llm_gpu_memory_utilization,
                    ),
                    route.llm_num_replicas,
                )
            )

            # Audio output
            if route.vocoder_fission:
                # Fissioned: separate Talker + AudioGeri
                if route.talker_num_replicas > 0:
                    from cornserve_tasklib.task.unit.omni import OmniTalkerEmbeddingTask

                    specs.append(
                        (
                            OmniTalkerEmbeddingTask(
                                model_id=app.model_id,  # type: ignore[arg-type]
                            ),
                            route.talker_num_replicas,
                        )
                    )
                if route.audio_geri_num_replicas > 0:
                    from cornserve_tasklib.task.unit.generator import AudioGeneratorTask

                    specs.append(
                        (
                            AudioGeneratorTask(
                                model_id=app.model_id,  # type: ignore[arg-type]
                            ),
                            route.audio_geri_num_replicas,
                        )
                    )
            else:
                # Combined TalkerVocoder
                if route.talker_vocoder_num_replicas > 0:
                    specs.append(
                        (
                            OmniTalkerVocoderTask(
                                model_id=app.model_id,  # type: ignore[arg-type]
                            ),
                            route.talker_vocoder_num_replicas,
                        )
                    )

        return _merge_task_replica_specs(specs)

    if isinstance(app, OmniFlexApp):
        from cornserve_tasklib.task.unit.encoder import (
            EncoderTask,
            Modality as EncoderModality,
        )
        from cornserve_tasklib.task.unit.llm import LLMUnitTask

        _short_to_full = {"img": "image", "vid": "video", "audio": "audio"}
        specs: list[tuple[UnitTask, int]] = []
        base_id = None  # No macro_ut_deployment_id at schema level

        for gi, group in enumerate(app.groups):
            group_dep_id = f"{base_id}_g{gi}" if base_id else None

            # Per-modality encoders
            _enc_replicas = {
                "img": group.img_eric_num_replicas,
                "vid": group.vid_eric_num_replicas,
                "audio": group.audio_eric_num_replicas,
            }
            for mod_short in group.offloaded_modalities:
                n_reps = _enc_replicas.get(mod_short, 0)
                if n_reps > 0:
                    specs.append(
                        (
                            EncoderTask(
                                model_ids={app.model_id},
                                modality=EncoderModality(
                                    _short_to_full[mod_short]
                                ),
                                max_batch_size=group.eric_max_batch_sizes.get(
                                    mod_short, 1
                                ),
                                macro_ut_deployment_id=group_dep_id,
                            ),
                            n_reps,
                        )
                    )

            # LLMs (all receive_embeddings=False)
            for t in group.thinkers:
                specs.append(
                    (
                        LLMUnitTask(
                            model_id=app.model_id,
                            receive_embeddings=False,
                            tp_size=t.llm_tp_size,
                            max_num_seqs=t.llm_max_num_seqs,
                            gpu_memory_utilization=t.llm_gpu_memory_utilization,
                            macro_ut_deployment_id=group_dep_id,
                        ),
                        t.llm_num_replicas,
                    )
                )

        # Shared audio output
        ao_dep_id = f"{base_id}_ao" if base_id else None
        if app.vocoder_fission:
            from cornserve_tasklib.task.unit.omni import OmniTalkerEmbeddingTask
            from cornserve_tasklib.task.unit.generator import AudioGeneratorTask

            if app.talker_num_replicas > 0:
                specs.append(
                    (
                        OmniTalkerEmbeddingTask(
                            model_id=app.model_id,  # type: ignore[arg-type]
                            macro_ut_deployment_id=ao_dep_id,
                        ),
                        app.talker_num_replicas,
                    )
                )
            if app.audio_geri_num_replicas > 0:
                specs.append(
                    (
                        AudioGeneratorTask(
                            model_id=app.model_id,  # type: ignore[arg-type]
                            macro_ut_deployment_id=ao_dep_id,
                        ),
                        app.audio_geri_num_replicas,
                    )
                )
        else:
            from cornserve_tasklib.task.unit.omni import OmniTalkerVocoderTask

            if app.talker_vocoder_num_replicas > 0:
                specs.append(
                    (
                        OmniTalkerVocoderTask(
                            model_id=app.model_id,  # type: ignore[arg-type]
                            macro_ut_deployment_id=ao_dep_id,
                        ),
                        app.talker_vocoder_num_replicas,
                    )
                )

        return _merge_task_replica_specs(specs)

    num_replicas: int = getattr(app, "num_replicas", 1)
    return [(app_to_expected_task(app), num_replicas)]


def app_to_expected_tasks(app: AppType) -> list[UnitTask]:
    """Return unique expected UnitTasks for an app."""
    return [task for task, _ in app_to_expected_task_replica_specs(app)]


def load_module_from_source(source_code: str, module_name: str) -> ModuleType:
    """Load a Python module from source code string.

    This is the same function used by the gateway to load app source code.

    Args:
        source_code: Python source code as a string
        module_name: Name for the module

    Returns:
        The loaded module

    Raises:
        ImportError: If the module cannot be loaded
    """
    spec = importlib.util.spec_from_loader(
        module_name, loader=None, origin="<cornserve_app>"
    )
    if spec is None:
        raise ImportError(f"Failed to create spec for module {module_name}")

    module = importlib.util.module_from_spec(spec)

    try:
        exec(source_code, module.__dict__)
        return module
    except Exception as e:
        raise ImportError(f"Failed to execute module code: {e}") from e


def extract_tasks_from_source(source_code: str) -> list[UnitTask]:
    """Extract unit tasks from app source code.

    This mimics how the gateway extracts tasks from registered apps.

    Args:
        source_code: The app source code

    Returns:
        List of unit tasks found in the source code

    Raises:
        ValueError: If the source code doesn't contain a valid Config class
    """
    module = load_module_from_source(source_code, "test_app")
    if not hasattr(module, "Config"):
        raise ValueError("Source code does not contain a Config class")

    config = module.Config
    if not hasattr(config, "tasks"):
        raise ValueError("Config class does not have a tasks attribute")

    tasks = discover_unit_tasks(config.tasks.values())
    return tasks
