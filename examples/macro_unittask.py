"""Qwen3 Omni macro-unit app.

This app exposes a single logical unit task while preserving the original
multi-executor Omni dataplane.
"""

from collections.abc import AsyncIterator

from cornserve_tasklib.task.composite.omni import OmniInput, Qwen3OmniMacroUnitTask
from cornserve_tasklib.task.unit.encoder import Modality
from cornserve_tasklib.task.unit.llm import OpenAIChatCompletionChunk

from cornserve.app.base import AppConfig

omni = Qwen3OmniMacroUnitTask(
    model_id="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    modalities=[Modality.IMAGE, Modality.VIDEO, Modality.AUDIO],
    encoder_fission=False,
    vocoder_fission=True,
    eric_max_batch_size=1,
    llm_tp_size=1,
    llm_max_num_seqs=32,
    llm_gpu_memory_utilization=0.9,
    macro_ut_deployment_id="qwen3_omni_macro_unittask",
)


class Config(AppConfig):
    """App configuration model."""

    tasks = {"omni": omni}


async def serve(request: OmniInput) -> AsyncIterator[OpenAIChatCompletionChunk]:
    """Main serve function for the app."""
    return await omni(request)
