"""An app that runs a Multimodal LLM task."""

from __future__ import annotations

from collections.abc import AsyncIterator

from cornserve.app.base import AppConfig
from cornserve.task.builtins.encoder import Modality
from cornserve.task.builtins.llm import DisaggregatedMLLMTask, OpenAIChatCompletionChunk, OpenAIChatCompletionRequest

mllm = DisaggregatedMLLMTask(
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    modalities=[Modality.IMAGE],
)

# model_id="google/gemma-3-4b-it",

class Config(AppConfig):
    """App configuration model."""

    tasks = {"mllm": mllm}


async def serve(request: OpenAIChatCompletionRequest) -> AsyncIterator[OpenAIChatCompletionChunk]:
    """Main serve function for the app."""
    return await mllm(request)
