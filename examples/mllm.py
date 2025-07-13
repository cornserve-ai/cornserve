"""An app that runs a Multimodal LLM task."""

from __future__ import annotations

from collections.abc import AsyncIterator

from cornserve.app.base import AppConfig
from cornserve.task.builtins.encoder import Modality
from cornserve.task.builtins.llm import MLLMTask, OpenAIChatCompletionChunk, OpenAIChatCompletionRequest

mllm = MLLMTask(
    model_id="Qwen/Qwen2-VL-7B-Instruct",
    # model_id="google/gemma-3-4b-it",
    modalities=[Modality.IMAGE],
)


class Config(AppConfig):
    """App configuration model."""

    tasks = {"mllm": mllm}


async def serve(request: OpenAIChatCompletionRequest) -> AsyncIterator[OpenAIChatCompletionChunk]:
    """Main serve function for the app."""
    return await mllm(request)
