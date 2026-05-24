"""Route Qwen2.5-VL-7B requests across mixed MLLM configs.

This app uses a weighted router over two MLLM paths:
1) tp=2, max_num_seqs=32, encoder_fission=False
2) tp=1, max_num_seqs=64, encoder_fission=True, eric_max_batch_size=1

Routing weights are [0.3, 0.7] respectively.

With the provided profile files, the app deploys onto 4 GPUs total:
- 2 GPUs for the tp=2 LLM task
- 1 GPU for the tp=1 LLM task
- 1 GPU for the image Eric task

Usage:
    cornserve deploy_profiles profiles
    cornserve register examples/qwen25vl_router.py
    cornserve invoke qwen25vl_router --aggregate-keys choices.0.delta.content --data - <<EOF
    model: "Qwen/Qwen2.5-VL-7B-Instruct"
    messages:
    - role: "user"
      content:
      - type: text
        text: "Describe the image briefly."
      - type: image_url
        image_url:
          url: "https://picsum.photos/id/237/512/512"
    EOF
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from cornserve_tasklib.task.composite.llm import MLLMTask
from cornserve_tasklib.task.composite.router import RouterApp
from cornserve_tasklib.task.unit.encoder import Modality
from cornserve_tasklib.task.unit.llm import (
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequest,
)

from cornserve.app.base import AppConfig
from cornserve.task.base import Stream

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

mllm_tp2_bs32 = MLLMTask(
    model_id=MODEL_ID,
    modalities=[Modality.IMAGE],
    encoder_fission=False,
    eric_max_batch_size=1,
    llm_tp_size=2,
    llm_max_num_seqs=32,
    llm_gpu_memory_utilization=0.9,
)

mllm_tp1_bs64_fission = MLLMTask(
    model_id=MODEL_ID,
    modalities=[Modality.IMAGE],
    encoder_fission=True,
    eric_max_batch_size=1,
    llm_tp_size=1,
    llm_max_num_seqs=64,
    llm_gpu_memory_utilization=0.9,
)

qwen25vl_router = RouterApp[OpenAIChatCompletionRequest, Stream[OpenAIChatCompletionChunk]](
    routing_tasks=[mllm_tp2_bs32, mllm_tp1_bs64_fission],
    routing_weights=[0.3, 0.7],
)


class Config(AppConfig):
    """App configuration model."""

    tasks = {"qwen25vl_router": qwen25vl_router}


async def serve(
    request: OpenAIChatCompletionRequest,
) -> AsyncIterator[OpenAIChatCompletionChunk]:
    """Route each request and stream the selected MLLM response."""
    return await qwen25vl_router(request)
