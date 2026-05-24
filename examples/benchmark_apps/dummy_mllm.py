"""An app that runs a Multimodal LLM task.

```console
$ cornserve register examples/dummy_mllm.py

$ cornserve invoke dummy_mllm --aggregate-keys choices.0.delta.content --data - <<EOF
model: "Qwen/Qwen2.5-VL-32B-Instruct"
messages:
- role: "user"
  content:
  - type: text
    text: "Write a poem about the images you see."
  - type: image_url
    image_url:
      url: "https://picsum.photos/id/12/480/560"
  - type: image_url
    image_url:
      url: "https://picsum.photos/id/234/960/960"
EOF

$ cornserve invoke dummy_mllm --aggregate-keys choices.0.delta.content usage --data - <<EOF
model: "OpenGVLab/InternVL3-38B"
messages:
- role: "user"
  content:
  - type: text
    text: "Write a poem about the images you see in twenty words."
  - type: image_url
    image_url:
      url: "https://picsum.photos/id/12/480/560"
  - type: image_url
    image_url:
      url: "https://picsum.photos/id/234/960/960"
ignore_eos: true
max_completion_tokens: 500
stream_options:
  include_usage: true
EOF
```
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from cornserve_tasklib.task.unit.llm import DummyMLLMUnitTask, OpenAIChatCompletionChunk, OpenAIChatCompletionRequest

from cornserve.app.base import AppConfig

mllm = DummyMLLMUnitTask(
    model_id="OpenGVLab/InternVL3-38B",
    # model_id="Qwen/Qwen2.5-VL-32B-Instruct",
    tp_size=1,
    max_num_seqs=256,
    gpu_memory_utilization=0.9,
)


class Config(AppConfig):
    """App configuration model."""

    tasks = {"mllm": mllm}


async def serve(
    request: OpenAIChatCompletionRequest,
) -> AsyncIterator[OpenAIChatCompletionChunk]:
    """Main serve function for the app."""
    return await mllm(request)
