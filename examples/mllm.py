"""An app that runs a Multimodal LLM task.

```console
$ cornserve register examples/mllm.py

$ cornserve invoke mllm --aggregate-keys choices.0.delta.content --data - <<EOF
model: "Qwen/Qwen2-VL-7B-Instruct"
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

$ cornserve invoke mllm --aggregate-keys choices.0.delta.content usage --data - <<EOF
model: "Qwen/Qwen2-VL-7B-Instruct"
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
stream_options:
  include_usage: true
EOF
```
"""

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
