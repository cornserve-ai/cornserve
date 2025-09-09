"""An app that runs a Multimodal LLM task.

```console
$ cornserve register examples/mllm.py

$ cornserve invoke mllm --aggregate-keys choices.0.delta.content --data - <<EOF
model: "Qwen/Qwen2.5-VL-7B-Instruct"
messages:
- role: "user"
  content:
  - type: text
    text: "Hello Write a poem about the images you see."
  - type: image_url
    image_url:
      url: "https://picsum.photos/id/12/480/560"
EOF
  - type: image_url
    image_url:
      url: "https://picsum.photos/id/234/960/960"

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

```
curl -X POST "localhost:30080/app/invoke/app-730cb2d18d4f4d5680b847e006b0b382" \
  -H "Content-Type: application/json" \
  -d '{{
    "model": "OpenGVLab/InternVL3-38B",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Write a poem about the images you see."
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://picsum.photos/id/12/480/560"
            }
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "https://picsum.photos/id/234/960/960"
            }
          }
        ]
      }
    ],
    "temperature": 0.0,
    "max_completion_tokens": 100,
    "stream_options": {
      "include_usage": true
    },
    "ignore_eos": false
  }'

```
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from cornserve.app.base import AppConfig
from cornserve.task.builtins.encoder import Modality
from cornserve.task.builtins.llm import NcclDisaggregatedMLLMTask, OpenAIChatCompletionChunk, OpenAIChatCompletionRequest

mllm = NcclDisaggregatedMLLMTask(
    model_id="Qwen/Qwen2.5-VL-7B-Instruct",
    # model_id="google/gemma-3-4b-it",
    modalities=[Modality.IMAGE],
    encoder_fission=False,
)


class Config(AppConfig):
    """App configuration model."""

    tasks = {"mllm": mllm}


async def serve(request: OpenAIChatCompletionRequest) -> AsyncIterator[OpenAIChatCompletionChunk]:
    """Main serve function for the app."""
    return await mllm(request)
