"""Stream transformation.

This example extends `examples/mllm.py` by showing how
a `Stream[T]` can be transformed into `Stream[U]` using the `transform` method.

```console
$ cornserve register examples/stream_transform.py

$ cornserve invoke stream_transform --aggregate-keys response --data - <<EOF
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
```
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from cornserve_tasklib.task.composite.llm import MLLMTask
from cornserve_tasklib.task.unit.encoder import Modality
from cornserve_tasklib.task.unit.llm import (
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequest,
)

from cornserve.app.base import AppConfig
from cornserve.task.base import TaskOutput

mllm = MLLMTask(
    model_id="Qwen/Qwen2-VL-7B-Instruct",
    # model_id="google/gemma-3-4b-it",
    modalities=[Modality.IMAGE],
)


class Config(AppConfig):
    """App configuration model."""

    tasks = {"mllm": mllm}


class OutputChunk(TaskOutput):
    """Output chunk model for the stream transformation."""

    response: str


def to_chunk(chunk: OpenAIChatCompletionChunk) -> OutputChunk:
    """Transform OpenAIChatCompletionChunk to OutputChunk."""
    content = chunk.choices[0].delta.content or ""
    return OutputChunk(response=content)


async def serve(request: OpenAIChatCompletionRequest) -> AsyncIterator[OutputChunk]:
    """Main serve function for the app."""
    stream = await mllm(request)
    return stream.transform(to_chunk)
