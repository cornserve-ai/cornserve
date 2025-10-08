"""An app that lets users compare different Gemma models.

This example extends `examples/encoder_sharing.py` by showing how to stream responses
from all models simulataneously.

Ensure you have a GPU large enough to run the 27B model. If not, remove it from the `gemma_model_ids` list.

```console
# Gemma models are gated.
$ kubectl create -n cornserve secret generic cornserve-env --from-literal=hf-token=$HF_TOKEN

$ cornserve register examples/gemmarena.py
$ cornserve invoke gemmarena --aggregate-keys gemma3-4b gemma3-12b --data - <<EOF
model: gemmas
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
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from pydantic import RootModel

from cornserve.app.base import AppConfig
from cornserve.task.base import Stream
from cornserve_tasklib.task.unit.encoder import Modality
from cornserve_tasklib.task.unit.llm import OpenAIChatCompletionChunk, OpenAIChatCompletionRequest
from cornserve_tasklib.task.composite.llm import MLLMTask

gemma_model_ids = {
    "gemma3-4b": "google/gemma-3-4b-it",
    "gemma3-12b": "google/gemma-3-12b-it",
    # "gemma3-27b": "google/gemma-3-27b-it",
}


# All tasks below will share the same encoder task and deployment.
gemma_tasks = {
    name: MLLMTask(
        modalities=[Modality.IMAGE],
        model_id=model_id,
        encoder_model_ids=set(gemma_model_ids.values()),
    )
    for name, model_id in gemma_model_ids.items()
}


class ArenaOutput(RootModel[dict[str, str]]):
    """Output model for the arena."""


class Config(AppConfig):
    """App configuration model."""

    tasks = {**gemma_tasks}


async def serve(request: OpenAIChatCompletionRequest) -> AsyncIterator[ArenaOutput]:
    """Main serve function for the app."""
    # NOTE: Doing `await` for each task separately will make them run sequentially.
    tasks: list[asyncio.Task[Stream[OpenAIChatCompletionChunk]]] = []
    for task in gemma_tasks.values():
        # Overwrite the model ID in the request to match the task's model ID.
        request_ = request.model_copy(update={"model": task.model_id}, deep=True)
        tasks.append(asyncio.create_task(task(request_)))
    responses = await asyncio.gather(*tasks)

    streams = {
        asyncio.create_task(anext(stream)): (stream, name)
        for stream, name in zip(responses, gemma_model_ids.keys(), strict=True)
    }

    while streams:
        done, _ = await asyncio.wait(streams.keys(), return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            stream, name = streams.pop(task)

            try:
                chunk = task.result()
            except StopAsyncIteration:
                continue

            delta = chunk.choices[0].delta.content
            if delta is None:
                continue
            yield ArenaOutput({name: delta})

            streams[asyncio.create_task(anext(stream))] = (stream, name)
