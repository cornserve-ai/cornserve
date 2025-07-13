"""An app that lets users compare different Gemma models.

This example extends `examples/encoder_sharing.py` by showing how to stream responses
from all models simulataneously.

```console
$ cornserve register examples/gemmarena.py
$ cornserve invoke gemmarena --aggregate-keys google/gemma-4-4b-it google/gemma-3-12b-it google/gemma-3-27b-it - <<EOF
-

"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from cornserve.app.base import AppConfig
from cornserve.task.builtins.llm import MLLMTask, Modality, OpenAIChatCompletionChunk, OpenAIChatCompletionRequest

gemma_model_ids = [
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
]

gemma4b = MLLMTask(
    modalities=[Modality.IMAGE],
    model_id="google/gemma-3-4b-it",
    encoder_model_ids=set(gemma_model_ids),
)

gemma12b = MLLMTask(
    modalities=[Modality.IMAGE],
    model_id="google/gemma-3-12b-it",
    encoder_model_ids=set(gemma_model_ids),
)

gemma27b = MLLMTask(
    modalities=[Modality.IMAGE],
    model_id="google/gemma-3-27b-it",
    encoder_model_ids=set(gemma_model_ids),
)


class Config(AppConfig):
    """App configuration model."""

    tasks = {"gemma4b": gemma4b, "gemma12b": gemma12b, "gemma27b": gemma27b}


async def serve(request: OpenAIChatCompletionRequest) -> AsyncIterator[dict[str, OpenAIChatCompletionChunk]]:
    """Main serve function for the app."""
    # NOTE: Doing `await` for each task separately will make them run sequentially.
    stream0, stream1, stream2 = await asyncio.gather(gemma4b(request), gemma12b(request), gemma27b(request))

    streams = {
        asyncio.create_task(anext(stream0)): (stream0, "Gemma 3 4B"),
        asyncio.create_task(anext(stream1)): (stream1, "Gemma 3 12B"),
        asyncio.create_task(anext(stream2)): (stream2, "Gemma 3 27B"),
    }

    while streams:
        done, _ = await asyncio.wait(streams.keys(), return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            stream, name = streams.pop(task)

            try:
                chunk = task.result()
            except StopAsyncIteration:
                continue

            yield {name: chunk}

            streams[asyncio.create_task(anext(stream))] = (stream, name)
