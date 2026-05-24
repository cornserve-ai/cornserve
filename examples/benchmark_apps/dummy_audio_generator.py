"""An app that runs a DummyAudioGeneratorTask for standalone Geri benchmarking.

This app invokes Geri in dummy mode, where the executor generates random tensors
instead of waiting for real embeddings from the Sidecar.

Since the embeddings are random, the generated audio will be garbage noise, and it
is dropped by the task dispatcher instead of being returned.

```console
$ cornserve register examples/dummy_audio_generator.py

$ cornserve invoke dummy_audio_generator --data - <<EOF
dummy_seq_len: 20000
EOF
```
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from cornserve_tasklib.task.unit.generator import (
    DummyAudioGeneratorInput,
    DummyAudioGeneratorTask,
)
from cornserve_tasklib.task.unit.llm import OpenAIChatCompletionChunk

from cornserve.app.base import AppConfig

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

geri = DummyAudioGeneratorTask(
    model_id=MODEL_ID,
    max_batch_size=1,
)


class Config(AppConfig):
    """App configuration model."""

    tasks = {"geri": geri}


async def serve(
    request: DummyAudioGeneratorInput,
) -> AsyncIterator[OpenAIChatCompletionChunk]:
    """Main serve function for the app."""
    return await geri(request)
