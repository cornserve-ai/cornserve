"""An app that runs a DummyImageGeneratorTask for standalone Geri benchmarking.

This app invokes Geri in dummy mode, where the executor generates random tensors
instead of waiting for real embeddings from the Sidecar.

```console
$ cornserve register examples/dummy_image_generator.py

$ cornserve invoke dummy_image_generator --data - <<EOF
dummy_seq_len: 256
height: 512
width: 512
num_inference_steps: 20
EOF
```
"""

from __future__ import annotations

from cornserve_tasklib.task.unit.generator import (
    DummyGeneratorOutput,
    DummyImageGeneratorInput,
    DummyImageGeneratorTask,
)

from cornserve.app.base import AppConfig

MODEL_ID = "Qwen/Qwen-Image"

geri = DummyImageGeneratorTask(
    model_id=MODEL_ID,
    max_batch_size=1,
    sp_size=2,
)


class Config(AppConfig):
    """App configuration model."""

    tasks = {"geri": geri}


async def serve(request: DummyImageGeneratorInput) -> DummyGeneratorOutput:
    """Main serve function for the app."""
    return await geri(request)
