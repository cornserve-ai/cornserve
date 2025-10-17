"""An app that generates images using Qwen-Image model.

```console
$ cornserve register examples/qwen_image_huggingface.py

$ cornserve invoke qwen_image_huggingface --data - <<EOF
prompt: "A beautiful landscape with mountains and a lake at sunset"
height: 512
width: 512
num_inference_steps: 20
EOF

$ cornserve invoke qwen_image_huggingface --data - <<EOF
prompt: "A cute robot playing with a cat in a futuristic city"
height: 1024
width: 1024
num_inference_steps: 50
EOF
```
"""

from __future__ import annotations

from cornserve_tasklib.task.unit.huggingface import (
    HuggingFaceQwenImageInput,
    HuggingFaceQwenImageOutput,
    HuggingFaceQwenImageTask,
)

from cornserve.app.base import AppConfig

qwen_image = HuggingFaceQwenImageTask(model_id="Qwen/Qwen-Image")


class Config(AppConfig):
    """App configuration model."""

    tasks = {"qwen_image": qwen_image}


async def serve(request: HuggingFaceQwenImageInput) -> HuggingFaceQwenImageOutput:
    """Main serve function for the app."""
    return await qwen_image(request)
