"""An app that uses Qwen 2.5 Omni via HuggingFace transformers.

```console
$ cornserve register examples/qwen_omni_huggingface.py

$ cornserve invoke qwen_omni_huggingface --aggregate-keys audio_chunk text_chunk --data - <<EOF
model: "Qwen/Qwen2.5-Omni-7B"
messages:
- role: "system"
  content: "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
- role: "user"
  content:
  - type: text
    text: "Hello, can you introduce yourself and your capabilities?"
return_audio: true
EOF
```
"""  # noqa: E501

from __future__ import annotations

from cornserve.app.base import AppConfig
from cornserve_tasklib.task.unit.huggingface import (
    HuggingFaceQwenOmniInput,
    HuggingFaceQwenOmniOutput,
    HuggingFaceQwenOmniTask,
)

# Create the HuggingFace Qwen 2.5 Omni task
qwen_omni = HuggingFaceQwenOmniTask(model_id="Qwen/Qwen2.5-Omni-7B")


class Config(AppConfig):
    """App configuration model."""

    tasks = {"qwen_omni": qwen_omni}


async def serve(request: HuggingFaceQwenOmniInput) -> HuggingFaceQwenOmniOutput:
    """Main serve function for the app."""
    return await qwen_omni(request)
