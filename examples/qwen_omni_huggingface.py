"""An app that uses Qwen 2.5 Omni via HuggingFace transformers.

```console
$ cornserve register examples/qwen_omni_huggingface.py

$ cornserve invoke qwen_omni --aggregate-keys audio_chunk text_chunk --data - <<EOF
model: "Qwen/Qwen2.5-Omni-7B"
messages:
- role: "user"
  content:
  - type: text
    text: "Hello, can you introduce yourself?"
return_audio: true
EOF

$ cornserve invoke qwen_omni --aggregate-keys text_chunk --data - <<EOF
model: "Qwen/Qwen2.5-Omni-7B"
messages:
- role: "user"
  content:
  - type: text
    text: "What is the capital of France?"
return_audio: false
EOF
```
"""

from __future__ import annotations

from cornserve.app.base import AppConfig
from cornserve.task.base import Stream
from cornserve.task.builtins.huggingface import (
    HuggingFaceQwenOmniInput,
    HuggingFaceQwenOmniOutput,
    HuggingFaceQwenOmniTask,
)

# Create the HuggingFace Qwen 2.5 Omni task
qwen_omni = HuggingFaceQwenOmniTask(model_id="Qwen/Qwen2.5-Omni-7B")


class Config(AppConfig):
    """App configuration model."""

    tasks = {"qwen_omni": qwen_omni}


async def serve(request: HuggingFaceQwenOmniInput) -> Stream[HuggingFaceQwenOmniOutput]:
    """Main serve function for the app."""
    return await qwen_omni(request)
