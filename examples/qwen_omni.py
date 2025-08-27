"""An app that generates text or audio using Qwen2.5-Omni-7B model.

```console
$ cornserve register examples/qwen_omni.py

$ cornserve invoke qwen_omni --data - <<EOF
model: "Qwen/Qwen2.5-Omni-7B"
messages:
- role: "user"
  content:
  - type: text
    text: "Descrive what you see and hear"
  - type: video_url
    video_url:
      url: "https://dedicated.junzema.com/draw.mp4"
return_audio: true
EOF

$ cornserve invoke qwen_omni --aggregate-keys text_chunk.choices.0.delta.content --data - <<EOF
model: "Qwen/Qwen2.5-Omni-7B"
messages:
- role: "user"
  content:
  - type: text
    text: "Descrive what you see and hear"
  - type: video_url
    video_url:
      url: "https://dedicated.junzema.com/draw.mp4"
return_audio: false
EOF
```
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from cornserve.app.base import AppConfig
from cornserve.task.builtins.encoder import Modality
from cornserve.task.builtins.omni import OmniInput, OmniOutputChunk, OmniTask

omni = OmniTask(
    model_id="Qwen/Qwen2.5-Omni-7B",
    modalities=[Modality.VIDEO],
)


class Config(AppConfig):
    """App configuration model."""

    tasks = {"omni": omni}


async def serve(request: OmniInput) -> AsyncIterator[OmniOutputChunk]:
    """Main serve function for the app."""
    return await omni(request)
