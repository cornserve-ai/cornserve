"""An app that generates text or audio using Qwen/Qwen3-Omni-30B-A3B-Instruct model.

For streaming audio generation, the returning audio chunk is an ChatCompletionAudio object
within each OpenAIChatCompletionChunk. The audio data is PCM16 based64 encoded, and can be
aggregated through `choices.[].delta.audio.data`.

```console
$ cornserve register examples/qwen3_omni.py

$ cornserve invoke qwen3_omni --audio-key choices.0.delta.audio.data --data - <<EOF
model: "Qwen/Qwen3-Omni-30B-A3B-Instruct"
messages:
- role: "user"
  content:
  - type: text
    text: "Describe this multimodal serving system architecture in one short sentence."
  - type: video_url
    video_url:
      url: "https://github.com/cornserve-ai/cornserve/raw/refs/heads/master/docs/assets/video/cornserve.mp4"
return_audio: true
EOF

$ cornserve invoke qwen3_omni --aggregate-keys choices.0.delta.content --data - <<EOF
model: "Qwen/Qwen3-Omni-30B-A3B-Instruct"
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
return_audio: false
EOF
```
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from cornserve_tasklib.task.composite.omni import OmniInput, OmniTask
from cornserve_tasklib.task.unit.encoder import Modality
from cornserve_tasklib.task.unit.llm import OpenAIChatCompletionChunk

from cornserve.app.base import AppConfig

omni = OmniTask(
    model_id="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    modalities=[Modality.IMAGE, Modality.VIDEO, Modality.AUDIO],
    encoder_fission=False,
)


class Config(AppConfig):
    """App configuration model."""

    tasks = {"omni": omni}


async def serve(request: OmniInput) -> AsyncIterator[OpenAIChatCompletionChunk]:
    """Main serve function for the app."""
    return await omni(request)
