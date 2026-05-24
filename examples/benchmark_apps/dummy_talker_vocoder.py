"""An app that runs a DummyOmniTalkerVocoderTask for colocated Talker & Vocoder benchmarking.

This app invokes the Talker & Vocoder colocated in dummy mode, where the executor
generates random thinker hidden states instead of waiting for real ones from the
Sidecar. The generated audio output is garbage and is discarded by the descriptor.


```console
$ cornserve register examples/dummy_talker_vocoder.py


NOTE:
- ignore_eos is necessary because with random simulated input tensors, Talker produces
random outputs and that means EOS token can appear early, stopping the run.
- With ignore_eos enabled, max_completion_tokens is necessary to explicitly control
when decode should end.
- Prefill time is driven by the length of the prompt.
- thinker_hidden_states_len and thinker_output_len do not drive the workload amount.
- So, the actual workload of this simulated dummy task is decided by the combination of
(prompt size) + (max_completion_tokens).

$ cornserve invoke dummy_talker_vocoder --data - <<EOF
model: "Qwen/Qwen3-Omni-30B-A3B-Instruct"
thinker_hidden_states_len: 50
thinker_output_len: 40
max_completion_tokens: 150
ignore_eos: true
messages:
- role: "user"
  content:
  - type: text
    text: "Hello"
- role: "assistant"
  content: "<tts_pad>"
EOF

```
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from cornserve_tasklib.task.unit.llm import OpenAIChatCompletionChunk
from cornserve_tasklib.task.unit.omni import (
    DummyOmniTalkerInput,
    DummyOmniTalkerVocoderTask,
)

from cornserve.app.base import AppConfig

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

talker = DummyOmniTalkerVocoderTask(
    model_id=MODEL_ID,
)


class Config(AppConfig):
    """App configuration model."""

    tasks = {"talker": talker}


async def serve(
    request: DummyOmniTalkerInput,
) -> AsyncIterator[OpenAIChatCompletionChunk]:
    """Main serve function for the app."""
    return await talker(request)
