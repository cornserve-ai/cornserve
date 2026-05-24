"""An app that runs a DummyOmniTalkerEmbeddingTask for standalone Talker benchmarking.

This app invokes the Talker LLM in dummy mode, where the executor generates
random thinker hidden states instead of waiting for real ones from the Sidecar.

Since the hidden states are random, the generated output will be garbage, and it
is discarded by the descriptor instead of being returned.


```console
$ cornserve register examples/dummy_talker.py


NOTE:
- ignore_eos is necessary because with random simulated input tensors, Talker produces
random outputs and that means EOS token can appear early, stopping the run.
- With ignore_eos enabled, max_completion_tokens is necessary to explicitly control
when decode should end.
- Prefill time is driven by the length of the prompt.
- thinker_hidden_states_len and thinker_output_len do not drive the workload amount.
- So, the actual workload of this simulated dummy task is decided by the combination of
(prompt size) + (max_completion_tokens).

$ cornserve invoke dummy_talker --data - <<EOF
model: "Qwen/Qwen3-Omni-30B-A3B-Instruct"
thinker_hidden_states_len: 50
thinker_output_len: 40
max_completion_tokens: 100
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

from cornserve_tasklib.task.unit.omni import (
    DummyOmniTalkerEmbeddingResponse,
    DummyOmniTalkerEmbeddingTask,
    DummyOmniTalkerInput,
)

from cornserve.app.base import AppConfig

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

talker = DummyOmniTalkerEmbeddingTask(
    model_id=MODEL_ID,
)


class Config(AppConfig):
    """App configuration model."""

    tasks = {"talker": talker}


async def serve(
    request: DummyOmniTalkerInput,
) -> DummyOmniTalkerEmbeddingResponse:
    """Main serve function for the app."""
    return await talker(request)
