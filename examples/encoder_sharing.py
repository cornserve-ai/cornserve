"""An app that shares an encoder task across multiple LLM tasks.

Ensure you have a GPU large enough to run the 27B model. If not, remove it from the `gemma_model_ids` list.

The request can specify any of the following model IDs and the encoder will be shared:
- google/gemma-3-4b-it
- google/gemma-3-12b-it
- google/gemma-3-27b-it

```console
# Gemma models are gated.
$ kubectl create -n cornserve secret generic cornserve-env --from-literal=hf-token=$HF_TOKEN

$ cornserve register examples/encoder_sharing.py

# `model` can be any of the model IDs in `gemma_model_ids`.
$ cornserve invoke encoder_sharing --aggregate-keys choices.0.delta.content --data - <<EOF
model: google/gemma-3-4b-it
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
EOF
```
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from cornserve.app.base import AppConfig
from cornserve.task.base import Task
from cornserve.tasklib.task.builtins.llm import MLLMTask, Modality, OpenAIChatCompletionChunk, OpenAIChatCompletionRequest

gemma_model_ids = {
    "gemma3-4b": "google/gemma-3-4b-it",
    "gemma3-12b": "google/gemma-3-12b-it",
    # "gemma3-27b": "google/gemma-3-27b-it",
}


# All tasks below will share the same encoder task and deployment.
gemma_tasks: dict[str, Task] = {
    name: MLLMTask(
        modalities=[Modality.IMAGE],
        model_id=model_id,
        encoder_model_ids=set(gemma_model_ids.values()),
    )
    for name, model_id in gemma_model_ids.items()
}


class Config(AppConfig):
    """App configuration model."""

    tasks = gemma_tasks


async def serve(request: OpenAIChatCompletionRequest) -> AsyncIterator[OpenAIChatCompletionChunk]:
    """Main serve function for the app."""
    match request.model:
        case "google/gemma-3-4b-it":
            return await gemma_tasks["gemma3-4b"](request)
        case "google/gemma-3-12b-it":
            return await gemma_tasks["gemma3-12b"](request)
        case "google/gemma-3-27b-it":
            return await gemma_tasks["gemma3-27b"](request)
        case default:
            raise ValueError(
                f"Unsupported model ID: {default}. Supported models are: {gemma_model_ids}.",
            )
