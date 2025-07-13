"""An app that shares an encoder task across multiple LLM tasks.

The request can specify any of the following model IDs and the encoder will be shared:
- google/gemma-3-4b-it
- google/gemma-3-12b-it
- google/gemma-3-27b-it
"""

from __future__ import annotations

from cornserve.app.base import AppConfig
from cornserve.task.base import Stream
from cornserve.task.builtins.llm import MLLMTask, Modality, OpenAIChatCompletionChunk, OpenAIChatCompletionRequest

gemma_model_ids = [
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "google/gemma-3-27b-it",
]


# All three tasks below will share the same encoder task and deployment.
gemma4b = MLLMTask(
    modalities=[Modality.IMAGE],
    model_id="google/gemma-3-4b-it",
    encoder_model_ids=set(gemma_model_ids),
)
gemma12b = MLLMTask(
    modalities=[Modality.IMAGE],
    model_id="google/gemma-3-12b-it",
    encoder_model_ids=set(gemma_model_ids),
)
gemma27b = MLLMTask(
    modalities=[Modality.IMAGE],
    model_id="google/gemma-3-27b-it",
    encoder_model_ids=set(gemma_model_ids),
)


class Config(AppConfig):
    """App configuration model."""

    tasks = {"gemma4b": gemma4b, "gemma12b": gemma12b, "gemma27b": gemma27b}


async def serve(request: OpenAIChatCompletionRequest) -> Stream[OpenAIChatCompletionChunk]:
    """Main serve function for the app."""
    match request.model:
        case "google/gemma-3-4b-it":
            return await gemma4b(request)
        case "google/gemma-3-12b-it":
            return await gemma12b(request)
        case "google/gemma-3-27b-it":
            return await gemma27b(request)
        case default:
            raise ValueError(
                f"Unsupported model ID: {default}. Supported models are: {gemma_model_ids}.",
            )
