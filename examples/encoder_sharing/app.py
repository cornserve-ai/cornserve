"""An app that shares an encoder task across multiple LLM tasks.

The request can specify any of the following model IDs and the encoder will be shared:
- google/gemma-3-4b-it
- google/gemma-3-12b-it
- google/gemma-3-27b-it
"""

from __future__ import annotations

from cornserve.app.base import AppRequest, AppResponse, AppConfig
from cornserve.task.builtins.mllm import MLLMInput, MLLMTask, Modality


class Request(AppRequest):
    """App request model.

    Attributes:
        prompt: The prompt to send to the LLM.
        model_id: The model ID to use for the task.
        multimodal_data: List of tuples (modality, data URL).
            "image", "video", etc. for modality.
        max_completion_tokens: Max number of tokens to generate in the response.
        seed: Optional random seed.
    """

    prompt: str
    model_id: str
    multimodal_data: list[tuple[str, str]] = []
    max_completion_tokens: int | None = None
    seed: int | None = None


class Response(AppResponse):
    """App response model.

    Attributes:
        response: The response from the LLM.
    """

    response: str


mllm = MLLMTask(
    modalities=[Modality.IMAGE],
    model_id="google/gemma-3-4b-it",
    adapter_model_ids=["google/gemma-3-12b-it", "google/gemma-3-27b-it"],
)


class Config(AppConfig):
    """App configuration model."""

    tasks = {"mllm": mllm}


async def serve(request: Request) -> Response:
    """Main serve function for the app."""
    mllm_input = MLLMInput(
        prompt=request.prompt,
        multimodal_data=request.multimodal_data,
        model_id=request.model_id,
        max_completion_tokens=request.max_completion_tokens,
        seed=request.seed,
    )
    mllm_output = await mllm(mllm_input)
    return Response(response=mllm_output.response)
