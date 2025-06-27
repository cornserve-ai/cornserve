"""An app that runs a Multimodal LLM task."""

from __future__ import annotations

from cornserve.app.base import AppRequest, AppResponse, AppConfig
from cornserve.task.builtins.mllm import MLLMInput, MLLMTask, Modality


class Request(AppRequest):
    """App request model.

    Attributes:
        prompt: The prompt to send to the LLM.
        multimodal_data: List of tuples (modality, data URL).
            "image", "video", etc. for modality.
    """

    prompt: str
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
    model_id="Qwen/Qwen2-VL-7B-Instruct",
    # model_id="google/gemma-3-4b-it",
    modalities=[Modality.IMAGE],
)


class Config(AppConfig):
    """App configuration model."""

    tasks = {"mllm": mllm}


async def serve(request: Request) -> Response:
    """Main serve function for the app."""
    mllm_input = MLLMInput(
        prompt=request.prompt,
        multimodal_data=request.multimodal_data,
        max_completion_tokens=request.max_completion_tokens,
        seed=request.seed,
    )
    mllm_output = await mllm(mllm_input)
    return Response(response=mllm_output.response)
