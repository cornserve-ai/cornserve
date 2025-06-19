"""An app that runs a Multimodal LLM task."""

from __future__ import annotations

from cornserve.app.base import AppRequest, AppResponse, AppConfig
from cornserve.task.builtins.encoder import Modality
from cornserve.task.builtins.omni import OmniTask, OmniInput

class Request(AppRequest):
    """App request model.

    Attributes:
        prompt: The prompt to send to the LLM.
        multimodal_data: List of tuples (modality, data URL).
            "image", "video", etc. for modality.
    """

    prompt: str
    multimodal_data: list[tuple[str, str]] = []
    return_audio: bool


class Response(AppResponse):
    """App response model.

    Attributes:
        response: The response from the LLM.
    """

    response: str | bytes



omni = OmniTask(
    model_id="Qwen/Qwen2.5-Omni-7B",
    modalities=[Modality.AUDIO, Modality.VIDEO],
)


class Config(AppConfig):
    """App configuration model."""

    tasks = {"omni": omni}


async def serve(request: Request) -> Response:
    """Main serve function for the app."""
    omni_input = OmniInput(
        prompt=request.prompt,
        multimodal_data=request.multimodal_data,
        return_audio=request.return_audio,
    )
    omni_output = await omni(omni_input)
    return Response(response=omni_output.response)
