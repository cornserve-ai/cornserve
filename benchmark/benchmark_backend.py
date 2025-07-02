from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from cornserve.services.gateway.models import AppInvocationRequest

from benchmark_dataset import SampleRequest


class Backend(enum.StrEnum):
    """Backend type for the benchmark."""
    CORNSERVE = "cornserve"
    ERIC = "eric"
    VLLM = "vllm"

@dataclass
class RequestInput:
    """Input for the benchmark request."""
    backend: Backend
    url: str
    payload: dict[str, Any]

@dataclass
class RequestOutput:
    """Output for the benchmark request."""
    success: bool = False
    latency: float = 0.0
    error: str = ""
    output: Any = None

def build_cornserve_vlm_input(
    base_url: str,
    app_id: str,
    model_id: str,
    sampled_request: SampleRequest,
    use_sampled_mm_data: bool = True,
    video_urls: list[str] = [],
    audio_urls: list[str] = [],
    image_urls: list[str] = [],
) -> RequestInput:
    """Builds the input for a Cornserve VLM app invocation.

    Args:
        base_url: The base URL of the Cornserve gateway.
        app_id: The ID of the app to invoke.
        model_id: Not used in this function, but included for consistency.
        sampled_request: The sampled request containing prompt and expected output length.
        use_sampled_mm_data: Whether to use the sampled multi-modal data or provided URLs.
        video_urls: List of overwritting video URLs.
        audio_urls: List of overwritting audio URLs.
        image_urls: List of overwritting image URLs.
    """
    data={"prompt": sampled_request.prompt, "multimodal_data": []}
    if use_sampled_mm_data:
        data["multimodal_data"].append(("image", sampled_request.multi_modal_data["image_url"]["url"]))
    else:
        for u in video_urls:
            data["multimodal_data"].append(("video", u))
        for u in audio_urls:
            data["multimodal_data"].append(("audio", u))
        for u in image_urls:
            data["multimodal_data"].append(("image", u))
    data["max_completion_tokens"] = sampled_request.expected_output_len

    api_url = f"{base_url}/app/invoke/{app_id}"
    request = AppInvocationRequest(request_data=data)
    return RequestInput(backend=Backend.CORNSERVE, url=api_url, payload=request.model_dump())

def build_vllm_input(
    base_url: str,
    app_id: str,
    model_id: str,
    sampled_request: SampleRequest,
    use_sampled_mm_data: bool = True,
    video_urls: list[str] = [],
    audio_urls: list[str] = [],
    image_urls: list[str] = [],
) -> RequestInput:
    """Builds the input for a request to vLLM.

    Args:
        base_url: The base URL of the vLLM server.
        app_id: Not used in this function, but included for consistency.
        model_id: The ID of the model to use.
        sampled_request: The sampled request containing prompt and expected output length.
        use_sampled_mm_data: Whether to use the sampled multi-modal data or provided URLs.
        video_urls: List of overwriting video URLs.
        audio_urls: List of overwriting audio URLs.
        image_urls: List of overwriting image URLs.
    """
    api_url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type" : "text", "text": sampled_request.prompt},
                ]
            },
        ],
        "max_completion_tokens": sampled_request.expected_output_len,
    }
    if use_sampled_mm_data:
        payload["messages"][0]["content"].append(sampled_request.multi_modal_data)
    else:
        for url in video_urls:
            payload["messages"][0]["content"].append({"type": "video_url", "video_url": url})
        for url in audio_urls:
            payload["messages"][0]["content"].append({"type": "audio_url", "audio_url": url})
        for url in image_urls:
            payload["messages"][0]["content"].append({"type": "image_url", "image_url": url})
    return RequestInput(backend=Backend.VLLM, url=api_url, payload=payload)


def build_eric_input(
    base_url: str,
    app_id: str,
    model_id: str,
    sampled_request: SampleRequest,
    use_sampled_mm_data: bool = True,
    video_urls: list[str] = [],
    audio_urls: list[str] = [],
    image_urls: list[str] = [],
) -> RequestInput:
    """Builds the input for a request to Eric.

    Args:
        base_url: The base URL of the Eric server.
        app_id: Not used in this function, but included for consistency.
        model_id: Not used in this function, but included for consistency.
        sampled_request: The sampled request containing prompt and expected output length.
        use_sampled_mm_data: Whether to use the sampled multi-modal data or provided URLs.
        video_urls: List of overwriting video URLs.
        audio_urls: List of overwriting audio URLs.
        image_urls: List of overwriting image URLs.
    """
    api_url = f"{base_url}/embeddings"

    payload = {"data": []}
    if use_sampled_mm_data:
        url = sampled_request.multi_modal_data["image_url"]["url"]
        payload["data"].append({"id": uuid.uuid4().hex, "modality": "image", "url": url})
    else:
        for url in video_urls:
            payload["data"].append({"id": uuid.uuid4().hex, "modality": "video", "url": url})
        for url in audio_urls:
            payload["data"].append({"id": uuid.uuid4().hex, "modality": "audio", "url": url})
        for url in image_urls:
            payload["data"].append({"id": uuid.uuid4().hex, "modality": "image", "url": url})
    return RequestInput(backend=Backend.ERIC, url=api_url, payload=payload)

TRANSFORM_FUNCS: dict[str, Callable] = {
    "cornserve": build_cornserve_vlm_input,
    "eric": build_eric_input,
    "vllm": build_vllm_input,
}
