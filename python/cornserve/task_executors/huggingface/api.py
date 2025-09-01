"""API schema for HuggingFace task executor."""

from __future__ import annotations

import enum

from pydantic import BaseModel

from cornserve.task.builtins.llm import StreamOptions


class TaskType(enum.StrEnum):
    """Task type for HuggingFace executor."""

    QWEN_IMAGE = "qwen-image"
    QWEN_OMNI = "qwen-omni"


class HuggingFaceRequest(BaseModel):
    """Request to HuggingFace task executor.

    Attributes:
        task_type: Type of task to execute.
        model_id: Model ID to use.

        # Qwen-Image fields
        prompt: Text prompt for image generation.
        height: Height of generated image in pixels.
        width: Width of generated image in pixels.
        num_inference_steps: Number of denoising steps.

        # Qwen-Omni fields (OpenAI chat completion compatible)
        messages: List of chat messages.
        return_audio: Whether to return audio response.
        frequency_penalty: Frequency penalty for generation.
        max_completion_tokens: Maximum completion tokens.
        presence_penalty: Presence penalty for generation.
        seed: Random seed for generation.
        stream_options: Streaming options.
        temperature: Temperature for generation.
        top_p: Top-p for generation.
        request_id: Unique request identifier.
    """

    task_type: TaskType
    model_id: str

    # Qwen-Image fields
    prompt: str | None = None
    height: int | None = None
    width: int | None = None
    num_inference_steps: int | None = None

    # Qwen-Omni fields
    messages: list[dict] | None = None
    return_audio: bool | None = None
    frequency_penalty: float | None = None
    max_completion_tokens: int | None = None
    presence_penalty: float | None = None
    seed: int | None = None
    stream_options: StreamOptions | None = None
    temperature: float | None = None
    top_p: float | None = None
    request_id: str | None = None


class Status(enum.IntEnum):
    """Status of operations."""

    SUCCESS = 0
    ERROR = 1


class HuggingFaceResponse(BaseModel):
    """Response from HuggingFace task executor.

    Attributes:
        status: Status of the operation.

        # Qwen-Image response
        image: Base64-encoded PNG image.

        # Qwen-Omni response (streaming chunks)
        audio_chunk: Base64-encoded audio chunk of np.float32 raw waveform.
        text_chunk: OpenAI chat completion chunk as dict.

        error_message: Error message if status is ERROR.
    """

    status: Status

    # Qwen-Image response
    image: str | None = None

    # Qwen-Omni response
    audio_chunk: str | None = None
    text_chunk: dict | None = None

    error_message: str | None = None
