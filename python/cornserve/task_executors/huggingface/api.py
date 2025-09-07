"""API schema for HuggingFace task executor."""

from __future__ import annotations

import enum

from pydantic import BaseModel

from cornserve.task.builtins.llm import StreamOptions


class ModelType(enum.StrEnum):
    """Model type for HuggingFace executor."""

    QWEN_IMAGE = "QWEN_IMAGE"
    QWEN_OMNI = "QWEN_OMNI"


class HuggingFaceRequest(BaseModel):
    """Request to HuggingFace task executor."""

    # Qwen-Image fields
    prompt: str | None = None
    height: int | None = None
    width: int | None = None
    num_inference_steps: int | None = None

    # Qwen-Omni fields
    messages: list[dict] | None = None
    model: str
    frequency_penalty: float | None = 0.0
    max_completion_tokens: int | None = None
    presence_penalty: float | None = 0.0
    seed: int | None = None
    stream_options: StreamOptions | None = None
    temperature: float | None = None
    top_p: float | None = None
    request_id: str
    ignore_eos: bool = False
    return_audio: bool | None = None


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
