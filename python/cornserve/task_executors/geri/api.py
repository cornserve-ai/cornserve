"""API schema for Geri."""

from __future__ import annotations

import enum

from pydantic import BaseModel


class Modality(enum.StrEnum):
    """Modality of the content to be generated."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class GenerationRequest(BaseModel):
    """Request to generate multimodal content.

    Attributes:
        height: Height of the generated content in pixels.
        width: Width of the generated content in pixels.
        num_inference_steps: Number of denoising steps to perform.
        embedding_data_id: Sidecar data ID for the prompt embeddings.
        skip_tokens: Number of initial tokens to skip from the embeddings.
    """

    height: int
    width: int
    num_inference_steps: int
    embedding_data_id: str
    skip_tokens: int = 0


class Status(enum.IntEnum):
    """Status of various operations."""

    SUCCESS = 0
    ERROR = 1


class GenerationResponse(BaseModel):
    """Response containing the generated content.

    Attributes:
        status: Status of the generation operation.
        generated: Bytes of the generated content, if successful.
            Bytes are in PNG format for images.
        error_message: Error message if the status is ERROR.
    """

    status: Status
    generated: bytes | None = None
    error_message: str | None = None
