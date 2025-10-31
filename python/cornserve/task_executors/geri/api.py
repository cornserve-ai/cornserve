"""API schema for Geri."""

from __future__ import annotations

import enum

from pydantic import BaseModel, ConfigDict, RootModel


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


class AudioGenerationRequest(BaseModel):
    """Request to generate audio content.

    Attributes:
        embedding_data_id: Sidecar data ID for the audio codes.
        chunk_size: number of codes to be processed at a time. If not supplied, the default for
            the loaded model will be used.
        left_context_size: number of codes immediately prior to each chunk to be processed as
            context. If not supplied, the default for the loaded model will be used.
    """

    embedding_data_id: str
    chunk_size: int | None = None
    left_context_size: int | None = None


class Status(enum.IntEnum):
    """Status of various operations."""

    SUCCESS = 0
    ERROR = 1
    FINISHED = 2


class GenerationResponse(BaseModel):
    """Response containing the generated content.

    Attributes:
        status: Status of the generation operation.
        generated: Base64 encoded bytes of the generated content, if successful.
            Bytes are in PNG format for images.
        error_message: Error message if the status is ERROR.
    """

    status: Status
    generated: str | None = None
    error_message: str | None = None


class AudioChunk(RootModel[bytes]):
    """Response containing a chunk of generated audio data."""

    model_config = ConfigDict(
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )
