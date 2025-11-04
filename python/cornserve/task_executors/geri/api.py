"""API schema for Geri."""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, RootModel

from cornserve.task_executors.geri.schema import (
    AudioEngineRequest,
    BatchEngineRequest,
    ImageEngineRequest,
    Status,
    StreamEngineRequest,
)


class Modality(enum.StrEnum):
    """Modality of the content to be generated."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


# ---------------- Base Geri request classes -----------------


class BatchGeriRequest(BaseModel, ABC):
    """An API request to generate batched content.

    Attributes:
        embedding_data_id: Sidecar data ID for the audio codes.

    Modality-specific generation request classes (e.g., ImageGeriRequest)
    that support batched generation should inherit from this class.

    Given that API requests must eventually be parsed into Engine requests,
    subclasses must define how to produce a corresponding BatchEngineRequest.
    """

    embedding_data_id: str

    @abstractmethod
    def to_batch_engine_request(self, request_id: str, span_context: dict[str, str] | None) -> BatchEngineRequest:
        """Produce a BatchEngineRequest."""


class StreamGeriRequest(BaseModel, ABC):
    """An API request to generate streamed content.

    Attributes:
        embedding_data_id: Sidecar data ID for the audio codes.

    Modality-specific generation request classes (e.g., AudioGeriRequest)
    that support streamed generation should inherit from this class.

    Given that API requests must eventually be parsed into Engine requests,
    subclasses must define how to produce a corresponding StreamEngineRequest.
    """

    embedding_data_id: str

    @abstractmethod
    def to_stream_engine_request(self, request_id: str, span_context: dict[str, str] | None) -> StreamEngineRequest:
        """Produce a StreamEngineRequest."""


# ---------- Modality specific generation request classes ----------


class ImageGeriRequest(BatchGeriRequest):
    """Request to generate image content.

    Attributes:
        height: Height of the generated content in pixels.
        width: Width of the generated content in pixels.
        num_inference_steps: Number of denoising steps to perform.
        skip_tokens: Number of initial tokens to skip from the embeddings.
    """

    height: int
    width: int
    num_inference_steps: int
    skip_tokens: int = 0

    def to_batch_engine_request(self, request_id: str, span_context: dict[str, str] | None) -> ImageEngineRequest:
        """Produce an ImageEngineRequest."""
        return ImageEngineRequest(
            request_id=request_id,
            height=self.height,
            width=self.width,
            num_inference_steps=self.num_inference_steps,
            embedding_data_id=self.embedding_data_id,
            skip_tokens=self.skip_tokens,
            span_context=span_context,
        )


class AudioGeriRequest(StreamGeriRequest):
    """Request to generate audio content.

    Attributes:
        chunk_size: number of codes to be processed at a time. If not supplied, the default for
            the loaded model will be used.
        left_context_size: number of codes immediately prior to each chunk to be processed as
            context. If not supplied, the default for the loaded model will be used.
    """

    chunk_size: int | None = None
    left_context_size: int | None = None

    def to_stream_engine_request(self, request_id: str, span_context: dict[str, str] | None) -> AudioEngineRequest:
        """Produce an AudioEngineRequest."""
        return AudioEngineRequest(
            request_id=request_id,
            embedding_data_id=self.embedding_data_id,
            chunk_size=self.chunk_size,
            left_context_size=self.left_context_size,
            span_context=span_context,
        )


# ---------------------- Response classes ----------------------


class BatchGeriResponse(BaseModel):
    """Response containing the full generated content.

    Attributes:
        status: Status of the generation operation.
        generated: Base64 encoded bytes of the generated content, if successful.
            Bytes are in PNG format for images.
        error_message: Error message if the status is ERROR.
    """

    status: Status
    generated: str | None = None
    error_message: str | None = None


class StreamGeriResponseChunk(RootModel[bytes]):
    """Response containing a chunk of generated streaming data.

    StreamGeriResponseChunk is meant to carry an individual
    unit of data for a streamed response.
    """

    model_config = ConfigDict(
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )
