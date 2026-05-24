"""API schema for Geri."""

from __future__ import annotations

import enum
from abc import ABC
from typing import Self

from pydantic import BaseModel, ConfigDict, RootModel, model_validator


class Status(enum.IntEnum):
    """Status of various operations."""

    SUCCESS = 0
    ERROR = 1
    FINISHED = 2


class Modality(enum.StrEnum):
    """Modality of the content to be generated."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


# ---------------- Base Geri request classes -----------------


class BatchGeriRequest(BaseModel, ABC):
    """An API request to generate batched content.

    Attributes:
        embedding_data_id: Sidecar data ID for prompt embeddings. Optional for
            request types that support direct text input.
        dummy_seq_len: Number of embedding tokens to simulate. This field is only used to
            determine the size of the random tensor to generate during dummy mode. In non-dummy
            mode, the data tensor is obtained simply by receiving them via sidecar.

    Modality-specific generation request classes (e.g., ImageGeriRequest)
    that support batched generation should inherit from this class.
    """

    embedding_data_id: str | None = None
    dummy_seq_len: int | None = None


class StreamGeriRequest(BaseModel, ABC):
    """An API request to generate streamed content.

    Attributes:
        embedding_data_id: Sidecar data ID for the audio codes.
        dummy_seq_len: Number of embedding tokens to simulate. This field is only used to
            determine the size of the random tensor to generate during dummy mode. In non-dummy
            mode, the data tensor is obtained simply by receiving them via sidecar.

    Modality-specific generation request classes (e.g., AudioGeriRequest)
    that support streamed generation should inherit from this class.
    """

    embedding_data_id: str
    dummy_seq_len: int | None = None


# ---------- Modality specific generation request classes ----------


class ImageGeriRequest(BatchGeriRequest):
    """Request to generate image content.

    Attributes:
        embedding_data_id: Sidecar data ID for prompt embeddings. Optional if `prompt` is
            provided directly.
        prompt: Prompt text for direct text-encoding mode. Optional if
            `embedding_data_id` is provided.
        height: Height of the generated content in pixels.
        width: Width of the generated content in pixels.
        num_inference_steps: Number of denoising steps to perform.
        skip_tokens: Number of initial tokens to skip from the embeddings.
    """

    prompt: str | None = None
    height: int
    width: int
    num_inference_steps: int
    skip_tokens: int = 0

    @model_validator(mode="after")
    def _validate_input_mode(self) -> Self:
        """Validate that exactly one input mode is selected.

        Image generation supports two input modes:
        1. Embedding mode: `embedding_data_id` is provided, `prompt` is omitted.
        2. Direct-text mode: `prompt` is provided, `embedding_data_id` is omitted.
        """
        has_embedding_data_id = bool(self.embedding_data_id and self.embedding_data_id.strip())
        has_prompt = bool(self.prompt and self.prompt.strip())

        if has_embedding_data_id == has_prompt:
            raise ValueError("Exactly one of `embedding_data_id` or `prompt` must be provided.")

        if has_prompt and self.skip_tokens != 0:
            raise ValueError("`skip_tokens` is only supported when using `embedding_data_id`.")

        return self


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
