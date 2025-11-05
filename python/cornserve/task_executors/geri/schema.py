"""Internal data schema definitions for Geri."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from enum import Enum
from typing import Self

import msgspec
import torch

from cornserve.task_executors.geri.api import (
    AudioGeriRequest,
    BatchGeriRequest,
    ImageGeriRequest,
    Status,
    StreamGeriRequest,
)


class EngineOpcode(Enum):
    """Engine operation codes."""

    GENERATE = b"\x00"
    SHUTDOWN = b"\x01"


class GeriMode(Enum):
    """Mode of the Engine core (batched vs streaming)."""

    BATCH = b"\x00"
    STREAMING = b"\x01"


@dataclass
class GenerationRequest:
    """Internal generation request data structure.

    Attributes:
        request_id: Unique request identifier.
        height: Height of generated content in pixels.
        width: Width of generated content in pixels.
        num_inference_steps: Number of denoising steps.
    """

    request_id: str
    height: int
    width: int
    num_inference_steps: int


# ---------------- Base Engine request classes -----------------


class EngineRequest(msgspec.Struct, array_like=True, omit_defaults=True):
    """Message sent to engine process for generation."""

    request_id: str
    embedding_data_id: str
    span_context: dict[str, str] | None


class BatchEngineRequest(EngineRequest):
    """Engine request for batched (i.e., non-streaming) generation.

    Important:
        Modality-specific engine request classes (e.g., ImageEngineRequest) that
        support batched generation should inherit from this class.
    """

    pass


class StreamEngineRequest(EngineRequest):
    """Engine request for streamed generation.

    Important:
        Modality-specific engine request classes (e.g., AudioEngineRequest) that
        support streamed generation should inherit from this class.
    """

    pass


# ---------- Engine request classes based on modality ----------


class ImageEngineRequest(BatchEngineRequest):
    """Engine generation request for images."""

    height: int
    width: int
    num_inference_steps: int
    skip_tokens: int = 0

    @classmethod
    def from_geri_request(
        cls,
        geri_request: BatchGeriRequest,
        request_id: str,
        span_context: dict[str, str] | None,
    ) -> Self:
        """Produce an ImageEngineRequest from an ImageGeriRequest."""
        if not isinstance(geri_request, ImageGeriRequest):
            raise TypeError(f"Expected ImageGeriRequest, got {type(geri_request).__name__}")
        return cls(
            request_id=request_id,
            embedding_data_id=geri_request.embedding_data_id,
            span_context=span_context,
            height=geri_request.height,
            width=geri_request.width,
            num_inference_steps=geri_request.num_inference_steps,
            skip_tokens=geri_request.skip_tokens,
        )


class AudioEngineRequest(StreamEngineRequest):
    """Engine generation request for images."""

    chunk_size: int | None
    left_context_size: int | None

    @classmethod
    def from_geri_request(
        cls,
        geri_request: StreamGeriRequest,
        request_id: str,
        span_context: dict[str, str] | None,
    ) -> Self:
        """Produce an AudioEngineRequest from an AudioGeriRequest."""
        if not isinstance(geri_request, AudioGeriRequest):
            raise TypeError(f"Expected AudioGeriRequest, got {type(geri_request).__name__}")
        return cls(
            request_id=request_id,
            embedding_data_id=geri_request.embedding_data_id,
            span_context=span_context,
            chunk_size=geri_request.chunk_size,
            left_context_size=geri_request.left_context_size,
        )


# -------- Producing Engine requests from API requests ---------


class BatchEngineRequestFactory:
    """Factory for parsing API requests into batch Engine requests."""

    @staticmethod
    def from_geri_request(
        geri_request: BatchGeriRequest,
        request_id: str,
        span_context: dict[str, str] | None,
    ) -> BatchEngineRequest:
        """Produce a BatchEngineRequest given a BatchGeriRequest, request ID, and span context."""
        match geri_request:
            case ImageGeriRequest():
                return ImageEngineRequest.from_geri_request(geri_request, request_id, span_context)
            case _:
                raise TypeError(
                    f"Unsupported request type {type(geri_request).__name__} for producing BatchEngineRequest."
                )


class StreamEngineRequestFactory:
    """Factory for parsing API requests into stream Engine requests."""

    @staticmethod
    def from_geri_request(
        geri_request: StreamGeriRequest,
        request_id: str,
        span_context: dict[str, str] | None,
    ) -> StreamEngineRequest:
        """Produce a StreamEngineRequest given a StreamGeriRequest, request ID, and span context."""
        match geri_request:
            case AudioGeriRequest():
                return AudioEngineRequest.from_geri_request(geri_request, request_id, span_context)
            case _:
                raise TypeError(
                    f"Unsupported request type {type(geri_request).__name__} for producing StreamEngineRequest."
                )


# ------------------ Engine response classes -------------------


class EngineResponse(msgspec.Struct, array_like=True, omit_defaults=True):
    """Response from engine process."""

    request_id: str
    status: Status
    error_message: str | None = None


class BatchEngineResponse(EngineResponse):
    """Batched response from engine process."""

    generated: str | None = None


class StreamEngineResponse(EngineResponse):
    """Streamed response from engine process."""

    generate_bytes: bytes | None = None


# ------------------ Executor response classes ------------------


@dataclass
class GenerationResult:
    """Generation result from a base ModelExecutor (internal, not serialized)."""

    status: Status
    error_message: str | None = None


@dataclass
class BatchGenerationResult(GenerationResult):
    """Generation result from a BatchExecutor."""

    generated: list[str] = field(default_factory=list)


@dataclass
class StreamGenerationResult(GenerationResult):
    """Generation result from a StreamExecutor."""

    generator: Generator[list[torch.Tensor | None], None, None] | None = None
