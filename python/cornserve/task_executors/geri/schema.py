"""Internal data schema definitions for Geri."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import msgspec

from cornserve.task_executors.geri.api import Status


class EngineOpcode(Enum):
    """Engine operation codes."""

    GENERATE = b"\x00"
    SHUTDOWN = b"\x01"


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


class EngineRequest(msgspec.Struct, array_like=True, omit_defaults=True):
    """Message sent to engine process for generation."""

    request_id: str
    embedding_data_id: str
    height: int
    width: int
    num_inference_steps: int
    skip_tokens: int = 0
    span_context: dict[str, str] | None = None


class EngineResponse(msgspec.Struct, array_like=True, omit_defaults=True):
    """Response from engine process."""

    request_id: str
    status: Status
    generated: str | None = None
    error_message: str | None = None


@dataclass
class GenerationResult:
    """Result from model executor (internal, not serialized)."""

    status: Status
    generated: list[str] = field(default_factory=list)
    error_message: str | None = None
