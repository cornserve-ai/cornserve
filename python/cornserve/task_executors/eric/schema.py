import enum
from dataclasses import dataclass

import torch
import msgspec
import numpy as np
from pydantic import BaseModel


class Modality(enum.IntEnum):
    """Modality of the data to be embedded."""

    IMAGE = 0


class EmbeddingRequest(BaseModel):
    """Request to embed data."""

    request_id: str
    urls: list[str]


class Status(enum.IntEnum):
    """Whether the embedding was successfully computed or not."""

    SUCCESS = 0
    ERROR = 1


class EmbeddingResponse(BaseModel):
    """Response containing the embedding."""

    status: Status
    error_message: str | None = None


class EngineOpcode(enum.Enum):
    """Instruction opcode for the engine."""

    ENQUEUE = b"\x00"
    PROFILE = b"\x01"


class EngineRequest(msgspec.Struct, array_like=True, omit_defaults=True):
    """Request sent from the router to the engine."""

    request_id: str
    data: list[dict[str, np.ndarray]]


class EngineResponse(msgspec.Struct, array_like=True, omit_defaults=True):
    """Response sent from the engine to the router."""

    request_ids: list[str]
    status: Status
    error_message: str | None = None


@dataclass
class Batch:
    """Embedding requests to run together in a single forward pass.

    The `data` dictionary has keys that match the signature of the model's
    forward method. The values are tensors that are batched together.
    """

    request_ids: list[str]
    data: dict[str, list[torch.Tensor]]


@dataclass
class BatchResult:
    """Embedding result for a batch of requests."""

    request_ids: list[str]
    status: Status
    error_message: str | None = None


@dataclass
class WorkerResult:
    """Result of a worker running a batch of data."""

    request_ids: list[str]
    status: Status
    error_message: str | None = None
