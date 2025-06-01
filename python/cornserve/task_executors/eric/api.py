"""API schema for Eric."""

from __future__ import annotations

import enum

from pydantic import BaseModel

ID = str


class Modality(enum.StrEnum):
    """Modality of the data to be embedded."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class EmbeddingData(BaseModel):
    """The data to be embedded.

    Attributes:
        id: Modality data ID unique within the request.
        modality: The modality of the data.
        url: The URL where the data can be downloaded from.
        receiver_sidecar_ranks: List of sidecar ranks to send the embeddings to.
            If omitted, tensors will not be sent to any sidecar.
    """

    id: ID
    modality: Modality
    url: str
    receiver_sidecar_ranks: list[list[int]] | None = None


class EmbeddingRequest(BaseModel):
    """Request to embed data.

    Attributes:
        data: List of data to be embedded.
    """

    data: list[EmbeddingData]


class Status(enum.IntEnum):
    """Status of various operations."""

    SUCCESS = 0
    ERROR = 1


class EmbeddingResponse(BaseModel):
    """Response containing the embedding."""

    status: Status
    error_message: str | None = None
