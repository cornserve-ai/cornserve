import enum

import msgspec
from pydantic import BaseModel

class Modality(enum.IntEnum):
    """Modality of the data to be embedded."""

    IMAGE = 0


class EmbeddingStatus(enum.IntEnum):
    """Whether the embedding was successfully computed or not."""

    SUCCESS = 0
    ERROR = 1


class EmbeddingRequest(BaseModel):
    """Request to embed data."""

    request_id: str
    urls: list[str]


class EmbeddingResponse(BaseModel):
    """Response containing the embedding."""

    status: EmbeddingStatus
    error_message: str | None = None


class EngineRequest(msgspec.Struct, array_like=True, omit_defaults=True):
    """Request sent from the router to the engine."""

    request_id: str
    shape: tuple[int, ...]
    dtype: str
    # TODO: Pickling (pickle.dumps with pickle.HIGHEST_PROTOCOL) the numpy array
    # and loading with pickle.loads is the fastest.
    # https://gist.github.com/tlrmchlsmth/8067f1b24a82b6e2f90450e7764fa103
    processed_tensors: bytes


class EngineResponse(msgspec.Struct, array_like=True, omit_defaults=True):
    """Response sent from the engine to the router."""

    request_id: str
    status: EmbeddingStatus
    error_message: str | None = None
