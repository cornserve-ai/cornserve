"""Public and private data schema definitions."""

import enum
from dataclasses import dataclass, field

import torch
import msgspec
import numpy as np
from pydantic import BaseModel


ID = str


class Modality(enum.Enum):
    """Modality of the data to be embedded."""

    IMAGE = "image"
    VIDEO = "video"


class EmbeddingData(BaseModel):
    """The data to be embedded.

    Attributes:
        id: Modality data ID unique within the request.
        modality: The modality of the data.
        url: The URL where the data can be downloaded from.
    """

    id: ID
    modality: Modality
    url: str


class EmbeddingRequest(BaseModel):
    """Request to embed data.

    Attributes:
        id: Cluster-wide unique request ID.
        data: List of data to be embedded.
        receiver_sidecar_ranks: Sidecar ranks that will receive the embeddings.
            If omitted, tensors will not be sent to any sidecar.
    """

    id: ID
    data: list[EmbeddingData]
    receiver_sidecar_ranks: list[int] | None = None


class Status(enum.IntEnum):
    """Status of various operations."""

    SUCCESS = 0
    ERROR = 1


class EmbeddingResponse(BaseModel):
    """Response containing the embedding."""

    status: Status
    error_message: str | None = None


class ProcessedEmbeddingData(msgspec.Struct, array_like=True, omit_defaults=True):
    """Processed embedding data.

    Attributes:
        id: Modality data ID unique within the request.
        modality: The modality of the data.
        data: List of processed data.
    """

    id: ID
    modality: Modality
    data: dict[str, np.ndarray]


class EngineOpcode(enum.Enum):
    """Instruction opcode for the engine."""

    ENQUEUE = b"\x00"
    PROFILE = b"\x01"


class EngineEnqueueRequest(msgspec.Struct, array_like=True, omit_defaults=True):
    """Enqueue request sent from the router to the engine."""

    request_id: str
    data: list[ProcessedEmbeddingData]


class EngineResponse(msgspec.Struct, array_like=True, omit_defaults=True):
    """Response sent from the engine to the router."""

    request_ids: list[ID]
    data_ids: list[ID]
    status: Status
    error_message: str | None = None


@dataclass
class Batch:
    """Embedding requests to run together in a single forward pass.

    Attributes:
        modality: Modality of the data to be embedded.
        request_ids: List of unique request IDs in the batch. If there
            are multiple modality data in the batch, this will be
            shorter than `data_ids`.
        data_ids: List of unique data IDs in the batch. This is a
            concatenation of all data IDs in the batch.
        data: Dictionary of data to be embedded. The keys are the
            tensor names as returned by the HF processor and the corresponding
            encoder model should be expecting these names.
    """

    # TODO: Can different modalities be batched together?
    modality: Modality
    request_ids: list[ID] = field(default_factory=list)
    data_ids: list[ID] = field(default_factory=list)
    data: dict[str, list[torch.Tensor]] = field(default_factory=dict)

    def add_request(self, request: EngineEnqueueRequest) -> None:
        """Add a request to the batch."""
        self.request_ids.append(request.request_id)
        # Add all modality data inside a request to the batch.
        for item in request.data:
            self.data_ids.append(item.id)
            for key, value in item.data.items():
                if key not in self.data:
                    self.data[key] = []
                self.data[key].append(torch.from_numpy(value))

        # Sanity check
        for key in self.data:
            assert len(self.data[key]) == len(self.data_ids), (
                f"Data length mismatch for key {key}: "
                f"{len(self.data[key])} != {len(self.data_ids)}"
            )


@dataclass
class BatchResult:
    """Embedding result for a batch of requests."""

    request_ids: list[ID]
    data_ids: list[ID]
    status: Status
    error_message: str | None = None


@dataclass
class WorkerResult:
    """Result of a worker running a batch of data."""

    request_ids: list[str]
    status: Status
    error_message: str | None = None
