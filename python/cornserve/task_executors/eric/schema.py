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
    receiver_sidecar_ranks: list[int] | None = None


class EngineResponse(msgspec.Struct, array_like=True, omit_defaults=True):
    """Response sent from the engine to the router."""

    request_ids: list[ID]
    status: Status
    error_message: str | None = None


@dataclass
class Batch:
    """Embedding requests to run together in a single forward pass.

    Currently, different modalities are not batched together because
    some models require different processing for different modalities.

    Attributes:
        modality: Modality of the data to be embedded.
        request_ids: List of request IDs in the batch. If there are multiple
            modality data in a single request, the request ID is repeated
            for each data ID.
        data_ids: List of unique data IDs in the batch. This is a
            concatenation of all data IDs in the batch.
        receiver_ranks: Sidecar ranks that will receive the embeddings.
        data: Dictionary of data to be embedded. The keys are the
            tensor names as returned by the HF processor and the corresponding
            encoder model should be expecting these names.
        _dump_prefix: Path to dump the batch data for debugging.
    """

    modality: Modality
    request_ids: list[ID] = field(default_factory=list)
    data_ids: list[ID] = field(default_factory=list)
    chunk_ids: list[int] = field(default_factory=list)
    num_chunks: list[int] = field(default_factory=list)
    receiver_ranks: list[list[int] | None] = field(default_factory=list)
    data: dict[str, list[torch.Tensor]] = field(default_factory=dict)

    _dump_prefix: str | None = None

    def __len__(self) -> int:
        """Return batch size."""
        return len(self.data_ids)

    def add_data(
        self,
        request_id: ID,
        data: list[ProcessedEmbeddingData],
        chunk_ids: list[int],
        num_chunks: list[int],
        receiver_ranks: list[int] | None = None,
    ) -> None:
        """Add a request to the batch."""
        # Add all modality data inside a request to the batch.
        for item, chunk_id, num_chunk in zip(data, chunk_ids, num_chunks, strict=True):
            if self.modality != item.modality:
                raise ValueError(
                    f"Cannot batch different modalities together: "
                    f"{self.modality} != {item.modality} (Data ID: {item.id})"
                )
            self.request_ids.append(request_id)
            self.data_ids.append(item.id)
            self.chunk_ids.append(chunk_id)
            self.num_chunks.append(num_chunk)
            self.receiver_ranks.append(receiver_ranks)
            for key, value in item.data.items():
                if key not in self.data:
                    self.data[key] = []
                self.data[key].append(torch.from_numpy(value))

        # Sanity check
        for key in self.data:
            assert len(self.data[key]) == len(self.data_ids), (
                f"Data length mismatch for key {key}: {len(self.data[key])} != {len(self.data_ids)}"
            )


@dataclass
class BatchResult:
    """Embedding result for a batch of requests."""

    request_ids: list[ID]
    data_ids: list[ID]
    chunk_ids: list[int]
    num_chunks: list[int]
    receiver_ranks: list[list[int] | None]
    status: Status
    error_message: str | None = None


@dataclass
class WorkerResult:
    """Result of a worker running a batch of data."""

    request_ids: list[str]
    status: Status
    error_message: str | None = None
