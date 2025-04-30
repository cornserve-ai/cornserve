"""Public and private data schema definitions."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import msgspec
import numpy as np
import torch
from opentelemetry import propagate, trace
from opentelemetry.context import Context
from opentelemetry.trace import Span

from cornserve.task_executors.eric.api import ID, Modality, Status

tracer = trace.get_tracer(__name__)
propagator = propagate.get_global_textmap()


class ProcessedEmbeddingData(msgspec.Struct, array_like=True, omit_defaults=True):
    """Processed embedding data.

    Attributes:
        id: Modality data ID unique within the request.
        modality: The modality of the data.
        data: List of processed data.
        receiver_sidecar_ranks: Sidecar ranks that will receive the embeddings.
            If omitted, tensors will not be sent to any sidecar.
    """

    id: ID
    modality: Modality
    data: dict[str, np.ndarray]
    receiver_sidecar_ranks: list[list[int]] | None = None


class EngineOpcode(enum.Enum):
    """Instruction opcode for the engine."""

    ENQUEUE = b"\x00"
    PROFILE = b"\x01"


class EngineEnqueueMessage(msgspec.Struct, array_like=True, omit_defaults=True):
    """Enqueue request message sent from the engine client to the engine.

    Attributes:
        request_id: Cluster-wide unique request ID.
        data: List of processed embedding data.
        otel_carrier: OpenTelemetry carrier that holds the context information for
            the request's span. This is used to propagate the context to the engine.
    """

    request_id: str
    data: list[ProcessedEmbeddingData]
    otel_carrier: dict[str, str] | None = None


@dataclass
class EngineEnqueueRequest:
    """Engine's internal data structure for the engine enqueue request.

    Attributes:
        request_id: Cluster-wide unique request ID.
        data: List of processed embedding data.
        context: OpenTelemetry context that holds the request's tracing context.
        span: OpenTelemetry span for the request.
    """

    request_id: str
    data: list[ProcessedEmbeddingData]
    context: Context | None = None
    span: Span | None = None

    @classmethod
    def from_msgpack(cls, msg: EngineEnqueueMessage) -> EngineEnqueueRequest:
        """Create an engine enqueue request from a engine enqueue message."""
        req = cls(request_id=msg.request_id, data=msg.data)
        if msg.otel_carrier:
            req.context = propagator.extract(msg.otel_carrier)
            req.span = tracer.start_span("Engine._request_receive_loop", context=req.context)
        return req


class EngineResponse(msgspec.Struct, array_like=True, omit_defaults=True):
    """Response sent from the engine to the router.

    Attributes:
        request_ids: List of request IDs that have completed processing.
        status: Status of the requests.
        error_message: Error message if the status is ERROR.
    """

    request_ids: list[ID]
    status: Status
    error_message: str | None = None


@dataclass
class SchedulerBatch:
    """Embedding requests to run together in a single forward pass from scheduelr output.

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
        otel_spans: List of OpenTelemetry spans for each request in the batch.
        otel_carriers: List of OpenTelemetry carriers for each request in the batch.
            Workers will extract these context.
        _dump_prefix: Path to dump the batch data for debugging.
    """

    modality: Modality
    request_ids: list[ID] = field(default_factory=list)
    data_ids: list[ID] = field(default_factory=list)
    chunk_ids: list[int] = field(default_factory=list)
    num_chunks: list[int] = field(default_factory=list)
    receiver_ranks: list[list[list[int]] | None] = field(default_factory=list)
    data: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    otel_spans: list[Span | None] = field(default_factory=list)
    otel_carriers: list[dict | None] = field(default_factory=list)

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
        otel_spans: list[Span | None],
        otel_carriers: list[dict | None],
    ) -> None:
        """Add a request to the batch."""
        # Add all modality data inside a request to the batch.
        for item, chunk_id, num_chunk, otel_span, otel_carrier in zip(
            data, chunk_ids, num_chunks, otel_spans, otel_carriers, strict=True
        ):
            if self.modality != item.modality:
                raise ValueError(
                    f"Cannot batch different modalities together: "
                    f"{self.modality} != {item.modality} (Data ID: {item.id})"
                )
            self.request_ids.append(request_id)
            self.data_ids.append(item.id)
            self.chunk_ids.append(chunk_id)
            self.num_chunks.append(num_chunk)
            self.receiver_ranks.append(item.receiver_sidecar_ranks)
            self.otel_spans.append(otel_span)
            self.otel_carriers.append(otel_carrier)
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
class WorkerBatch:
    """Worker's internal data structure for a forward pass batch.

    Currently, it is SchedulerBatch without the otel_spans field.

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
        otel_carriers: List of OpenTelemetry carriers for each request in the batch. Workers will extract these context.
        _dump_prefix: Path to dump the batch data for debugging.
    """

    modality: Modality
    request_ids: list[ID] = field(default_factory=list)
    data_ids: list[ID] = field(default_factory=list)
    chunk_ids: list[int] = field(default_factory=list)
    num_chunks: list[int] = field(default_factory=list)
    receiver_ranks: list[list[list[int]] | None] = field(default_factory=list)
    data: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    otel_carriers: list[dict | None] = field(default_factory=list)

    _dump_prefix: str | None = None

    def __len__(self) -> int:
        """Return batch size."""
        return len(self.data_ids)

    @classmethod
    def from_scheduler_batch(cls, batch: SchedulerBatch) -> WorkerBatch:
        """Create a worker batch from a scheduler batch."""
        return cls(
            modality=batch.modality,
            request_ids=batch.request_ids,
            data_ids=batch.data_ids,
            chunk_ids=batch.chunk_ids,
            num_chunks=batch.num_chunks,
            receiver_ranks=batch.receiver_ranks,
            data=batch.data,
            otel_carriers=batch.otel_carriers,
            _dump_prefix=batch._dump_prefix,
        )


@dataclass
class BatchResult:
    """Embedding result for a batch of requests."""

    request_ids: list[ID]
    data_ids: list[ID]
    chunk_ids: list[int]
    num_chunks: list[int]
    receiver_ranks: list[list[list[int]] | None]
    status: Status
    error_message: str | None = None


@dataclass
class WorkerResult:
    """Result of a worker running a batch of data."""

    request_ids: list[str]
    status: Status
    error_message: str | None = None
