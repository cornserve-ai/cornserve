"""Scheduler for batching generation requests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from opentelemetry import propagate, trace
from opentelemetry.trace import Span

from cornserve.logging import get_logger
from cornserve.task_executors.geri.schema import (
    AudioEngineRequest,
    EngineRequest,
    ImageEngineRequest,
)

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)
propagator = propagate.get_global_textmap()


@dataclass
class ScheduledRequest(ABC):
    """A request that has been scheduled for execution."""

    request_id: str
    embedding_data_id: str
    span: Span | None

    @classmethod
    @abstractmethod
    def from_engine_request(cls, engine_request: EngineRequest, span: trace.Span | None) -> ScheduledRequest:
        """Creates a scheduled request from an engine request."""

    @abstractmethod
    def requests_compatible(self, other: ScheduledRequest) -> bool:
        """Returns whether the given request can be in the same batch."""


@dataclass
class ScheduledImageRequest(ScheduledRequest):
    """An image generation request that has been scheduled for execution."""

    height: int
    width: int
    num_inference_steps: int
    skip_tokens: int = 0

    @classmethod
    def from_engine_request(cls, engine_request: EngineRequest, span: trace.Span | None) -> ScheduledImageRequest:
        """Creates a scheduled image request from an engine request."""
        assert isinstance(engine_request, ImageEngineRequest)
        return cls(
            request_id=engine_request.request_id,
            embedding_data_id=engine_request.embedding_data_id,
            span=span,
            height=engine_request.height,
            width=engine_request.width,
            num_inference_steps=engine_request.num_inference_steps,
            skip_tokens=engine_request.skip_tokens,
        )

    def requests_compatible(self, other: ScheduledRequest) -> bool:
        """Returns whether the given request can be in the same ImageSchedulerBatch.

        To be compatible, two image requests must be identical in these fields:
            ScheduledImageRequest.height
            ScheduledImageRequest.width
            ScheduledImageRequest.num_inference_steps
        """
        if not isinstance(other, ScheduledImageRequest):
            return False
        return (
            self.height == other.height
            and self.width == other.width
            and self.num_inference_steps == other.num_inference_steps
        )


@dataclass
class ScheduledAudioRequest(ScheduledRequest):
    """An audio generation request that has been scheduled for execution."""

    chunk_size: int | None = None
    left_context_size: int | None = None

    @classmethod
    def from_engine_request(cls, engine_request: EngineRequest, span: trace.Span | None) -> ScheduledAudioRequest:
        """Creates a scheduled audio request from an engine request."""
        assert isinstance(engine_request, AudioEngineRequest)
        return cls(
            request_id=engine_request.request_id,
            embedding_data_id=engine_request.embedding_data_id,
            span=span,
            chunk_size=engine_request.chunk_size,
            left_context_size=engine_request.left_context_size,
        )

    def requests_compatible(self, other: ScheduledRequest) -> bool:
        """Returns whether the given request can be in the same AudioSchedulerBatch.

        To be compatible, two audio requests must be identical in these fields:
            ScheduledAudioRequest.chunk_size
            ScheduledAudioRequest.left_context_size
        """
        if not isinstance(other, ScheduledAudioRequest):
            return False
        return self.chunk_size == other.chunk_size and self.left_context_size == other.left_context_size


@dataclass
class SchedulerBatch:
    """A batch of requests to be executed together."""

    requests: list[ScheduledRequest]

    # The type of ScheduledRequest class to be batched
    sched_request_type: ClassVar[type[ScheduledRequest]]

    def __post_init__(self) -> None:
        """Validate that all requests in the batch are compatible."""
        if not self.requests:
            raise ValueError("Batch cannot be empty")

        # Verify all requests have the right ScheduledRequest type and
        # have the same generation parameters
        first_req = self.requests[0]
        if not isinstance(first_req, self.sched_request_type):
            raise TypeError(f"Expected {self.sched_request_type.__name__}, got {type(first_req).__name__}")

        for req in self.requests[1:]:
            if not isinstance(req, self.sched_request_type):
                raise TypeError(f"Expected {self.sched_request_type.__name__}, got {type(req).__name__}")
            if first_req.requests_compatible(req):
                raise ValueError("All requests in a batch must have identical generation parameters")

    def __len__(self) -> int:
        """Return the number of requests in this batch."""
        return len(self.requests)

    @property
    def request_ids(self) -> list[str]:
        """Get list of request IDs in this batch."""
        return [req.request_id for req in self.requests]

    @property
    def embedding_data_ids(self) -> list[str]:
        """Get list of embedding data IDs in this batch."""
        return [req.embedding_data_id for req in self.requests]

    @property
    def spans(self) -> list[Span | None]:
        """Get list of tracing spans for this batch."""
        return [req.span for req in self.requests]


@dataclass
class ImageSchedulerBatch(SchedulerBatch):
    """A batch of image requests to be executed together.

    Extends the base SchedulerBatch class with fields specific to image batches.
    """

    height: int
    width: int
    num_inference_steps: int

    sched_request_type: ClassVar[type[ScheduledRequest]] = ScheduledImageRequest

    @property
    def skip_tokens(self) -> list[int]:
        """Get list of skip tokens for this batch."""
        return [getattr(req, "skip_tokens", 0) for req in self.requests]


@dataclass
class AudioSchedulerBatch(SchedulerBatch):
    """A batch of audio requests to be executed together.

    Extends the base SchedulerBatch class with fields specific to audio batches.
    """

    chunk_size: int | None = None
    left_context_size: int | None = None

    sched_request_type: ClassVar[type[ScheduledRequest]] = ScheduledAudioRequest


class RequestQueue:
    """A FCFS request queue that allows batching of consecutive requests with same parameters."""

    def __init__(self, sched_request_type: type[ScheduledRequest]) -> None:
        """Initialize the queue."""
        # Maintain FCFS order with a simple list
        self._requests: list[ScheduledRequest] = []
        self.sched_request_type: type[ScheduledRequest] = sched_request_type

    def enqueue(self, request: EngineRequest, span: Span | None = None) -> None:
        """Add a request to the queue in FCFS order."""
        scheduled_req = self.sched_request_type.from_engine_request(request, span)
        self._requests.append(scheduled_req)

        logger.debug(
            "Enqueued request %s (queue length: %d)",
            request.request_id,
            len(self._requests),
        )

    def __len__(self) -> int:
        """Return the total number of requests in the queue."""
        return len(self._requests)

    def has_requests(self) -> bool:
        """Check if there are any requests waiting."""
        return len(self._requests) > 0

    def peek_next_batch(self) -> ScheduledRequest | None:
        """Peek at the next request in the next batch without removing requests."""
        if not self._requests:
            return None

        # Always return the first request in FCFS order
        return self._requests[0]

    def pop_batch(
        self,
        next_request: ScheduledRequest,
        max_batch_size: int | None = None,
    ) -> list[ScheduledRequest]:
        """Pop a batch of consecutive requests in FCFS order with the parameters of the given request."""
        if not self._requests:
            return []

        # Find consecutive requests from the start that match the parameters
        batch_requests = []
        i = 0
        while i < len(self._requests) and (max_batch_size is None or len(batch_requests) < max_batch_size):
            req = self._requests[i]
            if next_request.requests_compatible(req):
                batch_requests.append(req)
                i += 1
            else:
                # Stop at first non-matching request to maintain FCFS order
                break

        # Remove the batched requests from the front of the list
        self._requests = self._requests[len(batch_requests) :]

        logger.debug(
            "Popped batch of %d requests",
            len(batch_requests),
        )

        return batch_requests


class Scheduler:
    """Scheduler for batching generation requests."""

    queue: RequestQueue

    def __init__(self, max_batch_size: int | None = None) -> None:
        """Initialize the scheduler.

        Args:
            max_batch_size: Maximum number of requests to batch together.
        """
        self.max_batch_size = max_batch_size

    @abstractmethod
    def schedule(self) -> SchedulerBatch | None:
        """Schedule the next batch of requests.

        Returns:
            A batch of requests to execute, or None if no requests are waiting.
        """

    def enqueue(self, request: EngineRequest, span: Span | None = None) -> None:
        """Add a request to the waiting queue."""
        if span:
            span.add_event("geri.engine.scheduler.enqueue")
        self.queue.enqueue(request, span)

    def has_waiting_requests(self) -> bool:
        """Check if there are any unfinished requests."""
        return self.queue.has_requests()

    def get_next_requests(self) -> list[ScheduledRequest] | None:
        """Return the next requests to batch."""
        if not self.queue.has_requests():
            return None

        next_request = self.queue.peek_next_batch()
        if not next_request:
            return None

        batch_requests = self.queue.pop_batch(next_request, self.max_batch_size)
        if not batch_requests:
            return None

        return batch_requests


class ImageScheduler(Scheduler):
    """Scheduler for batching image generation requests."""

    def __init__(self, max_batch_size: int | None = None) -> None:
        """Initialize the scheduler.

        Args:
            max_batch_size: Maximum number of requests to batch together.
        """
        self.queue = RequestQueue(ScheduledImageRequest)
        super().__init__(max_batch_size)

    def schedule(self) -> ImageSchedulerBatch | None:
        """Schedule the next batch of requests.

        Returns:
            A batch of requests to execute, or None if no requests are waiting.
        """
        batch_requests = self.get_next_requests()
        if not batch_requests:
            return None

        next_request = batch_requests[0]
        assert isinstance(next_request, ScheduledImageRequest)
        batch = ImageSchedulerBatch(
            requests=batch_requests,
            height=next_request.height,
            width=next_request.width,
            num_inference_steps=next_request.num_inference_steps,
        )

        logger.info(
            "Scheduled image batch of %d requests",
            len(batch_requests),
        )

        for span in batch.spans:
            if span:
                span.add_event("geri.engine.scheduler.schedule")

        return batch


class AudioScheduler(Scheduler):
    """Scheduler for batching audio generation requests."""

    def __init__(self, max_batch_size: int | None = None) -> None:
        """Initialize the scheduler.

        Args:
            max_batch_size: Maximum number of requests to batch together.
        """
        self.queue = RequestQueue(ScheduledAudioRequest)
        super().__init__(max_batch_size)

    def schedule(self) -> AudioSchedulerBatch | None:
        """Schedule the next batch of requests.

        Returns:
            A batch of requests to execute, or None if no requests are waiting.
        """
        batch_requests = self.get_next_requests()
        if not batch_requests:
            return None

        next_request = batch_requests[0]
        assert isinstance(next_request, ScheduledAudioRequest)
        batch = AudioSchedulerBatch(
            requests=batch_requests,
            chunk_size=next_request.chunk_size,
            left_context_size=next_request.left_context_size,
        )

        logger.info(
            "Scheduled audio batch of %d requests",
            len(batch_requests),
        )

        for span in batch.spans:
            if span:
                span.add_event("geri.engine.scheduler.schedule")

        return batch
