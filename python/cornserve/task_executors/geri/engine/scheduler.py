"""Scheduler for batching generation requests."""

from __future__ import annotations

from dataclasses import dataclass

from opentelemetry import propagate, trace
from opentelemetry.trace import Span

from cornserve.logging import get_logger
from cornserve.task_executors.geri.schema import EngineRequest, EngineRequestType

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)
propagator = propagate.get_global_textmap()


@dataclass
class ScheduledRequest:
    """A request that has been scheduled for execution."""

    request_id: str
    embedding_data_id: str
    height: int
    width: int
    num_inference_steps: int
    skip_tokens: int = 0
    span: Span | None = None

    request_type: EngineRequestType = EngineRequestType.NON_STREAMING
    chunk_size: int | None = None
    left_context_size: int | None = None


def requests_compatible(request1: ScheduledRequest, request2: ScheduledRequest) -> bool:
    """Returns whether two scheduled requests are capable of being the same batch.

    All requests in the same batch must be either streaming or non-streaming requests,
    indicated by the ScheduledRequest.request_type field.

    For non-streaming requests, these fields are compared:
        ScheduledRequest.height
        ScheduledRequest.width
        ScheduledRequest.num_inference_steps

    For streaming requests:
        ScheduledRequest.chunk_size
        ScheduledRequest.left_context_size
    """
    if request1.request_type != request2.request_type:
        return False

    if request1.request_type == EngineRequestType.NON_STREAMING:
        return (
            request1.height == request2.height
            and request1.width == request2.width
            and request1.num_inference_steps == request2.num_inference_steps
        )
    elif request1.request_type == EngineRequestType.STREAMING:
        return request1.chunk_size == request2.chunk_size and request1.left_context_size == request2.left_context_size

    return False


@dataclass
class SchedulerBatch:
    """A batch of requests to be executed together."""

    requests: list[ScheduledRequest]
    height: int
    width: int
    num_inference_steps: int
    request_type: EngineRequestType = EngineRequestType.NON_STREAMING

    chunk_size: int | None = None
    left_context_size: int | None = None

    def __post_init__(self) -> None:
        """Validate that all requests in the batch are compatible."""
        if not self.requests:
            raise ValueError("Batch cannot be empty")

        # Verify all requests have the same generation parameters
        first_req = self.requests[0]
        for req in self.requests[1:]:
            if not requests_compatible(first_req, req):
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

    @property
    def skip_tokens(self) -> list[int]:
        """Get list of skip tokens for this batch."""
        return [req.skip_tokens for req in self.requests]


class RequestQueue:
    """A FCFS request queue that allows batching of consecutive requests with same parameters."""

    def __init__(self) -> None:
        """Initialize the queue."""
        # Maintain FCFS order with a simple list
        self._requests: list[ScheduledRequest] = []

    def enqueue(self, request: EngineRequest, span: Span | None = None) -> None:
        """Add a request to the queue in FCFS order."""
        scheduled_req = ScheduledRequest(
            request_type=request.request_type,
            request_id=request.request_id,
            embedding_data_id=request.embedding_data_id,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            skip_tokens=request.skip_tokens,
            span=span,
            chunk_size=request.chunk_size,
            left_context_size=request.left_context_size,
        )

        self._requests.append(scheduled_req)

        logger.debug(
            "Enqueued request %s with params %dx%d, %d steps (queue length: %d)",
            request.request_id,
            request.height,
            request.width,
            request.num_inference_steps,
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
            if requests_compatible(next_request, req):
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

    def __init__(self, max_batch_size: int | None = None) -> None:
        """Initialize the scheduler.

        Args:
            max_batch_size: Maximum number of requests to batch together.
        """
        self.max_batch_size = max_batch_size
        self.queue = RequestQueue()

    def enqueue(self, request: EngineRequest, span: Span | None = None) -> None:
        """Add a request to the waiting queue."""
        if span:
            span.add_event("geri.engine.scheduler.enqueue")
        self.queue.enqueue(request, span)

    def has_waiting_requests(self) -> bool:
        """Check if there are any unfinished requests."""
        return self.queue.has_requests()

    def schedule(self) -> SchedulerBatch | None:
        """Schedule the next batch of requests.

        Returns:
            A batch of requests to execute, or None if no requests are waiting.
        """
        if not self.queue.has_requests():
            return None

        # Get the parameters for the next batch
        next_request = self.queue.peek_next_batch()
        if not next_request:
            return None

        # Pop requests for this batch
        batch_requests = self.queue.pop_batch(next_request, self.max_batch_size)

        if not batch_requests:
            return None

        logger.info(
            "Scheduled batch of %d requests",
            len(batch_requests),
        )

        batch = SchedulerBatch(
            requests=batch_requests,
            height=next_request.height,
            width=next_request.width,
            num_inference_steps=next_request.num_inference_steps,
            request_type=next_request.request_type,
            chunk_size=next_request.chunk_size,
            left_context_size=next_request.left_context_size,
        )

        for span in batch.spans:
            if span:
                span.add_event("geri.engine.scheduler.schedule")

        return batch
