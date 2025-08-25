"""Scheduler for batching generation requests."""

from __future__ import annotations

from dataclasses import dataclass

from opentelemetry import propagate, trace
from opentelemetry.trace import Span

from cornserve.logging import get_logger
from cornserve.task_executors.geri.schema import EngineRequest

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


@dataclass
class SchedulerBatch:
    """A batch of requests to be executed together."""

    requests: list[ScheduledRequest]
    height: int
    width: int
    num_inference_steps: int

    def __post_init__(self) -> None:
        """Validate that all requests in the batch are compatible."""
        if not self.requests:
            raise ValueError("Batch cannot be empty")

        # Verify all requests have the same generation parameters
        first_req = self.requests[0]
        for req in self.requests[1:]:
            if (
                req.height != first_req.height
                or req.width != first_req.width
                or req.num_inference_steps != first_req.num_inference_steps
            ):
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
            request_id=request.request_id,
            embedding_data_id=request.embedding_data_id,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            skip_tokens=request.skip_tokens,
            span=span,
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

    def peek_next_batch(self) -> tuple[int, int, int] | None:
        """Peek at the parameters of the next batch without removing requests."""
        if not self._requests:
            return None

        # Always return the parameters of the first request in FCFS order
        first_request = self._requests[0]
        return (first_request.height, first_request.width, first_request.num_inference_steps)

    def pop_batch(
        self, height: int, width: int, num_inference_steps: int, max_batch_size: int | None = None
    ) -> list[ScheduledRequest]:
        """Pop a batch of consecutive requests with the specified parameters in FCFS order."""
        if not self._requests:
            return []

        # Find consecutive requests from the start that match the parameters
        batch_requests = []
        i = 0
        while i < len(self._requests) and (max_batch_size is None or len(batch_requests) < max_batch_size):
            req = self._requests[i]
            if req.height == height and req.width == width and req.num_inference_steps == num_inference_steps:
                batch_requests.append(req)
                i += 1
            else:
                # Stop at first non-matching request to maintain FCFS order
                break

        # Remove the batched requests from the front of the list
        self._requests = self._requests[len(batch_requests) :]

        logger.debug(
            "Popped batch of %d requests with params %dx%d, %d steps",
            len(batch_requests),
            height,
            width,
            num_inference_steps,
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
        batch_params = self.queue.peek_next_batch()
        if not batch_params:
            return None

        height, width, num_inference_steps = batch_params

        # Pop requests for this batch
        batch_requests = self.queue.pop_batch(height, width, num_inference_steps, self.max_batch_size)

        if not batch_requests:
            return None

        logger.info(
            "Scheduled batch of %d requests with %dx%d, %d steps",
            len(batch_requests),
            height,
            width,
            num_inference_steps,
        )

        batch = SchedulerBatch(
            requests=batch_requests,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
        )

        for span in batch.spans:
            if span:
                span.add_event("geri.engine.scheduler.schedule")

        return batch
