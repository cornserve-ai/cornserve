"""The scheduler is responsible for batching embedding requests."""

from collections import deque

from cornserve.task_executors.eric.schema import Batch, EngineEnqueueRequest, Modality
from cornserve.logging import get_logger

logger = get_logger(__name__)


class Scheduler:
    """Scheduler for batching embedding requests."""

    def __init__(self, modality: Modality) -> None:
        """Initialize the scheduler."""
        self.modality = modality
        self.waiting_queue: deque[EngineEnqueueRequest] = deque()

    def enqueue(self, request: EngineEnqueueRequest) -> None:
        """Add a request to the waiting queue."""
        self.waiting_queue.append(request)

    def has_waiting_requests(self) -> bool:
        """Check if there are any unfinished requests."""
        return bool(self.waiting_queue)

    def schedule(self) -> Batch:
        """Schedule requests to run in the next batch."""
        # XXX: This is currently a dumb scheduler that dispatches everything
        # in the queue in a single batch.
        batch = Batch(modality=self.modality)
        while self.waiting_queue:
            request = self.waiting_queue.popleft()
            batch.add_request(request)
        return batch
