from collections import deque

from cornserve.task_executors.eric.schema import EngineRequest
from cornserve.logging import get_logger

logger = get_logger(__name__)


class Scheduler:
    """Scheduler for batching embedding requests."""

    def __init__(self) -> None:
        """Initialize the scheduler."""
        self.waiting_queue: deque[EngineRequest] = deque()

    def enqueue(self, request: EngineRequest) -> None:
        """Add a request to the waiting queue."""
        self.waiting_queue.append(request)

    def has_waiting_requests(self) -> bool:
        """Check if there are any unfinished requests."""
        return bool(self.waiting_queue)

    def schedule(self) -> Batch:
        """Schedule requests to run in the next batch."""
