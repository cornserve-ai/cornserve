from collections import deque

import torch

from cornserve.task_executors.eric.schema import Batch, EngineRequest
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
        # This is currently a dumb scheduler that dispatches everything
        # in the queue in a single batch.
        request_ids = []
        data = {}
        while self.waiting_queue:
            request = self.waiting_queue.popleft()
            for item in request.data:
                request_ids.append(request.request_id)
                for key, value in item.items():
                    if key not in data:
                        data[key] = []
                    data[key].append(torch.from_numpy(value))

        return Batch(request_ids=request_ids, data=data)
