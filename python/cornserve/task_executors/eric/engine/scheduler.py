"""The scheduler is responsible for batching embedding requests."""

from collections import defaultdict
from typing import Generator

from opentelemetry import propagate, trace

from cornserve.logging import get_logger
from cornserve.task_executors.eric.schema import (
    ID,
    EngineEnqueueRequest,
    Modality,
    ProcessedEmbeddingData,
    SchedulerBatch,
)

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)
propagator = propagate.get_global_textmap()


class RequestQueue:
    """A request queue that tracks the status of requests."""

    def __init__(self) -> None:
        """Initialize the queue."""
        # Python dicts preserve insertion order, so the dict is used as a queue.
        self.request_id_to_request: dict[str, EngineEnqueueRequest] = {}
        self.request_id_to_data_done: dict[str, dict[str, bool]] = defaultdict(dict)

    def __len__(self) -> int:
        """Return the number of requests in the queue."""
        return len(self.request_id_to_request)

    def enqueue(self, request: EngineEnqueueRequest) -> None:
        """Enqueue a new request."""
        self.request_id_to_request[request.request_id] = request
        for data in request.data:
            self.request_id_to_data_done[request.request_id][data.id] = False
        logger.debug(
            "Enqueued request %s with %d data items",
            request.request_id,
            len(request.data),
        )

    def mark_done(self, request_ids: list[str], data_ids: list[str]) -> list[ID]:
        """Mark the specified data as done for the given request IDs.

        Returns:
            A list of request IDs that are completely done.
        """
        assert len(request_ids) == len(data_ids)
        for request_id, data_id in zip(request_ids, data_ids, strict=True):
            self.request_id_to_data_done[request_id][data_id] = True

        # Requests that are completely done are removed from the queue.
        done_ids = []
        for request_id in set(request_ids):
            if all(self.request_id_to_data_done[request_id].values()):
                logger.debug("Request %s is done; removing from queue.", request_id)
                done_ids.append(request_id)
                del self.request_id_to_request[request_id]
                del self.request_id_to_data_done[request_id]

        return done_ids

    def is_done(self, request_id: str, data_id: str | None = None) -> bool:
        """Check whether something is done.

        Given only `request_id`, check if all data in the request is done.
        Given both `request_id` and `data_id`, check if the specific data is done.
        """
        # The request cannot be found because it was completed and removed.
        if request_id not in self.request_id_to_request:
            return True

        # Otherwise, the request is not done and must exist in the queue.
        if data_id is None:
            return False
        return self.request_id_to_data_done[request_id][data_id]

    def peek_request(self) -> EngineEnqueueRequest:
        """Peek the head of the queue without removing it."""
        assert self.request_id_to_request, "Queue is empty"
        return next(iter(self.request_id_to_request.values()))

    def peek_data(self) -> ProcessedEmbeddingData:
        """Peak the first waiting data of the head of the queue."""
        assert self.request_id_to_request, "Queue is empty"
        request = self.peek_request()
        assert request.data, "Request has no data"
        for data in request.data:
            if not self.is_done(request.request_id, data.id):
                return data
        raise ValueError("No waiting data in a waiting request")

    def iter_waiting(
        self,
        modality: Modality,
        max_items: int | None = None,
    ) -> Generator[tuple[EngineEnqueueRequest, ProcessedEmbeddingData], None, None]:
        """Iterate over the queue and yield waiting data with the given modality.

        When the modality of the data changes, stop iterating.
        """
        count = 0
        for request in self.request_id_to_request.values():
            for data in request.data:
                # We're only interested in data that is not done yet.
                if self.is_done(request.request_id, data.id):
                    continue

                # Modality change, stop!
                if data.modality != modality:
                    return

                yield request, data

                # If we have a limit on the number of items, stop when we reach it.
                count += 1
                if count == max_items:
                    return


class Scheduler:
    """Scheduler for batching embedding requests."""

    def __init__(self) -> None:
        """Initialize the scheduler."""
        self.queue = RequestQueue()

    def enqueue(self, request: EngineEnqueueRequest) -> None:
        """Add a request to the waiting queue."""
        if request.span:
            request.span.add_event("engine.scheduler.enqueue")
        self.queue.enqueue(request)

    def has_waiting_requests(self) -> bool:
        """Check if there are any unfinished requests."""
        return len(self.queue) > 0

    def schedule(self) -> SchedulerBatch:
        """Schedule requests to run in the next batch.

        This function is only called when there are requests in the queue.
        """
        # XXX: This is currently a simple scheduler that iterates over the queue
        # in order and batches *all* data items with the same modality.
        modality = self.queue.peek_data().modality
        batch = SchedulerBatch(modality=modality)
        # spans = []
        for req, data in self.queue.iter_waiting(modality=modality, max_items=None):
            if req.span:
                req.span.add_event("engine.scheduler.dequeue", attributes={"data_id": data.id})

            carrier = {}
            propagator.inject(carrier, req.context)

            batch.add_data(
                request_id=req.request_id,
                data=[data],
                chunk_ids=[0],
                num_chunks=[1],
                otel_spans=[req.span],
                otel_carriers=[carrier],
            )
            # spans.append(req.span)

        assert batch.request_ids, "Batch should not be empty"
        # # here we record all the batch_size for each request
        # for span in spans:
        #     if span:
        #         # here we cannot use attributes because it might be overwritten
        #         span.add_event("engine.scheduler.batch_size", attributes={"batch_size": len(batch)})

        return batch

    def process_batch_result(self, request_ids: list[ID], data_ids: list[ID]) -> list[ID]:
        """Process the result of a completed batch.

        Returns:
            A list of request IDs that are done (i.e., all data is done).
        """
        return self.queue.mark_done(request_ids, data_ids)
