"""Tests for the Geri engine scheduler implementation."""

import pytest

from cornserve.task_executors.geri.engine.scheduler import (
    ImageScheduler,
    ImageSchedulerBatch,
    RequestQueue,
    ScheduledImageRequest,
)
from cornserve.task_executors.geri.schema import ImageEngineRequest


def test_scheduler_batch_validation() -> None:
    """Test ImageSchedulerBatch validation and properties."""
    requests = [
        ScheduledImageRequest("req1", "embed1", span=None, height=256, width=256, num_inference_steps=10),
        ScheduledImageRequest("req2", "embed2", span=None, height=256, width=256, num_inference_steps=10),
    ]

    batch = ImageSchedulerBatch(
        requests=requests,
        sched_request_type=ScheduledImageRequest,
        height=256,
        width=256,
        num_inference_steps=10,
    )

    assert len(batch) == 2
    assert batch.request_ids == ["req1", "req2"]
    assert batch.embedding_data_ids == ["embed1", "embed2"]
    assert len(batch.spans) == 2
    assert all(span is None for span in batch.spans)


def test_scheduler_batch_validation_error() -> None:
    """Test ImageSchedulerBatch validation with mismatched parameters."""
    requests = [
        ScheduledImageRequest("req1", "embed1", span=None, height=256, width=256, num_inference_steps=10),
        ScheduledImageRequest("req2", "embed2", span=None, height=256, width=512, num_inference_steps=10),
    ]

    with pytest.raises(ValueError, match="identical generation parameters"):
        ImageSchedulerBatch(
            requests=requests,
            sched_request_type=ScheduledImageRequest,
            height=256,
            width=256,
            num_inference_steps=10,
        )


def test_request_queue_enqueue_and_length() -> None:
    """Test RequestQueue basic enqueue and length operations."""
    queue = RequestQueue(ScheduledImageRequest)
    assert len(queue) == 0
    assert not queue.has_requests()

    # Create test engine request
    engine_req = ImageEngineRequest(
        request_id="test-123",
        embedding_data_id="embed-456",
        height=512,
        width=512,
        num_inference_steps=20,
        span_context=None,
    )

    queue.enqueue(engine_req)
    assert len(queue) == 1
    assert queue.has_requests()


def test_request_queue_fcfs_ordering() -> None:
    """Test RequestQueue maintains FCFS ordering."""
    queue = RequestQueue(ScheduledImageRequest)

    # Add requests with different parameters in a specific order
    req1 = ImageEngineRequest(
        "req1", "embed1", height=512, width=512, num_inference_steps=20, span_context=None
    )  # First request
    req2 = ImageEngineRequest(
        "req2", "embed2", height=256, width=256, num_inference_steps=10, span_context=None
    )  # Different params
    req3 = ImageEngineRequest(
        "req3", "embed3", height=256, width=256, num_inference_steps=10, span_context=None
    )  # Same as req2

    queue.enqueue(req1)
    queue.enqueue(req2)
    queue.enqueue(req3)

    assert len(queue) == 3

    # Should return parameters of first request (FCFS)
    next_request = queue.peek_next_batch()
    assert isinstance(next_request, ScheduledImageRequest)
    next_params = next_request.height, next_request.width, next_request.num_inference_steps
    assert next_params == (512, 512, 20)


def test_request_queue_pop_batch_consecutive() -> None:
    """Test RequestQueue batch popping with consecutive matching requests."""
    queue = RequestQueue(ScheduledImageRequest)

    # Add multiple requests with same parameters
    for i in range(3):
        req = ImageEngineRequest(
            f"req{i}", f"embed{i}", height=256, width=256, num_inference_steps=10, span_context=None
        )
        queue.enqueue(req)

    # Pop batch of size 2
    req = queue.peek_next_batch()
    assert req is not None
    batch_requests = queue.pop_batch(req, max_batch_size=2)

    assert len(batch_requests) == 2
    assert batch_requests[0].request_id == "req0"
    assert batch_requests[1].request_id == "req1"
    assert len(queue) == 1  # One request should remain


def test_request_queue_pop_batch_nonexistent() -> None:
    """Test RequestQueue pop_batch with nonexistent parameters."""
    queue = RequestQueue(ScheduledImageRequest)

    req = ImageEngineRequest("req1", "embed1", height=256, width=256, num_inference_steps=10, span_context=None)
    queue.enqueue(req)

    # Try to pop batch with different parameters
    scheduled_req_to_pop = ScheduledImageRequest(
        "req2", "embed1", span=None, height=512, width=512, num_inference_steps=20
    )
    batch_requests = queue.pop_batch(scheduled_req_to_pop)
    assert len(batch_requests) == 0
    assert len(queue) == 1  # Original request should remain


def test_request_queue_pop_batch_fcfs_stops_at_mismatch() -> None:
    """Test RequestQueue pop_batch stops at first non-matching request to maintain FCFS."""
    queue = RequestQueue(ScheduledImageRequest)

    # Add requests in specific order: matching, non-matching, matching
    req1 = ImageEngineRequest(
        "req1", "embed1", height=256, width=256, num_inference_steps=10, span_context=None
    )  # Matches
    req2 = ImageEngineRequest(
        "req2", "embed2", height=512, width=512, num_inference_steps=20, span_context=None
    )  # Different params
    req3 = ImageEngineRequest(
        "req3", "embed3", height=256, width=256, num_inference_steps=10, span_context=None
    )  # Matches but after non-match

    queue.enqueue(req1)
    queue.enqueue(req2)
    queue.enqueue(req3)

    # Pop batch for 256x256x10 parameters
    scheduled_req1 = queue.peek_next_batch()
    assert scheduled_req1 is not None
    batch_requests = queue.pop_batch(scheduled_req1)

    # Should only get req1, not req3, because req2 blocks the batch in FCFS order
    assert len(batch_requests) == 1
    assert batch_requests[0].request_id == "req1"
    assert len(queue) == 2  # req2 and req3 should remain


def test_scheduler_enqueue() -> None:
    """Test Scheduler enqueue functionality."""
    scheduler = ImageScheduler(max_batch_size=4)

    assert not scheduler.has_waiting_requests()

    req = ImageEngineRequest("req1", "embed1", height=256, width=256, num_inference_steps=10, span_context=None)
    scheduler.enqueue(req)

    assert scheduler.has_waiting_requests()


def test_scheduler_schedule_empty() -> None:
    """Test Scheduler returns None when no requests are waiting."""
    scheduler = ImageScheduler()

    batch = scheduler.schedule()
    assert batch is None


def test_scheduler_schedule_single_request() -> None:
    """Test Scheduler scheduling a single request."""
    scheduler = ImageScheduler()

    req = ImageEngineRequest("req1", "embed1", height=512, width=256, num_inference_steps=15, span_context=None)
    scheduler.enqueue(req)

    batch = scheduler.schedule()
    assert batch is not None
    assert len(batch) == 1
    assert batch.request_ids == ["req1"]
    assert batch.height == 512
    assert batch.width == 256
    assert batch.num_inference_steps == 15


def test_scheduler_schedule_batch() -> None:
    """Test Scheduler scheduling multiple requests with same parameters."""
    scheduler = ImageScheduler(max_batch_size=3)

    # Add requests with same parameters
    for i in range(5):
        req = ImageEngineRequest(
            f"req{i}", f"embed{i}", height=128, width=128, num_inference_steps=5, span_context=None
        )
        scheduler.enqueue(req)

    # Schedule first batch
    batch = scheduler.schedule()
    assert batch is not None
    assert len(batch) == 3  # Limited by max_batch_size
    assert batch.request_ids == ["req0", "req1", "req2"]

    # Schedule second batch
    batch2 = scheduler.schedule()
    assert batch2 is not None
    assert len(batch2) == 2  # Remaining requests
    assert batch2.request_ids == ["req3", "req4"]

    # No more requests
    batch3 = scheduler.schedule()
    assert batch3 is None


def test_scheduler_schedule_different_params_fcfs() -> None:
    """Test Scheduler with requests having different parameters maintains FCFS order."""
    scheduler = ImageScheduler()

    # Add requests with different parameters in specific order
    req1 = ImageEngineRequest(
        "req1", "embed1", height=256, width=256, num_inference_steps=10, span_context=None
    )  # First
    req2 = ImageEngineRequest(
        "req2", "embed2", height=512, width=512, num_inference_steps=20, span_context=None
    )  # Second, different params
    req3 = ImageEngineRequest(
        "req3", "embed3", height=256, width=256, num_inference_steps=10, span_context=None
    )  # Third, same as first

    scheduler.enqueue(req1)
    scheduler.enqueue(req2)
    scheduler.enqueue(req3)

    # Should schedule first request only (req1) because req2 has different params
    batch = scheduler.schedule()
    assert batch is not None
    assert len(batch) == 1
    assert batch.request_ids == ["req1"]
    assert batch.height == 256
    assert batch.width == 256

    # Should schedule second request (req2)
    batch2 = scheduler.schedule()
    assert batch2 is not None
    assert len(batch2) == 1
    assert batch2.request_ids == ["req2"]
    assert batch2.height == 512
    assert batch2.width == 512

    # Should schedule third request (req3)
    batch3 = scheduler.schedule()
    assert batch3 is not None
    assert len(batch3) == 1
    assert batch3.request_ids == ["req3"]
    assert batch3.height == 256
    assert batch3.width == 256


def test_scheduler_schedule_consecutive_same_params() -> None:
    """Test Scheduler can batch consecutive requests with same parameters."""
    scheduler = ImageScheduler()

    # Add consecutive requests with same parameters
    req1 = ImageEngineRequest("req1", "embed1", height=256, width=256, num_inference_steps=10, span_context=None)
    req2 = ImageEngineRequest("req2", "embed2", height=256, width=256, num_inference_steps=10, span_context=None)
    req3 = ImageEngineRequest("req3", "embed3", height=256, width=256, num_inference_steps=10, span_context=None)

    scheduler.enqueue(req1)
    scheduler.enqueue(req2)
    scheduler.enqueue(req3)

    # Should batch all three requests together
    batch = scheduler.schedule()
    assert batch is not None
    assert len(batch) == 3
    assert batch.request_ids == ["req1", "req2", "req3"]
    assert batch.height == 256
    assert batch.width == 256
    assert batch.num_inference_steps == 10

    # No more requests
    batch2 = scheduler.schedule()
    assert batch2 is None


def test_scheduler_max_batch_size_none() -> None:
    """Test Scheduler with no max_batch_size limit."""
    scheduler = ImageScheduler(max_batch_size=None)

    # Add many requests with same parameters
    for i in range(10):
        req = ImageEngineRequest(f"req{i}", f"embed{i}", height=64, width=64, num_inference_steps=1, span_context=None)
        scheduler.enqueue(req)

    # Should batch all requests together
    batch = scheduler.schedule()
    assert batch is not None
    assert len(batch) == 10

    # No more requests
    batch2 = scheduler.schedule()
    assert batch2 is None
