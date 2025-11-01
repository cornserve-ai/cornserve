"""Tests for the Geri engine scheduler implementation."""

import pytest

from cornserve.task_executors.geri.engine.scheduler import RequestQueue, ScheduledRequest, Scheduler, SchedulerBatch
from cornserve.task_executors.geri.schema import EngineRequest


def test_scheduler_batch_validation() -> None:
    """Test SchedulerBatch validation and properties."""
    requests = [
        ScheduledRequest("req1", "embed1", 256, 256, 10),
        ScheduledRequest("req2", "embed2", 256, 256, 10),
    ]

    batch = SchedulerBatch(requests=requests, height=256, width=256, num_inference_steps=10)

    assert len(batch) == 2
    assert batch.request_ids == ["req1", "req2"]
    assert batch.embedding_data_ids == ["embed1", "embed2"]
    assert len(batch.spans) == 2
    assert all(span is None for span in batch.spans)


def test_scheduler_batch_validation_error() -> None:
    """Test SchedulerBatch validation with mismatched parameters."""
    requests = [
        ScheduledRequest("req1", "embed1", 256, 256, 10),
        ScheduledRequest("req2", "embed2", 256, 512, 10),
    ]

    with pytest.raises(ValueError, match="identical generation parameters"):
        SchedulerBatch(requests=requests, height=256, width=256, num_inference_steps=10)


def test_request_queue_enqueue_and_length() -> None:
    """Test RequestQueue basic enqueue and length operations."""
    queue = RequestQueue()
    assert len(queue) == 0
    assert not queue.has_requests()

    # Create test engine request
    engine_req = EngineRequest(
        request_id="test-123", embedding_data_id="embed-456", height=512, width=512, num_inference_steps=20
    )

    queue.enqueue(engine_req)
    assert len(queue) == 1
    assert queue.has_requests()


def test_request_queue_fcfs_ordering() -> None:
    """Test RequestQueue maintains FCFS ordering."""
    queue = RequestQueue()

    # Add requests with different parameters in a specific order
    req1 = EngineRequest("req1", "embed1", 512, 512, 20)  # First request
    req2 = EngineRequest("req2", "embed2", 256, 256, 10)  # Different params
    req3 = EngineRequest("req3", "embed3", 256, 256, 10)  # Same as req2

    queue.enqueue(req1)
    queue.enqueue(req2)
    queue.enqueue(req3)

    assert len(queue) == 3

    # Should return parameters of first request (FCFS)
    next_request = queue.peek_next_batch()
    next_params = next_request.height, next_request.width, next_request.num_inference_steps
    assert next_params == (512, 512, 20)


def test_request_queue_pop_batch_consecutive() -> None:
    """Test RequestQueue batch popping with consecutive matching requests."""
    queue = RequestQueue()

    # Add multiple requests with same parameters
    for i in range(3):
        req = EngineRequest(f"req{i}", f"embed{i}", 256, 256, 10)
        queue.enqueue(req)

    # Pop batch of size 2
    req = queue.peek_next_batch()
    batch_requests = queue.pop_batch(req, max_batch_size=2)

    assert len(batch_requests) == 2
    assert batch_requests[0].request_id == "req0"
    assert batch_requests[1].request_id == "req1"
    assert len(queue) == 1  # One request should remain


def test_request_queue_pop_batch_nonexistent() -> None:
    """Test RequestQueue pop_batch with nonexistent parameters."""
    queue = RequestQueue()

    req = EngineRequest("req1", "embed1", 256, 256, 10)
    queue.enqueue(req)

    # Try to pop batch with different parameters
    req = EngineRequest("req2", "embed1", 512, 512, 20)
    batch_requests = queue.pop_batch(req)
    assert len(batch_requests) == 0
    assert len(queue) == 1  # Original request should remain


def test_request_queue_pop_batch_fcfs_stops_at_mismatch() -> None:
    """Test RequestQueue pop_batch stops at first non-matching request to maintain FCFS."""
    queue = RequestQueue()

    # Add requests in specific order: matching, non-matching, matching
    req1 = EngineRequest("req1", "embed1", 256, 256, 10)  # Matches
    req2 = EngineRequest("req2", "embed2", 512, 512, 20)  # Different params
    req3 = EngineRequest("req3", "embed3", 256, 256, 10)  # Matches but after non-match

    queue.enqueue(req1)
    queue.enqueue(req2)
    queue.enqueue(req3)

    # Pop batch for 256x256x10 parameters
    batch_requests = queue.pop_batch(req1)

    # Should only get req1, not req3, because req2 blocks the batch in FCFS order
    assert len(batch_requests) == 1
    assert batch_requests[0].request_id == "req1"
    assert len(queue) == 2  # req2 and req3 should remain


def test_scheduler_enqueue() -> None:
    """Test Scheduler enqueue functionality."""
    scheduler = Scheduler(max_batch_size=4)

    assert not scheduler.has_waiting_requests()

    req = EngineRequest("req1", "embed1", 256, 256, 10)
    scheduler.enqueue(req)

    assert scheduler.has_waiting_requests()


def test_scheduler_schedule_empty() -> None:
    """Test Scheduler returns None when no requests are waiting."""
    scheduler = Scheduler()

    batch = scheduler.schedule()
    assert batch is None


def test_scheduler_schedule_single_request() -> None:
    """Test Scheduler scheduling a single request."""
    scheduler = Scheduler()

    req = EngineRequest("req1", "embed1", 512, 256, 15)
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
    scheduler = Scheduler(max_batch_size=3)

    # Add requests with same parameters
    for i in range(5):
        req = EngineRequest(f"req{i}", f"embed{i}", 128, 128, 5)
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
    scheduler = Scheduler()

    # Add requests with different parameters in specific order
    req1 = EngineRequest("req1", "embed1", 256, 256, 10)  # First
    req2 = EngineRequest("req2", "embed2", 512, 512, 20)  # Second, different params
    req3 = EngineRequest("req3", "embed3", 256, 256, 10)  # Third, same as first

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
    scheduler = Scheduler()

    # Add consecutive requests with same parameters
    req1 = EngineRequest("req1", "embed1", 256, 256, 10)
    req2 = EngineRequest("req2", "embed2", 256, 256, 10)
    req3 = EngineRequest("req3", "embed3", 256, 256, 10)

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
    scheduler = Scheduler(max_batch_size=None)

    # Add many requests with same parameters
    for i in range(10):
        req = EngineRequest(f"req{i}", f"embed{i}", 64, 64, 1)
        scheduler.enqueue(req)

    # Should batch all requests together
    batch = scheduler.schedule()
    assert batch is not None
    assert len(batch) == 10

    # No more requests
    batch2 = scheduler.schedule()
    assert batch2 is None
