from __future__ import annotations

from cornserve.task_executors.eric.engine.scheduler import Scheduler
from cornserve.task_executors.eric.schema import EngineEnqueueRequest, Modality, ProcessedEmbeddingData


def test_uniform_modality_and_adapter():
    """Batches should only have uniform modality and adapter name."""
    scheduler = Scheduler()

    scheduler.enqueue(
        EngineEnqueueRequest(
            request_id="1",
            data=[
                ProcessedEmbeddingData(id="im1", model_id="m1", modality=Modality.IMAGE, data={}),
                ProcessedEmbeddingData(id="im2", model_id="m1", modality=Modality.IMAGE, data={}),
            ],
        )
    )
    scheduler.enqueue(
        EngineEnqueueRequest(
            request_id="2",
            data=[
                ProcessedEmbeddingData(id="vid1", model_id="m1", modality=Modality.VIDEO, data={}),
                ProcessedEmbeddingData(id="im3", model_id="m1", modality=Modality.IMAGE, data={}),
            ],
        )
    )
    scheduler.enqueue(
        EngineEnqueueRequest(
            request_id="3",
            data=[
                ProcessedEmbeddingData(id="vid2", model_id="m1", modality=Modality.VIDEO, data={}),
                ProcessedEmbeddingData(id="vid3", model_id="m1", modality=Modality.VIDEO, data={}),
            ],
        )
    )
    scheduler.enqueue(
        EngineEnqueueRequest(
            request_id="4",
            data=[
                ProcessedEmbeddingData(id="vid4", model_id="m2", modality=Modality.VIDEO, data={}),
                ProcessedEmbeddingData(id="vid5", model_id="m1", modality=Modality.VIDEO, data={}),
            ],
        )
    )

    assert scheduler.has_waiting_requests()
    batch = scheduler.schedule()
    assert batch.modality == Modality.IMAGE
    assert batch.adapter_name == "m1"
    assert len(batch.request_ids) == 2
    scheduler.process_batch_result(request_ids=["1", "1"], data_ids=["im1", "im2"])

    assert scheduler.has_waiting_requests()
    batch = scheduler.schedule()
    assert batch.modality == Modality.VIDEO
    assert batch.adapter_name == "m1"
    assert len(batch.request_ids) == 1
    scheduler.process_batch_result(request_ids=["2"], data_ids=["vid1"])

    assert scheduler.has_waiting_requests()
    batch = scheduler.schedule()
    assert batch.modality == Modality.IMAGE
    assert batch.adapter_name == "m1"
    assert len(batch.request_ids) == 1
    scheduler.process_batch_result(request_ids=["2"], data_ids=["im3"])

    assert scheduler.has_waiting_requests()
    batch = scheduler.schedule()
    assert batch.modality == Modality.VIDEO
    assert batch.adapter_name == "m1"
    assert len(batch.request_ids) == 2
    scheduler.process_batch_result(request_ids=["3", "3"], data_ids=["vid2", "vid3"])

    assert scheduler.has_waiting_requests()
    batch = scheduler.schedule()
    assert batch.modality == Modality.VIDEO
    assert batch.adapter_name == "m2"
    assert len(batch.request_ids) == 1
    scheduler.process_batch_result(request_ids=["4"], data_ids=["vid4"])

    assert scheduler.has_waiting_requests()
    batch = scheduler.schedule()
    assert batch.modality == Modality.VIDEO
    assert batch.adapter_name == "m1"
    assert len(batch.request_ids) == 1
    scheduler.process_batch_result(request_ids=["4"], data_ids=["vid5"])

    assert not scheduler.has_waiting_requests()


def test_max_batch_size():
    """Test that the scheduler respects the max batch size."""
    scheduler = Scheduler(max_batch_size=1)

    scheduler.enqueue(
        EngineEnqueueRequest(
            request_id="1",
            data=[
                ProcessedEmbeddingData(id="im1", model_id="m1", modality=Modality.IMAGE, data={}),
                ProcessedEmbeddingData(id="im2", model_id="m1", modality=Modality.IMAGE, data={}),
            ],
        )
    )
    scheduler.enqueue(
        EngineEnqueueRequest(
            request_id="2",
            data=[
                ProcessedEmbeddingData(id="vid1", model_id="m1", modality=Modality.VIDEO, data={}),
                ProcessedEmbeddingData(id="vid2", model_id="m1", modality=Modality.VIDEO, data={}),
            ],
        )
    )

    assert scheduler.has_waiting_requests()
    batch = scheduler.schedule()
    assert len(batch.request_ids) == 1
    scheduler.process_batch_result(request_ids=["1"], data_ids=["im1"])

    assert scheduler.has_waiting_requests()
    batch = scheduler.schedule()
    assert len(batch.request_ids) == 1
    scheduler.process_batch_result(request_ids=["1"], data_ids=["im2"])

    assert scheduler.has_waiting_requests()
    batch = scheduler.schedule()
    assert len(batch.request_ids) == 1
    scheduler.process_batch_result(request_ids=["2"], data_ids=["vid1"])

    assert scheduler.has_waiting_requests()
    batch = scheduler.schedule()
    assert len(batch.request_ids) == 1
    scheduler.process_batch_result(request_ids=["2"], data_ids=["vid2"])
