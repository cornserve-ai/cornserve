from cornserve.task_executors.eric.engine.scheduler import Scheduler
from cornserve.task_executors.eric.schema import EngineEnqueueRequest, Modality, ProcessedEmbeddingData


def test_mixed_modality():
    """Batches should only hvae a single modality."""
    scheduler = Scheduler()

    scheduler.enqueue(
        EngineEnqueueRequest(
            request_id="1",
            data=[
                ProcessedEmbeddingData(id="im1", modality=Modality.IMAGE, data={}),
                ProcessedEmbeddingData(id="im2", modality=Modality.IMAGE, data={}),
            ],
        )
    )
    scheduler.enqueue(
        EngineEnqueueRequest(
            request_id="2",
            data=[
                ProcessedEmbeddingData(id="vid1", modality=Modality.VIDEO, data={}),
                ProcessedEmbeddingData(id="im3", modality=Modality.IMAGE, data={}),
            ],
        )
    )
    scheduler.enqueue(
        EngineEnqueueRequest(
            request_id="3",
            data=[
                ProcessedEmbeddingData(id="vid2", modality=Modality.VIDEO, data={}),
                ProcessedEmbeddingData(id="vid3", modality=Modality.VIDEO, data={}),
            ],
        )
    )

    assert scheduler.has_waiting_requests()
    batch = scheduler.schedule()
    assert batch.modality == Modality.IMAGE
    assert len(batch.request_ids) == 2
    scheduler.process_batch_result(request_ids=["1", "1"], data_ids=["im1", "im2"])

    assert scheduler.has_waiting_requests()
    batch = scheduler.schedule()
    assert batch.modality == Modality.VIDEO
    assert len(batch.request_ids) == 1
    scheduler.process_batch_result(request_ids=["2"], data_ids=["vid1"])

    assert scheduler.has_waiting_requests()
    batch = scheduler.schedule()
    assert batch.modality == Modality.IMAGE
    assert len(batch.request_ids) == 1
    scheduler.process_batch_result(request_ids=["2"], data_ids=["im3"])

    assert scheduler.has_waiting_requests()
    batch = scheduler.schedule()
    assert batch.modality == Modality.VIDEO
    assert len(batch.request_ids) == 2
    scheduler.process_batch_result(request_ids=["3", "3"], data_ids=["vid2", "vid3"])

    assert not scheduler.has_waiting_requests()
