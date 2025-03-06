"""Tests for the Qwen2-VL model's vision encoder."""

import pytest
import torch
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

from cornserve.task_executors.eric.distributed.parallel import destroy_distributed, init_distributed
from cornserve.task_executors.eric.executor.loader import load_model
from cornserve.task_executors.eric.schema import Status
from cornserve.task_executors.eric.executor.executor import ModelExecutor

from ..utils import ModalityData, assert_same_weights, batch_builder, NUM_GPUS

model_id = "Qwen/Qwen2-VL-7B-Instruct"


def test_weight_loading() -> None:
    """Check if weights are loaded correctly."""
    # Hugging Face model output
    hf_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto").visual

    # Load our model
    init_distributed(world_size=1, rank=0)
    our_model = load_model(model_id, torch_device=torch.device("cpu"))
    destroy_distributed()

    # Check if parameters are the same
    assert_same_weights(hf_model, our_model)


@pytest.mark.parametrize(
    "tp_size",
    list(filter(lambda x: x <= NUM_GPUS, [1, 2, 4, 8])),
    ids=lambda x: f"TP={x}",
)
def test_image_inference(test_images: list[ModalityData], tp_size: int) -> None:
    """Test if inference works correctly."""
    executor = ModelExecutor(
        model_id=model_id,
        tp_size=tp_size,
        sender_sidecar_ranks=list(range(tp_size)),
    )

    result = executor.execute_model(batch=batch_builder(model_id, "qwen2", test_images))

    assert result.status == Status.SUCCESS

    executor.shutdown()


@pytest.mark.parametrize(
    "tp_size",
    list(filter(lambda x: x <= NUM_GPUS, [1, 2, 4, 8])),
    ids=lambda x: f"TP={x}",
)
def test_video_inference(test_videos: list[ModalityData], tp_size: int) -> None:
    """Test if inference works correctly."""
    executor = ModelExecutor(
        model_id=model_id,
        tp_size=tp_size,
        sender_sidecar_ranks=list(range(tp_size)),
    )

    result = executor.execute_model(batch=batch_builder(model_id, "qwen2", test_videos[:2]))

    assert result.status == Status.SUCCESS

    executor.shutdown()
