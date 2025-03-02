import gc
import uuid

import pytest
import torch
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

from cornserve.task_executors.eric.distributed.parallel import destroy_distributed, init_distributed
from cornserve.task_executors.eric.executor.loader import load_model
from cornserve.task_executors.eric.schema import Batch, Modality, Status
from cornserve.task_executors.eric.executor.executor import ModelExecutor

from ..utils import ModalityData, assert_same_weights, batch_builder, NUM_GPUS


def test_weight_loading() -> None:
    """Check if weights are loaded correctly."""
    model_id = "Qwen/Qwen2-VL-7B-Instruct"

    # Hugging Face model output
    hf_model_llm = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto")
    hf_model = hf_model_llm.visual
    del hf_model_llm
    gc.collect()

    # Load our model
    init_distributed(world_size=1, rank=0)
    our_model = load_model(model_id, modality=Modality.IMAGE, torch_device=torch.device("cpu"))
    destroy_distributed()

    # Check if parameters are the same
    assert_same_weights(hf_model, our_model)


@pytest.mark.parametrize(
    "tp_size",
    list(filter(lambda x: x <= NUM_GPUS, [1, 2, 4, 8])),
    ids=lambda x: f"TP={x}",
)
def test_inference(test_images: list[ModalityData], tp_size: int) -> None:
    """Test if inference works correctly."""
    model_id = "Qwen/Qwen2-VL-7B-Instruct"

    executor = ModelExecutor(model_id=model_id, modality=Modality.IMAGE, tp_size=tp_size)

    result = executor.execute_model(batch=batch_builder(model_id, test_images))

    assert result.status == Status.SUCCESS

    executor.shutdown()
