"""Tests for the LLaVA-OneVision model's vision encoder."""

import os

import pytest
import torch
from transformers.models.llava_onevision.modeling_llava_onevision import LlavaOnevisionForConditionalGeneration

from cornserve.task_executors.eric.distributed.parallel import destroy_distributed, init_distributed
from cornserve.task_executors.eric.executor.loader import load_model
from cornserve.task_executors.eric.schema import Status
from cornserve.task_executors.eric.executor.executor import ModelExecutor
from cornserve.task_executors.eric.models.registry import MODEL_REGISTRY

from ..utils import ModalityData, assert_same_weights, batch_builder, NUM_GPUS, DUMP_DIR

model_id = "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf"
dump_prefix = os.getenv("CORNSERVE_TEST_DUMP_TENSOR_PREFIX", None)


def test_weight_loading() -> None:
    """Check if weights are loaded correctly."""
    # Hugging Face model output
    hf_model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto")

    # Load our model
    init_distributed(world_size=1, rank=0)
    our_model = load_model(model_id, torch_device=torch.device("cpu"))
    destroy_distributed()

    def check_qkv_proj_weight(
        our_name: str,
        our_param: torch.Tensor,
        hf_params: dict[str, torch.Tensor],
    ):
        """Check if the qkv_proj weights are the same."""
        separate_weights = []
        for key in ["q_proj", "k_proj", "v_proj"]:
            separate_weights.append(hf_params[our_name.replace("qkv_proj", key)])
        assert torch.allclose(our_param, torch.cat(separate_weights, dim=0))

    # Check if parameters are the same
    assert_same_weights(
        hf_model,
        our_model,
        required_prefixes=MODEL_REGISTRY[hf_model.config.model_type].weight.required_prefixes,
        ignored_prefixes=MODEL_REGISTRY[hf_model.config.model_type].weight.ignored_prefixes,
        transformed_weights={
            "*qkv_proj.weight": check_qkv_proj_weight,
            "*qkv_proj.bias": check_qkv_proj_weight,
        },
    )


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

    result = executor.execute_model(batch=batch_builder(model_id, "onevision", test_images))

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

    result = executor.execute_model(batch=batch_builder(model_id, "onevision", test_videos[:2]))

    assert result.status == Status.SUCCESS

    executor.shutdown()


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
