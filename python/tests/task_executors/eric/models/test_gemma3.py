"""Tests for the Gemma3 model's vision encoder."""

import os

import pytest
import torch
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration

from cornserve.task_executors.eric.distributed.parallel import destroy_distributed, init_distributed
from cornserve.task_executors.eric.executor.executor import ModelExecutor
from cornserve.task_executors.eric.executor.loader import load_model
from cornserve.task_executors.eric.models.registry import MODEL_REGISTRY
from cornserve.task_executors.eric.schema import Status

from ..utils import ModalityData, assert_same_weights, batch_builder, param_tp_size

model_id = "google/gemma-3-4b-it"


def test_weight_loading() -> None:
    """Check if weights are loaded correctly."""
    # Hugging Face model output
    hf_model = Gemma3ForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto")

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


@param_tp_size
def test_image_inference(test_images: list[ModalityData], tp_size: int) -> None:
    """Test if inference works correctly."""
    executor = ModelExecutor(model_id=model_id, tp_size=tp_size, sender_sidecar_ranks=None)

    result = executor.execute_model(batch=batch_builder(model_id, "gemma3", test_images))

    assert result.status == Status.SUCCESS

    executor.shutdown()


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
