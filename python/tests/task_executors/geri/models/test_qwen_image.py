"""Tests for the QwenImage model implementation."""

import torch

from cornserve.task_executors.geri.executor.loader import load_model, get_registry_entry
from cornserve.task_executors.geri.models.base import BatchGeriModel, GeriModel
from cornserve.task_executors.geri.models.qwen_image import QwenImageModel

from ..utils import assert_valid_png_results_list, create_dummy_embeddings

model_id = "Qwen/Qwen-Image"
pipeline_class_name = "QwenImagePipeline"


def test_model_loading() -> None:
    """Test model is correctly configured for model loader."""
    registry_entry, _ = get_registry_entry(model_id)
    model = load_model(model_id, torch.device("cuda"), registry_entry)
    assert isinstance(model, GeriModel)
    assert isinstance(model, BatchGeriModel)


def test_qwen_image_generation() -> None:
    """Test QwenImage generation functionality."""
    model = QwenImageModel(
        model_id=model_id,
        torch_dtype=torch.bfloat16,
        torch_device=torch.device("cuda"),
    )

    # Create dummy embeddings that match the expected shape
    prompt_embeds = create_dummy_embeddings(batch_size=1)

    # Generate a very small image with minimal steps for a quick test
    result = model.generate(
        prompt_embeds=prompt_embeds,
        height=128,
        width=128,
        num_inference_steps=1,
    )

    # Validate the output
    assert_valid_png_results_list(result, expected_batch_size=1)
