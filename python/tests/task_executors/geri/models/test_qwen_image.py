"""Tests for the QwenImage model implementation."""

import torch

from cornserve.task_executors.geri.models.qwen_image import QwenImageModel

from ..utils import assert_valid_png_results_list, create_dummy_embeddings

model_id = "Qwen/Qwen-Image"
pipeline_class_name = "QwenImagePipeline"


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
