"""Tests for the Qwen2-VL model's vision encoder."""

import os

import pytest
import torch
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

from cornserve.task_executors.eric.distributed.parallel import destroy_distributed, init_distributed
from cornserve.task_executors.eric.executor.executor import ModelExecutor
from cornserve.task_executors.eric.executor.loader import load_model
from cornserve.task_executors.eric.schema import Status

from ..utils import ModalityData, assert_same_weights, batch_builder, param_tp_size

model_id = "Qwen/Qwen2.5-VL-7B-Instruct"


def test_weight_loading() -> None:
    """Check if weights are loaded correctly."""
    # Hugging Face model output
    hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto").visual

    # Load our model
    init_distributed(world_size=1, rank=0)
    our_model = load_model(model_id, torch_device=torch.device("cpu"))
    destroy_distributed()

    # Check if parameters are the same
    assert_same_weights(hf_model, our_model)


@param_tp_size
def test_image_inference(test_images: list[ModalityData], tp_size: int) -> None:
    """Test if inference works correctly."""
    executor = ModelExecutor(model_id=model_id, tp_size=tp_size, sender_sidecar_ranks=None)

    result = executor.execute_model(batch=batch_builder(model_id, "qwen2_5", test_images))

    assert result.status == Status.SUCCESS

    executor.shutdown()


@param_tp_size
def test_video_inference(test_videos: list[ModalityData], tp_size: int) -> None:
    """Test if inference works correctly."""
    executor = ModelExecutor(model_id=model_id, tp_size=tp_size, sender_sidecar_ranks=None)

    result = executor.execute_model(batch=batch_builder(model_id, "qwen2_5", test_videos[:2]))

    assert result.status == Status.SUCCESS

    executor.shutdown()


@pytest.mark.skipif(os.getenv("CORNSERVE_TEST_DUMP_TENSOR_DIR") is None, reason="Dump directory not set")
def test_hf_reference(test_images: list[ModalityData], test_videos: list[ModalityData]) -> None:
    """Generate reference outputs from the Hugging Face model."""
    torch.set_grad_enabled(False)

    hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
    )
    model = hf_model.model.cuda().eval()

    image1 = test_images[0].processed(model_id)
    pixel_values = torch.asarray(image1["pixel_values"]).cuda()
    image_grid_thw = torch.asarray(image1["image_grid_thw"]).cuda()
    output1 = model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    image2 = test_images[1].processed(model_id)
    pixel_values = torch.asarray(image2["pixel_values"]).cuda()
    image_grid_thw = torch.asarray(image2["image_grid_thw"]).cuda()
    output2 = model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    torch.save(
        [output1.cpu(), output2.cpu()], os.path.join(os.getenv("CORNSERVE_TEST_DUMP_TENSOR_DIR"), "qwen2_5-hf_image.pt")
    )

    video1 = test_videos[0].processed(model_id)
    pixel_values_video = torch.asarray(video1["pixel_values_videos"]).cuda()
    video_grid_thw = torch.asarray(video1["video_grid_thw"]).cuda()
    output1 = model.get_video_features(pixel_values_videos=pixel_values_video, video_grid_thw=video_grid_thw)

    video2 = test_videos[1].processed(model_id)
    pixel_values_video2 = torch.asarray(video2["pixel_values_videos"]).cuda()
    video_grid_thw2 = torch.asarray(video2["video_grid_thw"]).cuda()
    output2 = model.get_video_features(pixel_values_videos=pixel_values_video2, video_grid_thw=video_grid_thw2)

    torch.save(
        [output1.cpu(), output2.cpu()], os.path.join(os.getenv("CORNSERVE_TEST_DUMP_TENSOR_DIR"), "qwen2_5-hf_video.pt")
    )
