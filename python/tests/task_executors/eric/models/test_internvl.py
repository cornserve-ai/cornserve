"""Tests for the InternVL model's vision encoder."""

import pytest
import torch
from transformers import AutoModelForCausalLM

from cornserve.task_executors.eric.distributed.parallel import destroy_distributed, init_distributed
from cornserve.task_executors.eric.executor.executor import ModelExecutor
from cornserve.task_executors.eric.executor.loader import load_model
from cornserve.task_executors.eric.models.registry import MODEL_REGISTRY
from cornserve.task_executors.eric.schema import Status

from ..utils import (
    TP_SIZES,
    ModalityData,
    assert_same_weights,
    assert_similar,
    batch_builder,
    depends_on,
    param_tp_size,
)

model_shorthand = "internvl"


@pytest.mark.parametrize("model_id", ["OpenGVLab/InternVL3-1B", "OpenGVLab/InternVL3-38B"])
def test_weight_loading(model_id: str) -> None:
    """Check if weights are loaded correctly."""
    # Hugging Face model output
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        trust_remote_code=True,
    )

    # Load our model
    init_distributed(world_size=1, rank=0)
    our_model = load_model(model_id, torch_device=torch.device("cpu"))
    destroy_distributed()

    assert_same_weights(
        hf_model,
        our_model,
        required_prefixes=MODEL_REGISTRY[hf_model.config.model_type].weight.required_prefixes,
        ignored_prefixes=MODEL_REGISTRY[hf_model.config.model_type].weight.ignored_prefixes,
    )


@param_tp_size
@pytest.mark.parametrize("model_id", ["OpenGVLab/InternVL3-1B", "OpenGVLab/InternVL3-38B"])
def test_image_inference(test_images: list[ModalityData], tp_size: int, dump_tensors: str, model_id: str) -> None:
    """Test if inference works correctly."""
    executor = ModelExecutor(model_id=model_id, adapter_model_ids=[], tp_size=tp_size, sender_sidecar_ranks=None)

    model_shorthand = f"internvl-{model_id.split('-')[-1]}"
    result = executor.execute_model(batch=batch_builder(model_id, model_shorthand, test_images))

    assert result.status == Status.SUCCESS

    executor.shutdown()


@param_tp_size
@pytest.mark.parametrize("model_id", ["OpenGVLab/InternVL3-1B", "OpenGVLab/InternVL3-38B"])
def test_video_inference(test_videos: list[ModalityData], tp_size: int, dump_tensors: str, model_id: str) -> None:
    """Test if inference works correctly."""
    executor = ModelExecutor(model_id=model_id, adapter_model_ids=[], tp_size=tp_size, sender_sidecar_ranks=None)

    model_shorthand = f"internvl-{model_id.split('-')[-1]}"
    result = executor.execute_model(batch=batch_builder(model_id, model_shorthand, test_videos[:2]))

    assert result.status == Status.SUCCESS

    executor.shutdown()


@pytest.mark.dependency(
    depends=[
        "test_image_inference[OpenGVLab/InternVL3-1B-TP=1]",
        "test_image_inference[OpenGVLab/InternVL3-1B-TP=2]",
        "test_image_inference[OpenGVLab/InternVL3-1B-TP=4]",
        "test_image_inference[OpenGVLab/InternVL3-38B-TP=1]",
        "test_image_inference[OpenGVLab/InternVL3-38B-TP=2]",
        "test_image_inference[OpenGVLab/InternVL3-38B-TP=4]",
        "test_video_inference[OpenGVLab/InternVL3-1B-TP=1]",
        "test_video_inference[OpenGVLab/InternVL3-1B-TP=2]",
        "test_video_inference[OpenGVLab/InternVL3-1B-TP=4]",
        "test_video_inference[OpenGVLab/InternVL3-38B-TP=1]",
        "test_video_inference[OpenGVLab/InternVL3-38B-TP=2]",
        "test_video_inference[OpenGVLab/InternVL3-38B-TP=4]",
    ]
)
@pytest.mark.parametrize("model_id", ["OpenGVLab/InternVL3-1B", "OpenGVLab/InternVL3-38B"])
def test_hf_reference(
    test_images: list[ModalityData], test_videos: list[ModalityData], dump_tensors: str, model_id: str
) -> None:
    """Generate reference outputs from the Hugging Face model."""
    torch.set_grad_enabled(False)

    model_shorthand = f"internvl-{model_id.split('-')[-1]}"

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    del hf_model.language_model
    model = hf_model.cuda().eval()

    image1 = test_images[0].processed(model_id)
    pixel_values = torch.asarray(image1["pixel_values_flat"]).to(hf_model.dtype).cuda()
    image_features = model.extract_feature(pixel_values).view(-1, model.config.hidden_size)
    output1 = image_features.cpu()

    image2 = test_images[1].processed(model_id)
    pixel_values = torch.asarray(image2["pixel_values_flat"]).to(hf_model.dtype).cuda()
    image_features = model.extract_feature(pixel_values).view(-1, model.config.hidden_size)
    output2 = image_features.cpu()

    for tp_degree in TP_SIZES:
        output = torch.load(f"{dump_tensors}/{model_shorthand}-image-tp{tp_degree}.pt")
        assert_similar([output1, output2], output)

    del output1, output2

    video1 = test_videos[0].processed(model_id)
    pixel_values_videos = torch.asarray(video1["pixel_values_flat_video"]).to(hf_model.dtype).cuda()
    video_features = model.extract_feature(pixel_values_videos).view(-1, model.config.hidden_size)
    output1 = video_features.cpu()

    video2 = test_videos[1].processed(model_id)
    pixel_values_videos = torch.asarray(video2["pixel_values_flat_video"]).to(hf_model.dtype).cuda()
    video_features = model.extract_feature(pixel_values_videos).view(-1, model.config.hidden_size)
    output2 = video_features.cpu()

    for tp_degree in TP_SIZES:
        output = torch.load(f"{dump_tensors}/{model_shorthand}-video-tp{tp_degree}.pt")
        assert_similar([output1, output2], output)

    del output1, output2


# pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
