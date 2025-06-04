"""Tests for the Qwen2.5-Omni model's vision and audio encoders."""

import pytest
import torch
from transformers.models.auto.processing_auto import AutoProcessor
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration

from cornserve.task_executors.eric.distributed.parallel import destroy_distributed, init_distributed
from cornserve.task_executors.eric.executor.executor import ModelExecutor
from cornserve.task_executors.eric.executor.loader import load_model
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

model_id = "Qwen/Qwen2.5-Omni-7B"
model_shorthand = "qwen2_5_omni"


def test_weight_loading() -> None:
    """Check if weights are loaded correctly."""
    # Hugging Face model output
    hf_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto").thinker

    # Load our model
    init_distributed(world_size=1, rank=0)
    our_model = load_model(model_id, torch_device=torch.device("cpu"))
    destroy_distributed()

    # Check if parameters are the same
    assert_same_weights(hf_model, our_model)


@param_tp_size
def test_image_inference(test_images: list[ModalityData], tp_size: int, dump_tensors: str) -> None:
    """Test if inference works correctly."""
    executor = ModelExecutor(model_id=model_id, tp_size=tp_size, sender_sidecar_ranks=None)

    result = executor.execute_model(batch=batch_builder(model_id, model_shorthand, test_images))

    assert result.status == Status.SUCCESS

    executor.shutdown()


@param_tp_size
def test_video_inference(test_videos: list[ModalityData], tp_size: int, dump_tensors: str) -> None:
    """Test if inference works correctly."""
    executor = ModelExecutor(model_id=model_id, tp_size=tp_size, sender_sidecar_ranks=None)

    result = executor.execute_model(batch=batch_builder(model_id, model_shorthand, test_videos[:2]))

    assert result.status == Status.SUCCESS

    executor.shutdown()


@param_tp_size
def test_audio_inference(test_audios: list[ModalityData], tp_size: int, dump_tensors: str) -> None:
    """Test if inference works correctly."""
    executor = ModelExecutor(model_id=model_id, tp_size=tp_size, sender_sidecar_ranks=None)

    result = executor.execute_model(batch=batch_builder(model_id, model_shorthand, test_audios[:2]))

    assert result.status == Status.SUCCESS

    executor.shutdown()


@depends_on("test_image_inference", "test_video_inference", "test_audio_inference")
def test_hf_reference(
    test_audios: list[ModalityData],
    test_images: list[ModalityData],
    test_videos: list[ModalityData],
    dump_tensors: str,
) -> None:
    """Generate reference outputs from the Hugging Face model."""
    torch.set_grad_enabled(False)

    hf_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
    )
    model = hf_model.thinker.cuda().eval()
    del model.model

    processor = AutoProcessor.from_pretrained(model_id)

    audio1 = processor.feature_extractor(
        [test_audios[0].raw()],
        sampling_rate=processor.feature_extractor.sampling_rate,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_features = audio1["input_features"].cuda()
    feature_attention_mask = audio1["attention_mask"].cuda()
    output1 = model.get_audio_features(
        input_features=input_features, feature_attention_mask=feature_attention_mask
    ).cpu()

    audio2 = processor.feature_extractor(
        [test_audios[1].raw()],
        sampling_rate=processor.feature_extractor.sampling_rate,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_features = audio2["input_features"].cuda()
    feature_attention_mask = audio2["attention_mask"].cuda()
    output2 = model.get_audio_features(
        input_features=input_features, feature_attention_mask=feature_attention_mask
    ).cpu()

    for tp_degree in TP_SIZES:
        output = torch.load(f"{dump_tensors}/{model_shorthand}-audio-tp{tp_degree}.pt")
        assert_similar([output1, output2], output)

    del output1, output2

    image1 = test_images[0].processed(model_id)
    pixel_values = torch.asarray(image1["pixel_values"]).cuda()
    image_grid_thw = torch.asarray(image1["image_grid_thw"]).cuda()
    output1 = model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw).cpu()

    image2 = test_images[1].processed(model_id)
    pixel_values = torch.asarray(image2["pixel_values"]).cuda()
    image_grid_thw = torch.asarray(image2["image_grid_thw"]).cuda()
    output2 = model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw).cpu()

    for tp_degree in TP_SIZES:
        output = torch.load(f"{dump_tensors}/{model_shorthand}-image-tp{tp_degree}.pt")
        assert_similar([output1, output2], output)

    del output1, output2

    video1 = test_videos[0].processed(model_id)
    pixel_values_video = torch.asarray(video1["pixel_values_videos"]).cuda()
    video_grid_thw = torch.asarray(video1["video_grid_thw"]).cuda()
    output1 = model.get_video_features(pixel_values_videos=pixel_values_video, video_grid_thw=video_grid_thw).cpu()

    video2 = test_videos[1].processed(model_id)
    pixel_values_video2 = torch.asarray(video2["pixel_values_videos"]).cuda()
    video_grid_thw2 = torch.asarray(video2["video_grid_thw"]).cuda()
    output2 = model.get_video_features(pixel_values_videos=pixel_values_video2, video_grid_thw=video_grid_thw2).cpu()

    for tp_degree in TP_SIZES:
        output = torch.load(f"{dump_tensors}/{model_shorthand}-video-tp{tp_degree}.pt")
        assert_similar([output1, output2], output)

    del output1, output2


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
