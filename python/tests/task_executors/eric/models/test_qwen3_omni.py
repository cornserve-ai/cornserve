"""Tests for the Qwen3-Omni model's vision and audio encoders."""

import pytest
import torch
from transformers import Qwen3OmniMoeProcessor, Qwen3OmniMoeThinkerForConditionalGeneration
from transformers.models.auto.processing_auto import AutoProcessor

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

model_id = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
model_shorthand = "qwen3_omni_moe"


def test_weight_loading() -> None:
    """Check if weights are loaded correctly."""
    # Hugging Face model output
    hf_model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(model_id, torch_dtype="auto")

    # Load our model
    init_distributed(world_size=1, rank=0)
    our_model = load_model(model_id, torch_device=torch.device("cpu"))
    destroy_distributed()

    # Check if parameters are the same
    assert_same_weights(
        hf_model,
        our_model,
        ignored_prefixes=MODEL_REGISTRY[model_shorthand].weight.ignored_prefixes,
    )


@param_tp_size
def test_image_inference(test_images: list[ModalityData], tp_size: int, dump_tensors: str) -> None:
    """Test if inference works correctly."""
    executor = ModelExecutor(model_id=model_id, adapter_model_ids=[], tp_size=tp_size, sender_sidecar_ranks=None)

    result = executor.execute_model(batch=batch_builder(model_id, model_shorthand, test_images))

    assert result.status == Status.SUCCESS

    executor.shutdown()


@param_tp_size
def test_video_inference(test_videos: list[ModalityData], tp_size: int, dump_tensors: str) -> None:
    """Test if inference works correctly."""
    executor = ModelExecutor(model_id=model_id, adapter_model_ids=[], tp_size=tp_size, sender_sidecar_ranks=None)

    result = executor.execute_model(batch=batch_builder(model_id, model_shorthand, test_videos[:2]))

    assert result.status == Status.SUCCESS

    executor.shutdown()


@param_tp_size
def test_audio_inference(test_audios: list[ModalityData], tp_size: int, dump_tensors: str) -> None:
    """Test if inference works correctly."""
    executor = ModelExecutor(model_id=model_id, adapter_model_ids=[], tp_size=tp_size, sender_sidecar_ranks=None)

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

    hf_model = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    del hf_model.model
    model = hf_model.cuda().eval()

    processor = AutoProcessor.from_pretrained(model_id)

    # Test audio encoder
    audio1 = processor.feature_extractor(
        [test_audios[0].raw()],
        sampling_rate=processor.feature_extractor.sampling_rate,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_features = audio1["input_features"].to(torch.bfloat16).cuda()
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
    input_features = audio2["input_features"].to(torch.bfloat16).cuda()
    feature_attention_mask = audio2["attention_mask"].cuda()
    output2 = model.get_audio_features(
        input_features=input_features, feature_attention_mask=feature_attention_mask
    ).cpu()

    for tp_degree in TP_SIZES:
        output = torch.load(f"{dump_tensors}/{model_shorthand}-audio-tp{tp_degree}.pt")
        assert_similar([output1, output2], output)

    del output1, output2

    # Transformers' implementation of the vision encoder returns the final features (`final_features`),
    # i.e. the result after running embeddings through all transformer blocks as a separate list from
    # interemdiate features (`deepstack_features_list`) produced by specific layers among the blocks.
    # The Eric implementation of the vision encoder instead concatenates them into a big tensor.
    # The following function exists to enable comparing results.
    def transform_HF_output(output, grid_thw):
        final_features, deepstack_features_list = output
        combined_hf_output = torch.cat([final_features] + deepstack_features_list, dim=1)

        # unbatch the result, just like the Eric implementation.
        spatial_merge_unit = model.visual.spatial_merge_unit
        seqlens = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]) // spatial_merge_unit
        return list(combined_hf_output.split(seqlens.tolist(), dim=0))

    # Test images
    image1 = test_images[0].processed(model_id)
    pixel_values = torch.asarray(image1["pixel_values"]).to(torch.bfloat16).cuda()
    image_grid_thw = torch.asarray(image1["image_grid_thw"]).cuda()
    output1 = model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    output1 = transform_HF_output(output1, image_grid_thw)[0]

    image2 = test_images[1].processed(model_id)
    pixel_values = torch.asarray(image2["pixel_values"]).to(torch.bfloat16).cuda()
    image_grid_thw = torch.asarray(image2["image_grid_thw"]).cuda()
    output2 = model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    output2 = transform_HF_output(output2, image_grid_thw)[0]

    for tp_degree in TP_SIZES:
        output = torch.load(
            f"{dump_tensors}/{model_shorthand}-image-tp{tp_degree}.pt",
            map_location=output1.device,
        )
        assert_similar([output1, output2], output)

    del output1, output2

    # Test videos
    video1 = test_videos[0].processed(model_id)
    pixel_values_video = torch.asarray(video1["pixel_values_videos"]).to(torch.bfloat16).cuda()
    video_grid_thw = torch.asarray(video1["video_grid_thw"]).cuda()
    output1 = model.get_video_features(pixel_values_videos=pixel_values_video, video_grid_thw=video_grid_thw)
    output1 = transform_HF_output(output1, video_grid_thw)
    output1 = output1[0]

    video2 = test_videos[1].processed(model_id)
    pixel_values_video = torch.asarray(video2["pixel_values_videos"]).to(torch.bfloat16).cuda()
    video_grid_thw = torch.asarray(video2["video_grid_thw"]).cuda()
    output2 = model.get_video_features(pixel_values_videos=pixel_values_video, video_grid_thw=video_grid_thw)
    output2 = transform_HF_output(output2, video_grid_thw)[0]

    for tp_degree in TP_SIZES:
        output = torch.load(
            f"{dump_tensors}/{model_shorthand}-video-tp{tp_degree}.pt",
            map_location=output1.device,
        )
        assert_similar([output1, output2], output)

    del output1, output2


pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")
