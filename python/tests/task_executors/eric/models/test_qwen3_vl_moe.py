"""Tests for the Qwen3-VL model's vision encoder."""

import torch
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration

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

model_id = "Qwen/Qwen3-VL-30B-A3B-Instruct"
model_shorthand = "qwen3_vl_moe"


def test_weight_loading() -> None:
    """Check if weights are loaded correctly."""
    hf_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(model_id, dtype="auto").visual

    init_distributed(world_size=1, rank=0)
    our_model = load_model(model_id, torch_device=torch.device("cpu"))
    destroy_distributed()

    def check_patch_embed_weight(our_name: str, our_param: torch.Tensor, hf_params: dict[str, torch.Tensor]):
        """Convert HF's Conv3d weight to Linear format and compare."""
        hf_param = hf_params.get(our_name)
        assert hf_param is not None, f"Parameter {our_name} not found in HF model"

        # HF has Conv3d weight: [out, in, kt, kh, kw]
        # We have Linear weight: [out, in*kt*kh*kw]
        if hf_param.dim() == 5:
            out_channels, in_channels, kt, kh, kw = hf_param.shape
            linear_weight = hf_param.reshape(out_channels, in_channels * kt * kh * kw)
            assert our_param.shape == linear_weight.shape, f"{our_name}: shape mismatch"
            assert torch.allclose(our_param, linear_weight), f"{our_name}: values don't match"
        else:
            # Fallback to regular comparison
            assert our_param.shape == hf_param.shape, f"{our_name}: shape mismatch"
            assert torch.allclose(our_param, hf_param), f"{our_name}: values don't match"

    assert_same_weights(
        hf_model,
        our_model,
        transformed_weights={"patch_embed.proj.weight": check_patch_embed_weight},
    )


@param_tp_size
def test_image_inference(test_images: list[ModalityData], tp_size: int, dump_tensors: str) -> None:
    """Test image inference."""
    executor = ModelExecutor(model_id=model_id, adapter_model_ids=[], tp_size=tp_size, sender_sidecar_ranks=None)

    result = executor.execute_model(batch=batch_builder(model_id, model_shorthand, test_images))

    assert result.status == Status.SUCCESS

    executor.shutdown()


@param_tp_size
def test_video_inference(test_videos: list[ModalityData], tp_size: int, dump_tensors: str) -> None:
    """Test video inference."""
    executor = ModelExecutor(model_id=model_id, adapter_model_ids=[], tp_size=tp_size, sender_sidecar_ranks=None)

    result = executor.execute_model(batch=batch_builder(model_id, model_shorthand, test_videos[:2]))

    assert result.status == Status.SUCCESS

    executor.shutdown()


@depends_on("test_image_inference", "test_video_inference")
def test_hf_reference(test_images: list[ModalityData], test_videos: list[ModalityData], dump_tensors: str) -> None:
    """Compare outputs with the Hugging Face implementation."""
    torch.set_grad_enabled(False)

    hf_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = hf_model.model.eval()
    model.visual.cuda()

    def transform_hf_output(output, grid_thw):
        final_features, deepstack_features_list = output
        combined = torch.cat([final_features] + deepstack_features_list, dim=1)

        spatial_merge_unit = model.visual.spatial_merge_unit
        seqlens = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]) // spatial_merge_unit
        return list(combined.split(seqlens.tolist(), dim=0))

    image1 = test_images[0].processed(model_id)
    pixel_values = torch.asarray(image1["pixel_values"]).to(torch.bfloat16).cuda()
    image_grid_thw = torch.asarray(image1["image_grid_thw"]).cuda()
    output1 = model.visual(pixel_values, grid_thw=image_grid_thw)
    output1 = transform_hf_output(output1, image_grid_thw)[0]

    image2 = test_images[1].processed(model_id)
    pixel_values = torch.asarray(image2["pixel_values"]).to(torch.bfloat16).cuda()
    image_grid_thw = torch.asarray(image2["image_grid_thw"]).cuda()
    output2 = model.visual(pixel_values, grid_thw=image_grid_thw)
    output2 = transform_hf_output(output2, image_grid_thw)[0]

    for tp_degree in TP_SIZES:
        output = torch.load(
            f"{dump_tensors}/{model_shorthand}-image-tp{tp_degree}.pt",
            map_location=output1.device,
        )
        assert_similar([output1, output2], output)

    del output1, output2

    video1 = test_videos[0].processed(model_id)
    pixel_values_video = torch.asarray(video1["pixel_values_videos"]).to(torch.bfloat16).cuda()
    video_grid_thw = torch.asarray(video1["video_grid_thw"]).cuda()
    output1 = model.visual(pixel_values_video, grid_thw=video_grid_thw)
    output1 = transform_hf_output(output1, video_grid_thw)[0]

    video2 = test_videos[1].processed(model_id)
    pixel_values_video = torch.asarray(video2["pixel_values_videos"]).to(torch.bfloat16).cuda()
    video_grid_thw = torch.asarray(video2["video_grid_thw"]).cuda()
    output2 = model.visual(pixel_values_video, grid_thw=video_grid_thw)
    output2 = transform_hf_output(output2, video_grid_thw)[0]

    for tp_degree in TP_SIZES:
        output = torch.load(
            f"{dump_tensors}/{model_shorthand}-video-tp{tp_degree}.pt",
            map_location=output1.device,
        )
        assert_similar([output1, output2], output)

    del output1, output2
