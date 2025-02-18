import gc

import torch
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

from cornserve.task_executors.eric.executor.model_loader import load_model
from cornserve.task_executors.eric.schema import Modality
from cornserve.task_executors.eric.router.processor import Processor
from cornserve.task_executors.eric.models.qwen2_vl import Qwen2VisionTransformer


def test_model_output(init_gpu_inference_env, large_image_url: str):
    init_gpu_inference_env(num_gpus=1)

    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    processed = Processor(model_id, Modality.IMAGE, 1)._do_process(large_image_url)

    # Hugging Face model output
    hf_model_llm = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        attn_implementation="flash_attention_2",
    )
    hf_model = hf_model_llm.visual.cuda()
    del hf_model_llm
    gc.collect()

    pixel_values = torch.from_numpy(processed["pixel_values"]).type(hf_model.get_dtype()).cuda()
    image_grid_thw = torch.from_numpy(processed["image_grid_thw"]).cuda()

    hf_output = hf_model(pixel_values, grid_thw=image_grid_thw)

    # Load our model
    our_model = load_model(model_id, weight_prefix="visual.")
    assert isinstance(our_model, Qwen2VisionTransformer)

    # Check if parameters are the same
    hf_params = dict(hf_model.named_parameters())
    our_params = dict(our_model.named_parameters())
    assert len(hf_params) == len(our_params)
    for hf_name, hf_param in hf_params.items():
        our_param = our_params[hf_name]
        assert hf_param.shape == our_param.shape, hf_name
        assert torch.allclose(hf_param, our_param), hf_name

    our_output = our_model(pixel_values, grid_thw=image_grid_thw)

    assert hf_output.shape == our_output.shape
    assert torch.allclose(hf_output, our_output), torch.mean(torch.abs(hf_output - our_output))
