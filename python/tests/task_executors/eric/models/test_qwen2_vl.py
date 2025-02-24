import gc

import torch
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

from cornserve.task_executors.eric.distributed.parallel import destroy_distributed, init_distributed
from cornserve.task_executors.eric.executor.loader import load_model
from cornserve.task_executors.eric.schema import Modality
from cornserve.task_executors.eric.router.processor import Processor
from cornserve.task_executors.eric.models.qwen2_vl import Qwen2VisionTransformer

from ..utils import assert_same_weights, InferenceTestCase


def test_weight_loading():
    init_distributed(world_size=1, rank=0, register_atexit_hook=False)

    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    # processed = Processor(model_id, Modality.IMAGE, 1)._do_process(large_image_url)

    # Hugging Face model output
    hf_model_llm = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        # attn_implementation="flash_attention_2",
    )
    hf_model = hf_model_llm.visual
    del hf_model_llm
    gc.collect()

    # pixel_values = torch.from_numpy(processed["pixel_values"]).type(hf_model.get_dtype()).cuda()
    # image_grid_thw = torch.from_numpy(processed["image_grid_thw"]).cuda()

    # hf_output = hf_model(pixel_values, grid_thw=image_grid_thw)

    # Load our model
    our_model = load_model(model_id, modality=Modality.IMAGE, weight_prefix="visual.", torch_device=torch.device("cpu"))
    # assert isinstance(our_model, Qwen2VisionTransformer)

    # Check if parameters are the same
    assert_same_weights(hf_model, our_model)

    # our_output = our_model(pixel_values, grid_thw=image_grid_thw)

    # assert hf_output.shape == our_output.shape
    # assert torch.allclose(hf_output, our_output), torch.mean(torch.abs(hf_output - our_output))


# class TensorParallelInferenceTestCase(InferenceTestCase):
#     """Test case for tensor parallel inference.
#
#     1. Weight loading test: 
#         We load the model weights from Hugging Face and our model.
#         We check if the parameters are the same.
#     2. Inference test:
#         We run the model with tensor parallel degree 1, 2, and 4.
#         Then, the outputs are compared with each other for consistency.
#     """
