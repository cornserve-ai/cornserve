from __future__ import annotations

from cornserve.task.base import TaskContext, task_context
from cornserve.task.builtins.mllm import MLLMInput, MLLMTask, Modality
from cornserve.task.forward import DataForward, ForwardableType, Tensor


def test_mllm_record():
    """Test MLLM task invocation recording."""
    task = MLLMTask(model_id="llava", modalities=[Modality.IMAGE])
    task_input = MLLMInput(prompt="Hello, world!", multimodal_data=[("image", "http://example.com/image.jpg")])

    ctx = TaskContext(task_id="mllm-test")
    task_context.set(ctx)
    with ctx.record():
        task_output = task.invoke(task_input)

    assert isinstance(task_output.response, str)
    assert task_output.response == ""

    assert len(ctx.invocations) == 2
    assert ctx.invocations[0].task == task.image_encoder
    assert ctx.invocations[0].task_input.data_urls == ["http://example.com/image.jpg"]
    assert len(ctx.invocations[0].task_output.embeddings) == 1
    assert (
        ctx.invocations[0].task_output.embeddings[0].data_type
        == DataForward[Tensor]().data_type
        == ForwardableType.TENSOR
    )
    assert ctx.invocations[1].task_input.prompt == "Hello, world!"
    assert ctx.invocations[0].task_output.embeddings[0] == ctx.invocations[1].task_input.embeddings[0]
