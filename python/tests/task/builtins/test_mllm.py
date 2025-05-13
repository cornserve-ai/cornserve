from __future__ import annotations

import asyncio

import pytest

from cornserve.task.base import TaskContext, TaskInvocation, task_context
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


@pytest.mark.asyncio
async def test_mllm_record_concurrent():
    """Test multiple concurrent MLLM task invocations."""

    task = MLLMTask(model_id="llava", modalities=[Modality.IMAGE, Modality.VIDEO])
    task_input = MLLMInput(
        prompt="Hello, world!",
        multimodal_data=[("image", "http://example.com/image.jpg"), ("video", "http://example.com/video.mp4")],
    )

    async def call(task: MLLMTask, task_input: MLLMInput) -> list[TaskInvocation]:
        task_context.set(TaskContext(task_id=task.id))
        return await asyncio.create_task(call_impl(task, task_input))

    async def call_impl(task: MLLMTask, task_input: MLLMInput) -> list[TaskInvocation]:
        ctx = task_context.get()

        with ctx.record():
            _ = task.invoke(task_input)

        return ctx.invocations

    invocations1, invocations2 = await asyncio.gather(
        call(task, task_input),
        call(task, task_input),
    )

    assert len(invocations1) == 3
    assert len(invocations2) == 3

    assert invocations1[0].task == task.image_encoder
    assert invocations1[0].task_input.data_urls == ["http://example.com/image.jpg"]
    assert invocations1[1].task == task.video_encoder
    assert invocations1[1].task_input.data_urls == ["http://example.com/video.mp4"]
    assert invocations1[2].task == task.llm

    assert invocations2[0].task == task.image_encoder
    assert invocations2[0].task_input.data_urls == ["http://example.com/image.jpg"]
    assert invocations2[1].task == task.video_encoder
    assert invocations2[1].task_input.data_urls == ["http://example.com/video.mp4"]
    assert invocations2[2].task == task.llm
