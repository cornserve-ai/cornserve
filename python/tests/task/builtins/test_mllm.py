from __future__ import annotations

import asyncio

import pytest

from cornserve.task.base import Stream, TaskContext, TaskInvocation, task_context
from cornserve.task.builtins.encoder import Modality
from cornserve.task.builtins.llm import (
    URL,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartVideoParam,
    ChatCompletionMessageParam,
    MLLMTask,
    OpenAIChatCompletionRequest,
)
from cornserve.task.forward import DataForward, ForwardableType, Tensor


def test_mllm_record():
    """Test MLLM task invocation recording."""
    task = MLLMTask(model_id="llava", modalities=[Modality.IMAGE])

    # Create OpenAI-compatible chat completion request with image
    task_input = OpenAIChatCompletionRequest(
        model="llava",
        messages=[
            ChatCompletionMessageParam(
                role="user",
                content=[
                    ChatCompletionContentPartTextParam(text="Hello, world!"),
                    ChatCompletionContentPartImageParam(image_url=URL(url="http://example.com/image.jpg")),
                ],
            )
        ],
    )

    ctx = TaskContext()
    task_context.set(ctx)
    with ctx.record():
        task_output = task.invoke(task_input)

    # Verify the task output is a stream
    assert isinstance(task_output, Stream)

    assert len(ctx.invocations) == 2
    assert ctx.invocations[0].task == task.encoders[Modality.IMAGE]
    assert ctx.invocations[0].task_input.data_urls == ["http://example.com/image.jpg"]
    assert len(ctx.invocations[0].task_output.embeddings) == 1
    assert (
        ctx.invocations[0].task_output.embeddings[0].data_type
        == DataForward[Tensor]().data_type
        == ForwardableType.TENSOR
    )
    assert ctx.invocations[1].task == task.llm
    assert len(ctx.invocations[1].task_input.cornserve_embeddings) == 1
    assert ctx.invocations[0].task_output.embeddings[0] == ctx.invocations[1].task_input.cornserve_embeddings[0]


@pytest.mark.asyncio
async def test_mllm_record_concurrent():
    """Test multiple concurrent MLLM task invocations."""

    task = MLLMTask(model_id="llava", modalities=[Modality.IMAGE, Modality.VIDEO])
    task_input = OpenAIChatCompletionRequest(
        model="llava",
        messages=[
            ChatCompletionMessageParam(
                role="user",
                content=[
                    ChatCompletionContentPartTextParam(text="Hello, world!"),
                    ChatCompletionContentPartImageParam(image_url=URL(url="http://example.com/image.jpg")),
                    ChatCompletionContentPartVideoParam(video_url=URL(url="http://example.com/video.mp4")),
                ],
            )
        ],
    )

    async def call(task: MLLMTask, task_input: OpenAIChatCompletionRequest) -> list[TaskInvocation]:
        task_context.set(TaskContext())
        return await asyncio.create_task(call_impl(task, task_input))

    async def call_impl(task: MLLMTask, task_input: OpenAIChatCompletionRequest) -> list[TaskInvocation]:
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

    assert invocations1[0].task == task.encoders[Modality.IMAGE]
    assert invocations1[0].task_input.data_urls == ["http://example.com/image.jpg"]
    assert invocations1[1].task == task.encoders[Modality.VIDEO]
    assert invocations1[1].task_input.data_urls == ["http://example.com/video.mp4"]
    assert invocations1[2].task == task.llm

    assert invocations2[0].task == task.encoders[Modality.IMAGE]
    assert invocations2[0].task_input.data_urls == ["http://example.com/image.jpg"]
    assert invocations2[1].task == task.encoders[Modality.VIDEO]
    assert invocations2[1].task_input.data_urls == ["http://example.com/video.mp4"]
    assert invocations2[2].task == task.llm


def test_mllm_coalesce_encoder_invocations_false():
    """Test MLLM task with coalesce_encoder_invocations=False (separate invocations)."""
    task = MLLMTask(model_id="llava", modalities=[Modality.IMAGE], coalesce_encoder_invocations=False)

    # Create request with multiple images
    task_input = OpenAIChatCompletionRequest(
        model="llava",
        messages=[
            ChatCompletionMessageParam(
                role="user",
                content=[
                    ChatCompletionContentPartTextParam(text="Compare these images:"),
                    ChatCompletionContentPartImageParam(image_url=URL(url="http://example.com/image1.jpg")),
                    ChatCompletionContentPartImageParam(image_url=URL(url="http://example.com/image2.jpg")),
                    ChatCompletionContentPartImageParam(image_url=URL(url="http://example.com/image3.jpg")),
                ],
            )
        ],
    )

    ctx = TaskContext()
    task_context.set(ctx)
    with ctx.record():
        task_output = task.invoke(task_input)

    # Verify the task output is a stream
    assert isinstance(task_output, Stream)

    # Should have 4 invocations: 3 separate encoder invocations + 1 LLM
    assert len(ctx.invocations) == 4

    # First three invocations should be separate encoder calls
    for i in range(3):
        assert ctx.invocations[i].task == task.encoders[Modality.IMAGE]
        assert ctx.invocations[i].task_input.data_urls == [f"http://example.com/image{i + 1}.jpg"]
        assert len(ctx.invocations[i].task_output.embeddings) == 1

    # Fourth invocation should be the LLM
    assert ctx.invocations[3].task == task.llm
    assert len(ctx.invocations[3].task_input.cornserve_embeddings) == 3


def test_mllm_mixed_modalities_without_coalesce():
    """Test MLLM task with mixed modalities and coalesce_encoder_invocations=False."""
    task = MLLMTask(model_id="llava", modalities=[Modality.IMAGE, Modality.VIDEO], coalesce_encoder_invocations=False)

    # Create request with multiple images and videos
    task_input = OpenAIChatCompletionRequest(
        model="llava",
        messages=[
            ChatCompletionMessageParam(
                role="user",
                content=[
                    ChatCompletionContentPartTextParam(text="Analyze this content:"),
                    ChatCompletionContentPartImageParam(image_url=URL(url="http://example.com/image1.jpg")),
                    ChatCompletionContentPartVideoParam(video_url=URL(url="http://example.com/video1.mp4")),
                    ChatCompletionContentPartImageParam(image_url=URL(url="http://example.com/image2.jpg")),
                    ChatCompletionContentPartVideoParam(video_url=URL(url="http://example.com/video2.mp4")),
                ],
            )
        ],
    )

    ctx = TaskContext()
    task_context.set(ctx)
    with ctx.record():
        task_output = task.invoke(task_input)

    # Verify the task output is a stream
    assert isinstance(task_output, Stream)

    # Should have 5 invocations: 4 separate encoder invocations + 1 LLM
    assert len(ctx.invocations) == 5

    # Verify each encoder invocation is separate and in order
    expected_calls = [
        (task.encoders[Modality.IMAGE], ["http://example.com/image1.jpg"]),
        (task.encoders[Modality.VIDEO], ["http://example.com/video1.mp4"]),
        (task.encoders[Modality.IMAGE], ["http://example.com/image2.jpg"]),
        (task.encoders[Modality.VIDEO], ["http://example.com/video2.mp4"]),
    ]

    for i, (expected_task, expected_urls) in enumerate(expected_calls):
        assert ctx.invocations[i].task == expected_task
        assert ctx.invocations[i].task_input.data_urls == expected_urls
        assert len(ctx.invocations[i].task_output.embeddings) == 1

    # Fifth invocation should be the LLM
    assert ctx.invocations[4].task == task.llm
    assert len(ctx.invocations[4].task_input.cornserve_embeddings) == 4
