from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Generic, TypeVar

import pytest
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

from cornserve.task.base import Stream, TaskGraphDispatch, TaskInput, TaskInvocation, TaskOutput, UnitTask
from cornserve_tasklib.task.builtins.encoder import EncoderInput, EncoderOutput, EncoderTask, Modality
from cornserve_tasklib.task.builtins.llm import (
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    LLMUnitTask,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequest,
)
from cornserve.task.forward import DataForward, Tensor

InputT = TypeVar("InputT", bound=TaskInput)
OutputT = TypeVar("OutputT", bound=TaskOutput)


# Toy classes for testing inheritance behavior
class ToyLLMTask(LLMUnitTask):
    """A toy class that inherits from LLMUnitTask to test root_unit_task_cls preservation."""

    pass


class ToyEncoderTask(EncoderTask):
    """A toy class that inherits from EncoderTask to test root_unit_task_cls preservation."""

    pass


class ToyBaseTask(UnitTask[InputT, OutputT], Generic[InputT, OutputT]):
    """A toy generic base task class to test inheritance behavior like the old LLMBaseTask."""

    model_id: str


class ToyOutput(TaskOutput):
    """A toy output for testing."""

    response: str


class ToyForwardOutput(TaskOutput):
    """A toy forward output for testing."""

    response: DataForward[str]


class ToyConcreteTask(ToyBaseTask[OpenAIChatCompletionRequest, ToyOutput]):
    """A toy concrete task that inherits from ToyBaseTask (like old LLMTask)."""

    def make_record_output(self, task_input: OpenAIChatCompletionRequest) -> ToyOutput:
        return ToyOutput(response="")


class ToyForwardTask(ToyBaseTask[OpenAIChatCompletionRequest, ToyForwardOutput]):
    """A toy forward task that inherits from ToyBaseTask (like old LLMForwardOutputTask)."""

    def make_record_output(self, task_input: OpenAIChatCompletionRequest) -> ToyForwardOutput:
        return ToyForwardOutput(response=DataForward[str]())


def test_root_unit_task_cls():
    """Tests whether the root unit task class is figured out correctly."""
    assert LLMUnitTask.root_unit_task_cls is LLMUnitTask
    assert EncoderTask.root_unit_task_cls is EncoderTask

    # Direct inheritence of existing concrete unit task
    assert ToyLLMTask.root_unit_task_cls is LLMUnitTask
    assert ToyEncoderTask.root_unit_task_cls is EncoderTask

    # Base unit task was meant for subclassing
    assert ToyConcreteTask.root_unit_task_cls is ToyBaseTask
    assert ToyForwardTask.root_unit_task_cls is ToyBaseTask


def test_serde_one():
    """Tests whether unit tasks can be serialized and deserialized."""
    invocation = TaskInvocation(
        task=LLMUnitTask(model_id="llama"),
        task_input=OpenAIChatCompletionRequest(
            model="llama",
            messages=[
                ChatCompletionMessageParam(role="user", content=[ChatCompletionContentPartTextParam(text="Hello")])
            ],
        ),
        task_output=Stream[OpenAIChatCompletionChunk](),
    )
    invocation_json = invocation.model_dump_json()

    invocation_deserialized = TaskInvocation.model_validate_json(invocation_json)
    assert invocation == invocation_deserialized


def test_serde_graph():
    """Tests whether task graph invocations can be serialized and deserialized."""
    encoder_invocation = TaskInvocation(
        task=EncoderTask(model_id="clip", modality=Modality.IMAGE),
        task_input=EncoderInput(model_id="clip", data_urls=["https://example.com/image.jpg"]),
        task_output=EncoderOutput(embeddings=[DataForward[Tensor]()]),
    )
    llm_invocation = TaskInvocation(
        task=LLMUnitTask(model_id="llama"),
        task_input=OpenAIChatCompletionRequest(
            model="llama",
            messages=[
                ChatCompletionMessageParam(role="user", content=[ChatCompletionContentPartTextParam(text="Hello")])
            ],
        ),
        task_output=Stream[OpenAIChatCompletionChunk](),
    )
    graph = TaskGraphDispatch(invocations=[encoder_invocation, llm_invocation])
    graph_json = graph.model_dump_json()

    graph_deserialized = TaskGraphDispatch.model_validate_json(graph_json)
    assert graph == graph_deserialized


def test_task_equivalence():
    """Tests whether unit task equivalence is determined correctly."""
    assert LLMUnitTask(model_id="llama").is_equivalent_to(LLMUnitTask(model_id="llama"))
    assert not LLMUnitTask(model_id="llama").is_equivalent_to(LLMUnitTask(model_id="mistral"))
    assert EncoderTask(model_id="clip", modality=Modality.IMAGE).is_equivalent_to(
        EncoderTask(model_id="clip", modality=Modality.IMAGE)
    )
    assert not EncoderTask(model_id="clip", modality=Modality.IMAGE).is_equivalent_to(
        EncoderTask(model_id="clip", modality=Modality.VIDEO)
    )


@pytest.mark.asyncio
async def test_stream():
    """Tests Stream functionality."""

    async def async_gen() -> AsyncGenerator[str]:
        for i in range(3):
            yield (
                OpenAIChatCompletionChunk(
                    id="chunk",
                    choices=[Choice(index=i, delta=ChoiceDelta(content=f"Chunk {i}"))],
                    created=1234567890,
                    model="llama",
                    object="chat.completion.chunk",
                ).model_dump_json()
                + "\n"
            )

    stream = Stream[OpenAIChatCompletionChunk](async_iterator=async_gen())

    i = 0
    async for chunk in stream:
        assert isinstance(chunk, OpenAIChatCompletionChunk)
        assert chunk.id == "chunk"
        assert chunk.created == 1234567890
        assert chunk.model == "llama"
        assert chunk.object == "chat.completion.chunk"
        assert chunk.choices[0].delta.content == f"Chunk {i}"
        i += 1
    assert i == 3
