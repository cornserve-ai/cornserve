from __future__ import annotations

from cornserve.task.base import TaskGraphDispatch, TaskInvocation
from cornserve.task.builtins.encoder import EncoderInput, EncoderOutput, EncoderTask, Modality
from cornserve.task.builtins.llm import LLMBaseTask, LLMForwardOutputTask, LLMInput, LLMOutput, LLMTask
from cornserve.task.forward import DataForward, Tensor


def test_root_unit_task_cls():
    """Tests whether the root unit task class is figured out correctly."""
    assert LLMTask.root_unit_task_cls is LLMBaseTask
    assert LLMForwardOutputTask.root_unit_task_cls is LLMBaseTask
    assert EncoderTask.root_unit_task_cls is EncoderTask


def test_serde_one():
    """Tests whether unit tasks can be serialized and deserialized."""
    invocation = TaskInvocation(
        task=LLMTask(model_id="llama"),
        task_input=LLMInput(prompt="Hello", multimodal_data=[]),
        task_output=LLMOutput(response="Hello"),
    )
    invocation_json = invocation.model_dump_json()

    invocation_deserialized = TaskInvocation.model_validate_json(invocation_json)
    assert invocation == invocation_deserialized


def test_serde_graph():
    """Tests whether task graph invocations can be serialized and deserialized."""
    encoder_invocation = TaskInvocation(
        task=EncoderTask(model_id="clip", modality=Modality.IMAGE),
        task_input=EncoderInput(data_urls=["https://example.com/image.jpg"]),
        task_output=EncoderOutput(embeddings=[DataForward[Tensor]()]),
    )
    llm_invocation = TaskInvocation(
        task=LLMTask(model_id="llama"),
        task_input=LLMInput(prompt="Hello", multimodal_data=[("image", "https://example.com/image.jpg")]),
        task_output=LLMOutput(response="Hello"),
    )
    graph = TaskGraphDispatch(
        task_id="test-graph",
        invocations=[encoder_invocation, llm_invocation],
    )
    graph_json = graph.model_dump_json()

    graph_deserialized = TaskGraphDispatch.model_validate_json(graph_json)
    assert graph == graph_deserialized


def test_task_equivalence():
    """Tests whether unit task equivalence is determined correctly."""
    assert LLMTask(model_id="llama").is_equivalent_to(LLMTask(model_id="llama"))
    assert not LLMTask(model_id="llama").is_equivalent_to(LLMTask(model_id="mistral"))
    assert EncoderTask(model_id="clip", modality=Modality.IMAGE).is_equivalent_to(
        EncoderTask(model_id="clip", modality=Modality.IMAGE)
    )
    assert not EncoderTask(model_id="clip", modality=Modality.IMAGE).is_equivalent_to(
        EncoderTask(model_id="clip", modality=Modality.VIDEO)
    )
