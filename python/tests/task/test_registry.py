from __future__ import annotations

import pytest

from cornserve.services.task_registry import TASK_CLASS_REGISTRY


def test_task_registry():
    """Tests whether the task registry is initialized correctly."""
    llm_task = TASK_CLASS_REGISTRY.get_unit_task("LLMUnitTask")
    encoder_task = TASK_CLASS_REGISTRY.get_unit_task("EncoderTask")

    from cornserve.task.base import Stream
    from cornserve_tasklib.task.unit.encoder import EncoderInput, EncoderOutput, EncoderTask
    from cornserve_tasklib.task.unit.llm import LLMUnitTask, OpenAIChatCompletionChunk, OpenAIChatCompletionRequest

    assert llm_task == (LLMUnitTask, OpenAIChatCompletionRequest, Stream[OpenAIChatCompletionChunk])
    assert encoder_task == (EncoderTask, EncoderInput, EncoderOutput)

    assert "_NonExistentTask" not in TASK_CLASS_REGISTRY
    with pytest.raises(KeyError):
        TASK_CLASS_REGISTRY.get_unit_task("_NonEistentTask")
