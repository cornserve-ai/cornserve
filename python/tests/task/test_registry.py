from __future__ import annotations

import pytest

from cornserve.task.registry import TASK_REGISTRY


def test_task_registry():
    """Tests whether the task registry is initialized correctly."""
    llm_task = TASK_REGISTRY.get("LLMUnitTask")
    encoder_task = TASK_REGISTRY.get("EncoderTask")

    from cornserve.task.base import Stream
    from cornserve.task.builtins.encoder import EncoderInput, EncoderOutput, EncoderTask
    from cornserve.task.builtins.llm import LLMUnitTask, OpenAIChatCompletionChunk, OpenAIChatCompletionRequest

    assert llm_task == (LLMUnitTask, OpenAIChatCompletionRequest, Stream[OpenAIChatCompletionChunk])
    assert encoder_task == (EncoderTask, EncoderInput, EncoderOutput)

    assert "_NonExistentTask" not in TASK_REGISTRY
    with pytest.raises(KeyError):
        TASK_REGISTRY.get("_NonEistentTask")
