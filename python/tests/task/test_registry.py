from __future__ import annotations

import pytest

from cornserve.task.registry import TASK_REGISTRY


def test_task_registry():
    """Tests whether the task registry is initialized correctly."""
    llm_task = TASK_REGISTRY.get("LLMTask")
    llm_forward_output_task = TASK_REGISTRY.get("LLMForwardOutputTask")
    encoder_task = TASK_REGISTRY.get("EncoderTask")

    from cornserve.task.builtins.encoder import EncoderInput, EncoderOutput, EncoderTask
    from cornserve.task.builtins.llm import LLMForwardOutput, LLMForwardOutputTask, LLMInput, LLMOutput, LLMTask

    assert llm_task == (LLMTask, LLMInput, LLMOutput)
    assert llm_forward_output_task == (LLMForwardOutputTask, LLMInput, LLMForwardOutput)
    assert encoder_task == (EncoderTask, EncoderInput, EncoderOutput)

    assert "_NonExistentTask" not in TASK_REGISTRY
    with pytest.raises(KeyError):
        TASK_REGISTRY.get("_NonEistentTask")
