from __future__ import annotations

import base64
import inspect
import textwrap

import pytest

from cornserve.services.task_registry.task_class_registry import TASK_CLASS_REGISTRY


def _b64(s: str) -> str:
    # Mirror CLI: utf-8 -> base64 bytes -> ascii text
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def setup_function() -> None:
    TASK_CLASS_REGISTRY.clear()


def teardown_function() -> None:
    TASK_CLASS_REGISTRY.clear()


def test_loads_and_registers_unit_task_with_io_models() -> None:
    # Use real tasklib classes via a thin wrapper module
    import cornserve_tasklib.task.unit.encoder as encoder_mod

    src = textwrap.dedent(
        f"""
        from __future__ import annotations
        from {encoder_mod.__name__} import EncoderTask, EncoderInput, EncoderOutput
        """
    )
    TASK_CLASS_REGISTRY.load_from_source(
        source_code=_b64(src),
        task_class_name="EncoderTask",
        module_name="x.mod.encoder_wrapper",
        is_unit_task=True,
    )

    task_cls, input_cls, output_cls = TASK_CLASS_REGISTRY.get_unit_task("EncoderTask")
    assert task_cls.__name__ == "EncoderTask"
    assert input_cls.__name__ == "EncoderInput"
    assert output_cls.__name__ == "EncoderOutput"

    # Ensure the module is importable via sys.modules path the loader used
    import sys

    assert encoder_mod.__name__ in sys.modules


def test_missing_generic_args_raises() -> None:
    source = textwrap.dedent(
        """
        from __future__ import annotations
        from cornserve.task.base import UnitTask

        class BadUnit(UnitTask):
            pass
        """
    ).strip()

    with pytest.raises(ValueError):
        TASK_CLASS_REGISTRY.load_from_source(
            source_code=_b64(source),
            task_class_name="BadUnit",
            module_name="x.bad.unit",
            is_unit_task=True,
        )


def test_registers_composite_task_without_io_models() -> None:
    # Use real composite task from tasklib via a thin wrapper
    import cornserve_tasklib.task.composite.llm as mllm_mod

    src = textwrap.dedent(
        f"""
        from __future__ import annotations
        from {mllm_mod.__name__} import MLLMTask
        """
    )
    TASK_CLASS_REGISTRY.load_from_source(
        source_code=_b64(src),
        task_class_name="MLLMTask",
        module_name="x.mod.mllm_wrapper",
        is_unit_task=False,
    )

    # Composite tasks aren't accessible via get_unit_task
    with pytest.raises(KeyError):
        TASK_CLASS_REGISTRY.get_unit_task("MLLMTask")


def test_contains_and_list_and_clear() -> None:
    import cornserve_tasklib.task.unit.llm as llm_mod

    src = textwrap.dedent(
        f"""
        from __future__ import annotations
        from {llm_mod.__name__} import LLMUnitTask, OpenAIChatCompletionRequest, OpenAIChatCompletionChunk
        """
    )
    TASK_CLASS_REGISTRY.load_from_source(
        source_code=_b64(src),
        task_class_name="LLMUnitTask",
        module_name="x.mod.llm_wrapper",
        is_unit_task=True,
    )

    assert "LLMUnitTask" in TASK_CLASS_REGISTRY
    names = TASK_CLASS_REGISTRY.list_registered_unit_tasks()
    assert "LLMUnitTask" in names

    TASK_CLASS_REGISTRY.clear()
    assert "LLMUnitTask" not in TASK_CLASS_REGISTRY
