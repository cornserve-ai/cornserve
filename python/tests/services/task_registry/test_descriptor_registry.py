from __future__ import annotations

import base64
import textwrap

import pytest

from cornserve.services.task_registry.task_class_registry import TASK_CLASS_REGISTRY
from cornserve.services.task_registry.descriptor_registry import DESCRIPTOR_REGISTRY


def _b64(s: str) -> str:
    # Mirror CLI: utf-8 -> base64 bytes -> ascii text
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def setup_function() -> None:
    TASK_CLASS_REGISTRY.clear()
    DESCRIPTOR_REGISTRY.clear()


def teardown_function() -> None:
    TASK_CLASS_REGISTRY.clear()
    DESCRIPTOR_REGISTRY.clear()


def test_descriptor_registers_immediately_when_task_is_present() -> None:
    # Use real Encoder task via a thin wrapper module
    import cornserve_tasklib.task.unit.encoder as encoder_mod
    task_source = textwrap.dedent(
        f"""
        from __future__ import annotations
        from {encoder_mod.__name__} import EncoderTask, EncoderInput, EncoderOutput
        """
    )

    # Define a simple descriptor class in-memory
    desc_source = textwrap.dedent(
        """
        from __future__ import annotations
        from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor
        from cornserve.task.base import TaskInput, TaskOutput, UnitTask

        class MyDescriptor(TaskExecutionDescriptor[UnitTask, TaskInput, TaskOutput]):
            def create_executor_name(self) -> str:
                return "exec"
            def get_container_image(self) -> str:
                return "img"
            def get_container_args(self, gpus, port: int) -> list[str]:
                return ["--port", str(port)]
            def get_api_url(self, base: str) -> str:
                return base + "/api"
            def to_request(self, task_input: TaskInput, task_output: TaskOutput) -> dict:
                return {}
            def from_response(self, task_output: TaskOutput, response):
                return task_output
        """
    )

    # Load task first so descriptor can register immediately
    TASK_CLASS_REGISTRY.load_from_source(
        source_code=_b64(task_source),
        task_class_name="EncoderTask",
        module_name="x.mod.encoder_wrapper",
        is_unit_task=True,
    )

    DESCRIPTOR_REGISTRY.load_from_source(
        source_code=_b64(desc_source),
        descriptor_class_name="MyDescriptor",
        module_name="x.mod.encoder_desc",
        task_class_name="EncoderTask",
    )

    task_cls, _, _ = TASK_CLASS_REGISTRY.get_unit_task("EncoderTask")
    # Should be retrievable as default
    desc_cls = DESCRIPTOR_REGISTRY.get(task_cls)
    assert desc_cls.__name__ == "MyDescriptor"


def test_descriptor_queued_then_bound_when_task_arrives() -> None:
    desc_source = textwrap.dedent(
        """
        from __future__ import annotations
        from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor
        from cornserve.task.base import TaskInput, TaskOutput, UnitTask

        class MyDescriptor(TaskExecutionDescriptor[UnitTask, TaskInput, TaskOutput]):
            def create_executor_name(self) -> str:
                return "exec"
            def get_container_image(self) -> str:
                return "img"
            def get_container_args(self, gpus, port: int) -> list[str]:
                return ["--port", str(port)]
            def get_api_url(self, base: str) -> str:
                return base + "/api"
            def to_request(self, task_input: TaskInput, task_output: TaskOutput) -> dict:
                return {}
            def from_response(self, task_output: TaskOutput, response):
                return task_output
        """
    )

    # Load descriptor first; task not present yet → goes pending (targeting LLMUnitTask)
    DESCRIPTOR_REGISTRY.load_from_source(
        source_code=_b64(desc_source),
        descriptor_class_name="MyDescriptor",
        module_name="x.mod.later_desc",
        task_class_name="LLMUnitTask",
    )

    # Now load a real unit task via a wrapper module
    import cornserve_tasklib.task.unit.llm as llm_mod
    task_source = textwrap.dedent(
        f"""
        from __future__ import annotations
        from {llm_mod.__name__} import LLMUnitTask, OpenAIChatCompletionRequest, OpenAIChatCompletionChunk
        """
    )

    TASK_CLASS_REGISTRY.load_from_source(
        source_code=_b64(task_source),
        task_class_name="LLMUnitTask",
        module_name="x.mod.llm_wrapper",
        is_unit_task=True,
    )

    # In production, binding is triggered by the k8s watcher; for unit tests, bind explicitly
    task_cls, _, _ = TASK_CLASS_REGISTRY.get_unit_task("LLMUnitTask")
    DESCRIPTOR_REGISTRY.bind_pending_descriptor_for_task_class(task_cls)
    desc_cls = DESCRIPTOR_REGISTRY.get(task_cls)
    assert desc_cls.__name__ == "MyDescriptor"


def test_descriptor_errors_are_informative() -> None:
    # No registration for the task
    class Dummy:
        __name__ = "NonTask"

    with pytest.raises(ValueError) as ei:
        DESCRIPTOR_REGISTRY.get(Dummy)  # type: ignore[arg-type]
    assert "No descriptors registered" in str(ei.value)


def test_default_vs_named_descriptor_resolution() -> None:
    # Prepare a real task
    import cornserve_tasklib.task.unit.encoder as encoder_mod
    task_source = textwrap.dedent(
        f"""
        from __future__ import annotations
        from {encoder_mod.__name__} import EncoderTask, EncoderInput, EncoderOutput
        """
    )
    TASK_CLASS_REGISTRY.load_from_source(
        source_code=_b64(task_source),
        task_class_name="EncoderTask",
        module_name="x.mod.encoder_wrapper",
        is_unit_task=True,
    )

    # Two descriptors for same task; first is default, second is named
    desc_default = textwrap.dedent(
        """
        from __future__ import annotations
        from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor
        from cornserve.task.base import TaskInput, TaskOutput, UnitTask

        class MyDefault(TaskExecutionDescriptor[UnitTask, TaskInput, TaskOutput]):
            def create_executor_name(self) -> str: return "d"
            def get_container_image(self) -> str: return "img"
            def get_container_args(self, gpus, port: int) -> list[str]: return [str(port)]
            def get_api_url(self, base: str) -> str: return base
            def to_request(self, task_input: TaskInput, task_output: TaskOutput) -> dict: return {}
            def from_response(self, task_output: TaskOutput, response): return task_output
        """
    )

    desc_named = textwrap.dedent(
        """
        from __future__ import annotations
        from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor
        from cornserve.task.base import TaskInput, TaskOutput, UnitTask

        class MyNamed(TaskExecutionDescriptor[UnitTask, TaskInput, TaskOutput]):
            def create_executor_name(self) -> str: return "n"
            def get_container_image(self) -> str: return "img"
            def get_container_args(self, gpus, port: int) -> list[str]: return [str(port)]
            def get_api_url(self, base: str) -> str: return base
            def to_request(self, task_input: TaskInput, task_output: TaskOutput) -> dict: return {}
            def from_response(self, task_output: TaskOutput, response): return task_output
        """
    )

    # Register default first
    DESCRIPTOR_REGISTRY.load_from_source(
        source_code=_b64(desc_default),
        descriptor_class_name="MyDefault",
        module_name="x.mod.desc_default",
        task_class_name="EncoderTask",
    )

    # Register named variant (non-default) by loading the class without default registration
    task_cls, _, _ = TASK_CLASS_REGISTRY.get_unit_task("EncoderTask")
    import sys, types
    from importlib.machinery import ModuleSpec
    module_name = "x.mod.desc_named"
    mod = types.ModuleType(module_name)
    mod.__spec__ = ModuleSpec(module_name, loader=None, is_package=False)
    mod.__package__ = module_name.rpartition('.')[0] or module_name
    mod.__file__ = f"<test:{module_name}>"
    exec(desc_named, mod.__dict__)
    sys.modules[module_name] = mod
    DESCRIPTOR_REGISTRY.register(task_cls, getattr(mod, "MyNamed"), name="MyNamed", default=False)

    # Now both should be accessible; default without name
    assert DESCRIPTOR_REGISTRY.get(task_cls).__name__ == "MyDefault"
    assert DESCRIPTOR_REGISTRY.get(task_cls, name="MyNamed").__name__ == "MyNamed"


