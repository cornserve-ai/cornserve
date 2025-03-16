"""Task invoke methods and patching."""

from typing import Literal
from types import MethodType

from cornserve.frontend.tasks import Task, LLMTask
from cornserve.services.gateway.app.models import AppClasses


def patch_task_invoke(app_classes: AppClasses) -> None:
    """Patch the invoke method of tasks in the app classes."""
    for task in app_classes.config_cls.tasks.values():
        if not isinstance(task, Task):
            raise ValueError(f"Invalid task type: {type(task)}")
        if isinstance(task, LLMTask):
            object.__setattr__(task, "invoke", MethodType(llm_task_invoke, task))
        else:
            raise ValueError(f"Unsupported task type: {type(task)}")


async def llm_task_invoke(
    self: LLMTask,
    prompt: str,
    multimodal_data: list[tuple[Literal["image", "video"], str]] | None = None,
) -> str:
    """Invoke the LLM task."""
    invoke_input = self._InvokeInput(prompt=prompt, multimodal_data=multimodal_data)
    return invoke_input.model_dump_json()
