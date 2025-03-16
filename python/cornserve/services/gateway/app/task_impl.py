"""Task invoke methods and patching."""

from typing import Literal
from types import MethodType

import httpx

from cornserve import constants
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
    async with httpx.AsyncClient(timeout=60.0) as client:
        string = invoke_input.model_dump()
        response = await client.post(
            url=f"{constants.K8S_TASK_DISPATCHER_HTTP_URL}/tasks",
            json=string,
        )
        return response.text
