"""Task invoke methods and patching."""

from types import MethodType

from cornserve.frontend.tasks import Task, LLMTask
from cornserve.services.gateway.app.models import AppClasses


def patch_task_invoke(app_classes: AppClasses) -> None:
    """Patch the invoke method of tasks in the app classes."""
    for task in app_classes.config_cls.tasks.values():
        if not isinstance(task, Task):
            raise ValueError(f"Invalid task type: {type(task)}")
        if isinstance(task, LLMTask):
            task.invoke = MethodType(llm_task_invoke, task)
        else:
            raise ValueError(f"Unsupported task type: {type(task)}")


async def llm_task_invoke(
    self: LLMTask,
    prompt: str,
    images: list[str] | None = None,
    videos: list[str] | None = None,
) -> str:
    """Invoke the LLM task."""
    # TODO: Send the request to Task Dispatcher
    print(self)
    return "Hi Mom!"
