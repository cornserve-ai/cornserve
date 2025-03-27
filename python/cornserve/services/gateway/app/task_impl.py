"""Task invoke methods and patching.

This module contains concrete implementations of various pre-defined tasks.
These functions are invoked by the App Driver when they call `invoke` on
their task objects. They send requests to the Task Dispatcher, which
coordinates task execution on the shared data plane.
"""

from contextvars import ContextVar
from types import MethodType
from typing import Literal

import httpx
from opentelemetry import trace

from cornserve import constants
from cornserve.frontend.tasks import LLMTask, Task
from cornserve.services.gateway.app.models import AppClasses, AppContext
from cornserve.services.task_dispatcher.models import TaskDispatchRequest

# Context variable to store the app and request context.
# This is set by the App Manager when the app is invoked. Right afterwards,
# the App Manager will spin up an async task to run the app, and this app
# context will be available to the app via `app_context.get()`.
# It had to be a context variable because we don't want to change the
# signature of tasks's invoke method, and thread locals are not suitable
# for async tasks.
app_context: ContextVar[AppContext] = ContextVar("app_context")
tracer = trace.get_tracer(__name__)


def patch_task_invoke(app_classes: AppClasses) -> None:
    """Patch the invoke method of tasks in the app classes."""
    for task in app_classes.config_cls.tasks.values():
        if not isinstance(task, Task):
            raise ValueError(f"Invalid task type: {type(task)}")
        if isinstance(task, LLMTask):
            # Tasks are Pydantic models, which does not allow overwriting methods.
            # So we just bypass Pydantic's `__setattr__`.
            object.__setattr__(task, "invoke", MethodType(llm_task_invoke, task))
        else:
            raise ValueError(f"Unsupported task type: {type(task)}")


@tracer.start_as_current_span("LLMTask.llm_task_invoke")
async def llm_task_invoke(
    self: LLMTask,
    prompt: str,
    multimodal_data: list[tuple[Literal["image", "video"], str]] | None = None,
) -> str:
    """Invoke the LLM task.

    Signature should be kept in sync with `cornserve.frontend.tasks.LLMTask.invoke`.
    """
    span = trace.get_current_span()
    invoke_input = self._InvokeInput(prompt=prompt, multimodal_data=multimodal_data)
    ctx = app_context.get()
    span.set_attribute("gateway.app_driver.task.task_id", self.id)
    async with httpx.AsyncClient(timeout=60.0) as client:
        request = TaskDispatchRequest(
            app_id=ctx.app_id,
            task_id=self.id,
            request_id=ctx.request_id,
            request_data=invoke_input.model_dump_json(),
        )
        span.add_event("task_dispatch.start")
        response = await client.post(
            url=f"http://{constants.K8S_TASK_DISPATCHER_HTTP_URL}/task",
            json=request.model_dump(),
        )
        response.raise_for_status()
        span.add_event("task_dispatch.done")
        return response.text
