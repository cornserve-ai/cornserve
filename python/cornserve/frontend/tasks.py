"""Supported tasks and configuration options."""

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field


@runtime_checkable
class Task(Protocol):
    """Base class for tasks."""

    async def invoke(self, *args, **kwargs) -> Any:
        """Invoke the task."""
        ...


class LLMTask(Task, BaseModel):
    """A task that invokes an LLM."""

    modalities: list[Literal["text", "image", "video"]] = Field(
        default=["text"], description="The modalities to use for the task."
    )
    model_id: str = Field(description="The ID of the model to use for the task.")

    async def invoke(
        self,
        prompt: str,
        images: list[str] | None = None,
        videos: list[str] | None = None,
    ) -> str:
        """Invoke the task."""
        ...
