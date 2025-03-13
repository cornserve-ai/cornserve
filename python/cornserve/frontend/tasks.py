"""Supported tasks and configuration options."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Literal, final

from pydantic import BaseModel, Field, field_validator


class Task(ABC, BaseModel):
    """Base class for tasks.

    Attributes:
        id: The ID of the task.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)

    @abstractmethod
    async def invoke(self, *args, **kwargs) -> Any:
        """Invoke the task."""
        ...

    @final
    @staticmethod
    def from_json(type: str, json: str) -> Task:
        """Try to reconstruct the Task from a JSON string."""
        match type:
            case "LLM":
                return LLMTask.model_validate_json(json)
            case _:
                raise ValueError(f"Unknown task type: {type}")


class LLMTask(Task):
    """A task that invokes an LLM.

    Attributes:
        modalities: The modalities to use for the task. Text is required.
        model_id: The ID of the model to use for the task.
    """

    model_id: str
    modalities: set[Literal["text", "image", "video"]] = {"text"}

    @field_validator("modalities")
    @classmethod
    def _check_modalities(cls, v: set[str]) -> set[str]:
        """Check whether the modalities are valid."""
        if "text" not in v:
            raise ValueError("Text modality is required.")
        return v

    async def invoke(
        self,
        prompt: str,
        multimodal_data: list[tuple[Literal["image", "video"], str]] | None = None,
    ) -> str:
        """Invoke the task."""
        ...
