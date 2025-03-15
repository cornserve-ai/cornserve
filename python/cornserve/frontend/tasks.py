"""Supported tasks and configuration options."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Literal, final, get_type_hints

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

    def __init_subclass__(cls, **kwargs):
        """Ensure that the task's invoke method parameters are consistent with internal models."""
        super().__init_subclass__(**kwargs)

        # Get the InvokeInput model from the class
        try:
            invoke_input_cls = cls._InvokeInput  # type: ignore
        except AttributeError as e:
            raise TypeError(f"{cls.__name__} must define an _InvokeInput model") from e

        if not issubclass(invoke_input_cls, BaseModel):
            raise TypeError(f"{cls.__name__}._InvokeInput must be a Pydantic model")

        try:
            invoke_output_cls = cls._InvokeOutput.default  # type: ignore
        except AttributeError as e:
            raise TypeError(f"{cls.__name__} must define an _InvokeOutput model") from e

        # Make sure InvokeInput and invoke are consistent
        invoke_params = get_type_hints(cls.invoke)
        invoke_model_fields = get_type_hints(invoke_input_cls)
        invoke_model_fields["return"] = invoke_output_cls

        if len(invoke_params) != len(invoke_model_fields):
            raise TypeError(f"{cls.__name__}.InvokeInput and invoke have different number of parameters")
        for name, param in invoke_params.items():
            if name == "self":
                continue
            if name not in invoke_model_fields:
                raise TypeError(f"Parameter {name} in {cls.__name__}.invoke is not in _InvokeInput")
            if param != invoke_model_fields[name]:
                raise TypeError(
                    f"Parameter {name} in {cls.__name__}.invoke ({param}) is "
                    f"not the same type as in _InvokeInput ({invoke_model_fields[name]})"
                )

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

    async def invoke(
        self,
        prompt: str,
        multimodal_data: list[tuple[Literal["image", "video"], str]] | None = None,
    ) -> str:
        """Invoke the task."""
        ...

    @field_validator("modalities")
    @classmethod
    def _check_modalities(cls, v: set[str]) -> set[str]:
        """Check whether the modalities are valid."""
        if "text" not in v:
            raise ValueError("Text modality is required.")
        return v

    class _InvokeInput(BaseModel):
        """Input model for the invoke method."""

        prompt: str
        multimodal_data: list[tuple[Literal["image", "video"], str]] | None = None

    _InvokeOutput = str
