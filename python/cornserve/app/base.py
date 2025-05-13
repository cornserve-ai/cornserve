"""Base classes for cornserve applications."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from cornserve.task.base import Task


class AppRequest(BaseModel):
    """Base class for application requests.

    All user-defined request classes must inherit from this.
    """


class AppResponse(BaseModel):
    """Base class for application responses.

    All user-defined response classes must inherit from this.
    """


class AppConfig(BaseModel):
    """Base class for application configuration.

    All user-defined config classes must inherit from this.
    """

    tasks: ClassVar[dict[str, Task]] = Field(
        default_factory=dict,
        description="Dictionary of tasks that the app requires.",
    )

    model_config = ConfigDict(extra="forbid")
