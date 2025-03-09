"""Base classes for cornserve applications."""

from pydantic import BaseModel, Field

from .tasks import Task


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

    tasks: dict[str, Task] = Field(default_factory=dict, description="Dictionary of tasks that the app requires.")

    class Config:
        extra = "forbid"
