"""Type definitions for the App Manager."""

from __future__ import annotations

import enum
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

from pydantic import BaseModel, Field

from cornserve.app.base import AppConfig, AppRequest, AppResponse
from cornserve.task.base import UnitTask


class AppState(enum.StrEnum):
    """Possible states of a registered app."""

    NOT_READY = "not ready"
    READY = "ready"


@dataclass
class AppClasses:
    """Container for a registered app.

    Attributes:
        request_cls: The class that defines the app's request schema.
        response_cls: The class that defines the app's response schema.
        config_cls: The class that specifies the app's configuration.
        serve_fn: The function that implements the app's logic.
    """

    request_cls: type[AppRequest]
    response_cls: type[AppResponse]
    config_cls: type[AppConfig]
    serve_fn: Callable[[AppRequest], Coroutine[None, None, AppResponse]]


@dataclass
class AppDefinition:
    """Full definition of a registered app.

    Attributes:
        app_id: The ID of the app.
        module: The module that contains the app's code.
        source_code: The Python source code of the app.
        classes: The classes that define the app's schema and logic.
        tasks: The unit tasks discovered in the app's config.
    """

    app_id: str
    module: ModuleType
    source_code: str
    classes: AppClasses
    tasks: list[UnitTask]


class RegistrationInitialResponse(BaseModel):
    """Initial response sent when app validation is complete."""

    type: Literal["initial"] = "initial"
    app_id: str
    task_names: list[str]


class RegistrationFinalResponse(BaseModel):
    """Final response sent when app deployment is complete."""

    type: Literal["final"] = "final"
    message: str


class RegistrationErrorResponse(BaseModel):
    """Response sent on any registration error."""

    type: Literal["error"] = "error"
    message: str


class RegistrationStatusEvent(BaseModel):
    """Wrapper for all app registration status events."""

    event: RegistrationInitialResponse | RegistrationFinalResponse | RegistrationErrorResponse = Field(
        discriminator="type"
    )
