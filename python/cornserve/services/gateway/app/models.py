"""Type definitions for the App Manager."""

from __future__ import annotations

import enum
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass
from types import ModuleType

from pydantic import BaseModel

from cornserve.app.base import AppConfig
from cornserve.task.base import UnitTask


class AppState(enum.StrEnum):
    """Possible states of a registered app."""

    NOT_READY = "not ready"
    READY = "ready"


@dataclass
class AppComponents:
    """Container for a registered app.

    Attributes:
        request_cls: The class that defines the app's request schema.
        response_cls: The class that defines the app's response schema.
        config_cls: The class that specifies the app's configuration.
        serve_fn: The function that implements the app's logic.
        is_streaming: Whether the app returns a streaming response.
    """

    request_cls: type[BaseModel]
    response_cls: type[BaseModel]
    config_cls: type[AppConfig]
    serve_fn: Callable[[BaseModel], Coroutine[None, None, BaseModel | AsyncIterator[BaseModel]]]
    is_streaming: bool


@dataclass
class AppDefinition:
    """Full definition of a registered app.

    Attributes:
        app_id: The ID of the app.
        module: The module that contains the app's code.
        source_code: The Python source code of the app.
        components: The components that define the app's schema and logic.
        tasks: The unit tasks discovered in the app's config.
    """

    app_id: str
    module: ModuleType
    source_code: str
    components: AppComponents
    tasks: list[UnitTask]
