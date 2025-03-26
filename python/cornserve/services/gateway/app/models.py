"""Type definitions for the App Manager."""

import enum
import uuid
from dataclasses import dataclass, field
from types import ModuleType
from typing import Callable, Coroutine, Type

from cornserve.frontend.app import AppConfig, AppRequest, AppResponse


@dataclass
class AppContext:
    """Context information for the invocation of an app.

    Attributes:
        app_id: The ID of the app.
        request_id: The ID of the request.
    """

    app_id: str
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)


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

    request_cls: Type[AppRequest]
    response_cls: Type[AppResponse]
    config_cls: Type[AppConfig]
    serve_fn: Callable[[AppRequest], Coroutine[None, None, AppResponse]]


@dataclass
class AppDefinition:
    """Full definition of a registered app.

    Attributes:
        app_id: The ID of the app.
        module: The module that contains the app's code.
        source_code: The Python source code of the app.
        classes: The classes that define the app's schema and logic.
    """

    app_id: str
    module: ModuleType
    source_code: str
    classes: AppClasses
