"""Type definitions for the Gateway service."""

import enum
from typing import Coroutine, Type, Callable

from cornserve.frontend.app import AppConfig, AppRequest, AppResponse


class AppState(enum.StrEnum):
    """Possible states of a registered app."""

    NOT_READY = "not ready"
    READY = "ready"


class AppClasses:
    """Container for the required classes of a registered app."""

    def __init__(
        self,
        request_cls: Type[AppRequest],
        response_cls: Type[AppResponse],
        config_cls: Type[AppConfig],
        serve_fn: Callable[[AppRequest], Coroutine[None, None, AppResponse]],
    ):
        self.request_cls = request_cls
        self.response_cls = response_cls
        self.config_cls = config_cls
        self.serve_fn = serve_fn


class AppDefinition:
    """Full definition of a registered app."""

    def __init__(
        self,
        app_id: str,
        source_code: str,
        state: AppState,
        classes: AppClasses,
    ):
        self.app_id = app_id
        self.source_code = source_code
        self.state = state
        self.classes = classes
