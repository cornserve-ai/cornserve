"""Gateway request and response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class AppRegistrationRequest(BaseModel):
    """Request for registering a new application.

    Attributes:
        source_code: The Python source code of the application.
    """

    source_code: str


class AppRegistrationResponse(BaseModel):
    """Response for registering a new application.

    Attributes:
        app_id: The unique identifier for the registered application.
        task_names: The names of the unit tasks discovered in the application.
    """

    app_id: str
    task_names: list[str]


class AppInvocationRequest(BaseModel):
    """Request for invoking a registered application.

    Attributes:
        request_data: The input data for the application. Should be a valid
            JSON object that matches the `Request` schema of the application.
    """

    request_data: dict[str, Any]
