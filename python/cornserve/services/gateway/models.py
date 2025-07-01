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
    """

    app_id: str


class AppInvocationRequest(BaseModel):
    """Request for invoking a registered application.

    Attributes:
        request_data: The input data for the application. Should be a valid
            JSON object that matches the `Request` schema of the application.
    """

    request_data: dict[str, Any]


class ScaleTaskRequest(BaseModel):
    """Request to scale a unit task up or down.

    Attributes:
        task_id: The task_id of the unit task to scale.
        num_gpus: The number of GPUs to add or remove. Positive values will
            scale up, and negative values will scale down.
    """

    task_id: str
    num_gpus: int
