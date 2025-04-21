"""Gateway request and response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class RegisterAppRequest(BaseModel):
    """Request for registering a new application.

    Attributes:
        app_id: The unique identifier for the application.
        source_code: The Python source code of the application.
    """

    source_code: str


class AppRegistrationResponse(BaseModel):
    """Response for registering a new application.

    Attributes:
        app_id: The unique identifier for the registered application.
    """

    app_id: str


class AppRequest(BaseModel):
    """Request for invoking a registered application.

    Attributes:
        request_data: The input data for the application. Should be a valid
            JSON object that matches the `Request` schema of the application.
    """

    request_data: dict[str, Any]


class UnitTaskSpec(BaseModel):
    """Specification of a concrete instantiated unit task.

    Attributes:
        task_class_name: The name of the task class.
        task_config: Task configuration as a dictionary.
    """

    task_class_name: str
    task_config: dict[str, Any]


class UnitTaskDeploymentRequest(BaseModel):
    """Request for deploying unit tasks.

    The system has a registry of known task classes. The task class name is used
    to retrieve the task class (Pydantic BaseModel) from the resgistry, and the
    task configuration is used to instantiate the task class.

    Attributes:
        tasks: One or more unit tasks to be deployed.
    """

    tasks: list[UnitTaskSpec]


class UnitTaskDeploymentResponse(BaseModel):
    """Response for deploying unit tasks.

    Attributes:
        task_ids: The IDs of the deployed unit tasks.
    """

    task_ids: list[str]


class UnitTaskTeardownRequest(BaseModel):
    """Request for tearing down unit tasks.

    Attributes:
        task_ids: The IDs of the unit tasks to be torn down.
    """

    task_ids: list[str]
