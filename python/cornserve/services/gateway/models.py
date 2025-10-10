"""Gateway request and response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AppRegistrationRequest(BaseModel):
    """Request for registering a new application.

    Attributes:
        source_code: The Python source code of the application.
    """

    source_code: str


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


class TaskDefinitionPayload(BaseModel):
    """Payload for a task definition."""
    source_b64: str
    task_class_name: str
    task_definition_name: str
    module_name: str
    is_unit_task: bool = True


class DescriptorDefinitionPayload(BaseModel):
    """Payload for a descriptor definition."""
    source_b64: str
    descriptor_class_name: str
    descriptor_definition_name: str
    module_name: str
    task_class_name: str


class TasksDeploymentRequest(BaseModel):
    """Request to deploy tasks (unit or composite) and descriptors from provided sources."""

    task_definitions: list[TaskDefinitionPayload] = []
    descriptor_definitions: list[DescriptorDefinitionPayload] = []

