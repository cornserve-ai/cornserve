"""Base task execution descriptor class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

from cornserve import constants
from cornserve.services.resource_manager.resource import GPU
from cornserve.task.base import Task, TaskInput, TaskOutput

TaskT = TypeVar("TaskT", bound=Task)
InputT = TypeVar("InputT", bound=TaskInput)
OutputT = TypeVar("OutputT", bound=TaskOutput)
RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


class TaskExecutionDescriptor(BaseModel, ABC, Generic[TaskT, InputT, OutputT, RequestT, ResponseT]):
    """Base class for task execution descriptors.

    Attributes:
        task: The task to be executed.
    """

    task: TaskT

    @abstractmethod
    def create_executor_name(self) -> str:
        """Create a name for the task executor."""

    @abstractmethod
    def get_container_image(self) -> str:
        """Get the container image name for the task executor."""

    @abstractmethod
    def get_container_args(self, gpus: list[GPU], port: int) -> list[str]:
        """Get the container command for the task executor."""

    def get_container_volumes(self) -> list[tuple[str, str, str]]:
        """Get the container volumes for the task manager.

        Returns:
            A list of tuples: name, host path, container path.
        """
        return [
            ("hf-cache", constants.VOLUME_HF_CACHE, "/root/.cache/huggingface"),
            ("shm", constants.VOLUME_SHM, "/dev/shm"),
        ]

    @abstractmethod
    def to_request(self, task_input: InputT) -> RequestT:
        """Convert TaskInput to a request object for the task executor."""

    @abstractmethod
    def from_response(self, response: ResponseT) -> OutputT:
        """Convert the task executor response to TaskOutput."""
