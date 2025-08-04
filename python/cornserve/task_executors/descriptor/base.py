"""Base task execution descriptor class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import httpx
import kubernetes_asyncio.client as kclient
from pydantic import BaseModel

from cornserve import constants
from cornserve.services.resource import GPU
from cornserve.task.base import TaskInput, TaskOutput, UnitTask

TaskT = TypeVar("TaskT", bound=UnitTask)
InputT = TypeVar("InputT", bound=TaskInput)
OutputT = TypeVar("OutputT", bound=TaskOutput)


class TaskExecutionDescriptor(BaseModel, ABC, Generic[TaskT, InputT, OutputT]):
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

    def get_service_ports(self, gpus: list[GPU]) -> list[tuple[str, int]]:
        """Get the additional service ports for the task executor."""
        return []

    def get_container_envs(self, gpus: list[GPU]) -> list[tuple[str, str]]:
        """Get the additional environment variables for the task executor."""
        return [
            ("CUDA_VISIBLE_DEVICES", ",".join(str(gpu.local_rank) for gpu in gpus)),
        ]

    def get_kubernetes_envs(self, gpus: list[GPU]) -> list[kclient.V1EnvVar]:
        """Get the kubernetes environment variables for the task executor."""
        envs = [kclient.V1EnvVar(name=n, value=v) for n, v in self.get_container_envs(gpus)]
        return envs

    @abstractmethod
    def get_api_url(self, base: str) -> str:
        """Get the task executor's base URL for API calls.

        Args:
            base: The base URL of the task executor.
        """

    @abstractmethod
    def to_request(self, task_input: InputT, task_output: OutputT) -> dict[str, Any]:
        """Convert TaskInput to a request object for the task executor.

        The task output object is needed because this specific task executor may
        have to forward data to the next task executor, and for that, we need to
        know the destination sidecar ranks annotated in the task output.
        """

    @abstractmethod
    def from_response(self, task_output: OutputT, response: httpx.Response) -> OutputT:
        """Convert the task executor response to TaskOutput.

        In general, the `task_output` object will be deep-copied and concrete values
        will be filled in from the response.
        """
