"""Provides a single function to launch a particular task executor.

Each task executor should have its enum variant in `TaskManagerType`.
Then, the task executor should subclass the `TaskExecutorLaunchInfo`
and update `

This module is here so that other module do not have to import code for
individual task executors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from cornserve import constants
from cornserve.services.resource_manager.resource import GPU
from cornserve.services.task_manager.models import EncoderConfig, LLMConfig, TaskManagerConfig


class TaskExecutorLaunchInfo(ABC):
    """Information about the task executor launch."""

    def __init__(self, task_manager_config: TaskManagerConfig) -> None:
        """Initialize the task executor launch information."""
        self.task_manager_type = task_manager_config.type
        self.task_manager_config = task_manager_config

    @staticmethod
    def from_task_manager_config(task_manager_config: TaskManagerConfig) -> TaskExecutorLaunchInfo:
        """Create a task executor launch info from a task manager config."""
        if isinstance(task_manager_config, EncoderConfig):
            return EncoderLaunchInfo(task_manager_config)
        if isinstance(task_manager_config, LLMConfig):
            return LLMLaunchInfo(task_manager_config)
        raise ValueError(f"Unknown task manager type: {task_manager_config.type}")

    @abstractmethod
    def get_executor_name(self) -> str:
        """Get the executor name for the task manager."""

    @abstractmethod
    def get_container_image(self) -> str:
        """Get the container image for the task manager."""

    @abstractmethod
    def get_container_args(self, gpus: list[GPU], port: int) -> list[str]:
        """Get the container command for the task manager."""

    def get_container_volumes(self) -> list[tuple[str, str, str]]:
        """Get the container volumes for the task manager.

        Returns:
            A list of tuples: name, host path, container path.
        """
        return [
            ("hf-cache", constants.VOLUME_HF_CACHE, "/root/.cache/huggingface"),
        ]


class EncoderLaunchInfo(TaskExecutorLaunchInfo):
    """Launch information for Eric, the multimodal data encoder task executor."""

    def __init__(self, task_manager_config: EncoderConfig) -> None:
        """Initialize the encoder launch information."""
        self.task_manager_config = task_manager_config

    def get_executor_name(self) -> str:
        """Get the executor name for the encoder task manager."""
        name = "-".join(
            [
                self.task_manager_config.type,
                *[x for x in self.task_manager_config.modalities],
                self.task_manager_config.model_id.split("/")[-1],
            ]
        ).lower()
        return name

    def get_container_image(self) -> str:
        """Get the container image for the encoder task manager."""
        return constants.CONTAINER_IMAGE_ERIC

    def get_container_args(self, gpus: list[GPU], port: int) -> list[str]:
        """Get the container arguments for the encoder task manager."""
        # fmt: off
        cmd = [
            "--model.id", self.task_manager_config.model_id,
            "--model.tp-size", str(len(gpus)),
            "--server.port", str(port),
            "--sidecar.ranks", *[str(gpu.global_rank) for gpu in gpus],
        ]
        # fmt: on
        return cmd


class LLMLaunchInfo(TaskExecutorLaunchInfo):
    """Launch information for vLLM, the LLM inference task executor."""

    def __init__(self, task_manager_config: LLMConfig) -> None:
        """Initialize the LLM launch information."""
        self.task_manager_config = task_manager_config

    def get_executor_name(self) -> str:
        """Get the executor name for the LLM task manager."""
        return "-".join(["llm", self.task_manager_config.model_id.split("/")[-1]]).lower()

    def get_container_image(self) -> str:
        """Get the container image for the LLM task manager."""
        return constants.CONTAINER_IMAGE_VLLM

    def get_container_args(self, gpus: list[GPU], port: int) -> list[str]:
        """Get the container arguments for the LLM task manager."""
        # fmt: off
        cmd = [
            self.task_manager_config.model_id,
            "--tensor-parallel-size", str(len(gpus)),
            "--port", str(port),
            "--limit-mm-per-prompt", "image=5",  # TODO: Make this configurable.
            "--cornserve-sidecar-ranks", *[str(gpu.global_rank) for gpu in gpus],
            "--enforce-eager",
        ]
        # fmt: on
        return cmd
