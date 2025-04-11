"""Built-in task execution descriptor for Encoder tasks."""

from __future__ import annotations

from typing import override

from cornserve.services.resource_manager.resource import GPU
from cornserve.task.builtins.encoder import EncoderInput, EncoderOutput, EncoderTask
from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor


class EncoderDescriptor(
    TaskExecutionDescriptor[EncoderTask, EncoderInput, EncoderOutput, EncoderInput, EncoderOutput],
):
    """Task execution descriptor for Encoder tasks."""

    def create_executor_name(self) -> str:
        """Create a name for the task executor."""
        return f"encoder-{self.task.model_id}"

    def get_container_image(self) -> str:
        """Get the container image name for the task executor."""
        return self.task.model_id

    def get_container_args(self, gpus: list[GPU], port: int) -> list[str]:
        """Get the container command for the task executor."""
        return ["python", "encoder.py", "--port", str(port)]
