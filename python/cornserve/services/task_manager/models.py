"""Data structures for the task manager service."""

from __future__ import annotations

import enum
import uuid
from typing_extensions import override

from pydantic import BaseModel

from cornserve.frontend.tasks import Task, LLMTask
from cornserve.services.pb import task_manager_pb2


class TaskManagerType(enum.StrEnum):
    """Enumeration of task manager types.

    This class should be kept in sync with `TaskManagerType` in
    `task_manager.proto`.
    """

    ENCODER = "ENCODER"
    LLM = "LLM"

    @classmethod
    def from_pb(cls, pb: task_manager_pb2.TaskManagerType) -> TaskManagerType:
        """Convert a protobuf TaskManagerType to a TaskManagerType."""
        return TaskManagerType(task_manager_pb2.TaskManagerType.Name(pb))


class TaskManagerConfig(BaseModel):
    """Base class for task manager configuration."""

    type: TaskManagerType

    @staticmethod
    def from_task(task: Task) -> list[TaskManagerConfig]:
        """Create task manager configurations from a task."""
        configs = []
        if isinstance(task, LLMTask):
            # The text modality is handled by the LLM server.
            configs.append(LLMConfig(model_id=task.model_id))
            # All other modalities are handled by the encoder server.
            modalities = set(modality for modality in task.modalities if modality != "text")
            configs.append(EncoderConfig(model_id=task.model_id, modalities=modalities))
        else:
            raise ValueError(f"Unknown task type: {type(task)}")

        return configs

    def create_id(self) -> str:
        """Construct a unique ID for the task manager."""
        return f"{self.type}-{uuid.uuid4().hex}"


class EncoderConfig(TaskManagerConfig):
    """Configuration for the multimodal data encoder server.

    Attributes:
        model_id: The ID of the model to use for the task.
        modalities: The modalities to use for the task.
    """

    type: TaskManagerType = TaskManagerType.ENCODER

    model_id: str
    modalities: set[str] = {"image", "video"}

    @override
    def create_id(self) -> str:
        """Construct a unique ID for the task manager."""
        pieces = [
            self.type,
            self.model_id.split("/")[-1],
            "+".join(sorted(self.modalities)),
            uuid.uuid4().hex[:8],
        ]
        return "-".join(pieces)


class LLMConfig(TaskManagerConfig):
    """Configuration for the LLM server.

    Attributes:
        model_id: The ID of the model to use for the task.
    """

    type: TaskManagerType = TaskManagerType.LLM

    model_id: str

    @override
    def create_id(self) -> str:
        """Construct a unique ID for the task manager."""
        pieces = [self.type, self.model_id.split("/")[-1], uuid.uuid4().hex[:8]]
        return "-".join(pieces)
