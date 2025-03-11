"""Data structures for the task manager service."""

from __future__ import annotations

import uuid
from typing import Literal

from pydantic import BaseModel

from cornserve.frontend.tasks import Task, LLMTask


class TaskManagerConfig(BaseModel):
    """Base class for task manager configuration."""

    # Keep in sync with TaskManagerType in `task_manager.proto`.
    type: Literal["ENCODER", "LLM"]

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

    type: Literal["ENCODER", "LLM"] = "ENCODER"

    model_id: str
    modalities: set[str] = {"image"}

    def get_id(self) -> str:
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

    type: Literal["ENCODER", "LLM"] = "LLM"

    model_id: str

    def get_id(self) -> str:
        """Construct a unique ID for the task manager."""
        pieces = [self.type, self.model_id.split("/")[-1], uuid.uuid4().hex[:8]]
        return "-".join(pieces)
