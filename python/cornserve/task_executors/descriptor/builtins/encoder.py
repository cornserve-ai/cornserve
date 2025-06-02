"""Built-in task execution descriptor for Encoder tasks."""

from __future__ import annotations

from typing import Any

from cornserve import constants
from cornserve.services.resource_manager.resource import GPU
from cornserve.task.builtins.encoder import EncoderInput, EncoderOutput, EncoderTask
from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor
from cornserve.task_executors.descriptor.registry import DESCRIPTOR_REGISTRY
from cornserve.task_executors.eric.api import EmbeddingData, EmbeddingRequest, EmbeddingResponse, Modality, Status


class EricDescriptor(TaskExecutionDescriptor[EncoderTask, EncoderInput, EncoderOutput]):
    """Task execution descriptor for Encoder tasks.

    This descriptor handles launching Eric (multimodal encoder) tasks and converting between
    the external task API types and internal executor types.
    """

    def create_executor_name(self) -> str:
        """Create a name for the task executor."""
        name = "-".join(
            [
                "eric",
                self.task.modality,
                self.task.model_id.split("/")[-1].lower(),
            ]
        ).lower()
        return name

    def get_container_image(self) -> str:
        """Get the container image name for the task executor."""
        return constants.CONTAINER_IMAGE_ERIC

    def get_container_args(self, gpus: list[GPU], port: int) -> list[str]:
        """Get the container command for the task executor."""
        # fmt: off
        cmd = [
            "--model.id", self.task.model_id,
            "--model.tp-size", str(len(gpus)),
            "--model.modality", self.task.modality.value,
            "--server.port", str(port),
            "--sidecar.ranks", *[str(gpu.global_rank) for gpu in gpus],
        ]
        # fmt: on
        return cmd

    def get_api_url(self, base: str) -> str:
        """Get the task executor's base URL for API calls."""
        return f"{base}/embeddings"

    def to_request(self, task_input: EncoderInput, task_output: EncoderOutput) -> dict[str, Any]:
        """Convert TaskInput to a request object for the task executor."""
        data: list[EmbeddingData] = []
        for url, forward in zip(task_input.data_urls, task_output.embeddings, strict=True):
            if forward.dst_sidecar_ranks is None:
                raise ValueError("Destination sidecar ranks must be specified for each forward.")
            data.append(
                EmbeddingData(
                    id=forward.id,
                    modality=Modality(self.task.modality.value),
                    url=url,
                    receiver_sidecar_ranks=forward.dst_sidecar_ranks,
                )
            )
        req = EmbeddingRequest(data=data)
        return req.model_dump()

    def from_response(self, task_output: EncoderOutput, response: dict[str, Any]) -> EncoderOutput:
        """Convert the task executor response to TaskOutput."""
        resp = EmbeddingResponse.model_validate(response)
        if resp.status == Status.SUCCESS:
            return EncoderOutput(embeddings=task_output.embeddings)
        else:
            raise RuntimeError(f"Error in encoder task: {resp.error_message}")


DESCRIPTOR_REGISTRY.register(EncoderTask, EricDescriptor, default=True)
