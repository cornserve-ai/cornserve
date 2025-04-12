"""Built-in task execution descriptor for Encoder tasks."""

from __future__ import annotations

from cornserve import constants
from cornserve.services.resource_manager.resource import GPU
from cornserve.task.builtins.encoder import EncoderInput, EncoderOutput, EncoderTask
from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor
from cornserve.task_executors.descriptor.registry import DESCRIPTOR_REGISTRY
from cornserve.task_executors.eric.api import EmbeddingRequest, EmbeddingResponse


class EricDescriptor(
    TaskExecutionDescriptor[EncoderTask, EncoderInput, EncoderOutput, EmbeddingRequest, EmbeddingResponse]
):
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
            "--server.port", str(port),
            "--sidecar.ranks", *[str(gpu.global_rank) for gpu in gpus],
        ]
        # fmt: on
        return cmd

    def to_request(
        self,
        task_input: EncoderInput,
    ) -> EmbeddingRequest:
        """Convert TaskInput to a request object for the task executor."""
        # This also needs the `DataForward` objects in order to
        # populate receiver_sidecar_ranks.

    def from_response(self, response: EmbeddingResponse) -> EncoderOutput:
        """Convert the task executor response to TaskOutput."""
        if response.status == 0:
            # This needs the `DataForward` objects as is.
            return EncoderOutput(embeddings=response.embeddings)
        else:
            raise RuntimeError(f"Error in encoder task: {response.error_message}")


DESCRIPTOR_REGISTRY.register(EncoderTask, EricDescriptor, default=True)
