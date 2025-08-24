"""Built-in task execution descriptor for Generator tasks."""

from __future__ import annotations

from typing import Any

import aiohttp

from cornserve import constants
from cornserve.services.resource import GPU
from cornserve.task.builtins.generator import GeneratorInput, GeneratorOutput, GeneratorTask
from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor
from cornserve.task_executors.descriptor.registry import DESCRIPTOR_REGISTRY
from cornserve.task_executors.geri.api import GenerationRequest, GenerationResponse, Status


class GeriDescriptor(TaskExecutionDescriptor[GeneratorTask, GeneratorInput, GeneratorOutput]):
    """Task execution descriptor for Generator tasks.

    This descriptor handles launching Geri (multimodal generator) tasks and converting between
    the external task API types and internal executor types.
    """

    def create_executor_name(self) -> str:
        """Create a name for the task executor."""
        model_name = self.task.model_id.split("/")[-1].lower()
        name = "-".join(["geri", self.task.modality, model_name]).lower()
        return name

    def get_container_image(self) -> str:
        """Get the container image name for the task executor."""
        return constants.CONTAINER_IMAGE_GERI

    def get_container_args(self, gpus: list[GPU], port: int) -> list[str]:
        """Get the container command for the task executor."""
        # fmt: off
        cmd = [
            "--model.id", self.task.model_id,
            "--model.modality", self.task.modality.value.upper(),
            "--server.port", str(port),
            "--server.max-batch-size", str(self.task.max_batch_size),
            "--sidecar.ranks", *[str(gpu.global_rank) for gpu in gpus],
        ]
        # fmt: on
        return cmd

    def get_api_url(self, base: str) -> str:
        """Get the task executor's base URL for API calls."""
        return f"{base}/generate"

    def to_request(self, task_input: GeneratorInput, task_output: GeneratorOutput) -> dict[str, Any]:
        """Convert TaskInput to a request object for the task executor."""
        embedding_data_id = task_input.embeddings.id

        req = GenerationRequest(
            embedding_data_id=embedding_data_id,
            height=task_input.height,
            width=task_input.width,
            num_inference_steps=task_input.num_inference_steps,
            skip_tokens=task_input.skip_tokens,
        )
        return req.model_dump()

    async def from_response(self, task_output: GeneratorOutput, response: aiohttp.ClientResponse) -> GeneratorOutput:
        """Convert the task executor response to TaskOutput."""
        response_data = await response.json()
        resp = GenerationResponse.model_validate(response_data)
        if resp.status == Status.SUCCESS:
            if resp.generated is None:
                raise RuntimeError("No generated content received from Geri")

            return GeneratorOutput(generated=resp.generated)
        else:
            raise RuntimeError(f"Error in generator task: {resp.error_message}")


DESCRIPTOR_REGISTRY.register(GeneratorTask, GeriDescriptor, default=True)
