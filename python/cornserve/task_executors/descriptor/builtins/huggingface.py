"""Built-in task execution descriptor for HuggingFace tasks."""

from __future__ import annotations

from typing import Any

import aiohttp

from cornserve import constants
from cornserve.services.resource import GPU
from cornserve.task.builtins.huggingface import (
    HuggingFaceQwenImageInput,
    HuggingFaceQwenImageOutput,
    HuggingFaceQwenImageTask,
    HuggingFaceQwenOmniInput,
    HuggingFaceQwenOmniOutput,
    HuggingFaceQwenOmniTask,
)
from cornserve.task.builtins.llm import OpenAIChatCompletionChunk
from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor
from cornserve.task_executors.descriptor.registry import DESCRIPTOR_REGISTRY
from cornserve.task_executors.huggingface.api import HuggingFaceRequest, HuggingFaceResponse, ModelType, Status


class HuggingFaceQwenImageDescriptor(
    TaskExecutionDescriptor[HuggingFaceQwenImageTask, HuggingFaceQwenImageInput, HuggingFaceQwenImageOutput]
):
    """Task execution descriptor for HuggingFace Qwen-Image tasks."""

    def create_executor_name(self) -> str:
        """Create a name for the task executor."""
        model_name = self.task.model_id.split("/")[-1].lower()
        return f"hf-qwen-image-{model_name}"

    def get_container_image(self) -> str:
        """Get the container image name for the task executor."""
        return constants.CONTAINER_IMAGE_HUGGINGFACE

    def get_container_args(self, gpus: list[GPU], port: int) -> list[str]:
        """Get the container command for the task executor."""
        return [
            "--model.model-type",
            ModelType.QWEN_IMAGE.value,
            "--model.id",
            self.task.model_id,
            "--server.port",
            str(port),
            "--server.max-batch-size",
            str(self.task.max_batch_size),
        ]

    def get_api_url(self, base: str) -> str:
        """Get the task executor's base URL for API calls."""
        return f"{base}/generate"

    def to_request(
        self, task_input: HuggingFaceQwenImageInput, task_output: HuggingFaceQwenImageOutput
    ) -> dict[str, Any]:
        """Convert TaskInput to a request object for the task executor."""
        return HuggingFaceRequest(**task_input.model_dump()).model_dump()

    async def from_response(
        self, task_output: HuggingFaceQwenImageOutput, response: aiohttp.ClientResponse
    ) -> HuggingFaceQwenImageOutput:
        """Convert the task executor response to TaskOutput."""
        response_data = await response.json()
        hf_response = HuggingFaceResponse.model_validate(response_data)

        if hf_response.status != Status.SUCCESS:
            raise RuntimeError(f"Error in HuggingFace Qwen-Image task: {hf_response.error_message}")

        if hf_response.image is None:
            raise RuntimeError("No image received from HuggingFace Qwen-Image task")

        return HuggingFaceQwenImageOutput(image=hf_response.image)


class HuggingFaceQwenOmniDescriptor(
    TaskExecutionDescriptor[HuggingFaceQwenOmniTask, HuggingFaceQwenOmniInput, HuggingFaceQwenOmniOutput]
):
    """Task execution descriptor for HuggingFace Qwen 2.5 Omni tasks."""

    def create_executor_name(self) -> str:
        """Create a name for the task executor."""
        model_name = self.task.model_id.split("/")[-1].lower().replace(".", "-")
        return f"hf-qwen-omni-{model_name}"

    def get_container_image(self) -> str:
        """Get the container image name for the task executor."""
        return constants.CONTAINER_IMAGE_HUGGINGFACE

    def get_container_args(self, gpus: list[GPU], port: int) -> list[str]:
        """Get the container command for the task executor."""
        return [
            "--model.model-type",
            ModelType.QWEN_OMNI.value,
            "--model.id",
            self.task.model_id,
            "--server.port",
            str(port),
            "--server.max-batch-size",
            str(self.task.max_batch_size),
        ]

    def get_api_url(self, base: str) -> str:
        """Get the task executor's base URL for API calls."""
        return f"{base}/generate"

    def to_request(
        self,
        task_input: HuggingFaceQwenOmniInput,
        task_output: HuggingFaceQwenOmniOutput,
    ) -> dict[str, Any]:
        """Convert TaskInput to a request object for the task executor."""

        return HuggingFaceRequest(**task_input.model_dump()).model_dump()

    async def from_response(
        self,
        task_output: HuggingFaceQwenOmniOutput,
        response: aiohttp.ClientResponse,
    ) -> HuggingFaceQwenOmniOutput:
        """Convert the task executor response to TaskOutput."""
        response_data = await response.json()
        hf_response = HuggingFaceResponse.model_validate(response_data)

        if hf_response.status != Status.SUCCESS:
            raise RuntimeError(f"Error in HuggingFace Qwen-Omni task: {hf_response.error_message}")

        if hf_response.audio_chunk is None:
            serialized_text_chunk = OpenAIChatCompletionChunk.model_validate(hf_response.text_chunk)
            return HuggingFaceQwenOmniOutput(
                audio_chunk=None,
                text_chunk=serialized_text_chunk,
            )
        return HuggingFaceQwenOmniOutput(audio_chunk=hf_response.audio_chunk, text_chunk=None)


# Register descriptors with the registry
DESCRIPTOR_REGISTRY.register(HuggingFaceQwenImageTask, HuggingFaceQwenImageDescriptor, default=True)
DESCRIPTOR_REGISTRY.register(HuggingFaceQwenOmniTask, HuggingFaceQwenOmniDescriptor, default=True)
