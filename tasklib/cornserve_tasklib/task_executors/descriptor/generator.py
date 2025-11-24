"""Built-in task execution descriptor for Generator tasks."""

from __future__ import annotations

import base64
import json
import time
from collections.abc import AsyncGenerator
from typing import Any

import aiohttp
from cornserve import constants
from cornserve.logging import get_logger
from cornserve.services.resource import GPU
from cornserve.task.base import Stream
from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor
from cornserve.task_executors.geri.api import (
    AudioGeriRequest,
    BatchGeriResponse,
    ImageGeriRequest,
    Status,
    StreamGeriResponseChunk,
)

from cornserve_tasklib.task.unit.generator import (
    AudioGeneratorInput,
    AudioGeneratorTask,
    GeneratorInput,
    GeneratorOutput,
    GeneratorTask,
)
from cornserve_tasklib.task.unit.llm import OpenAIChatCompletionChunk

logger = get_logger(__name__)


async def parse_geri_chunks(
    response: aiohttp.ClientResponse,
) -> AsyncGenerator[str]:
    buffer = b""
    async for chunk in response.content.iter_chunked(4096):
        buffer += chunk
        while b"\n" in buffer:
            line_bytes, buffer = buffer.split(b"\n", 1)

            line = line_bytes.decode().strip()
            if not line or not line.startswith("data: "):
                continue

            chunk_str = line[6:].strip()
            chunk_model = StreamGeriResponseChunk.model_validate_json(chunk_str)

            if not chunk_model.root:
                logger.info("Chunk model root was empty")
                continue

            base64_audio_data = base64.b64encode(chunk_model.root).decode()
            payload_dict = {
                "id": "0",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"audio": {"data": base64_audio_data}},
                    }
                ],
            }

            payload_str = json.dumps(payload_dict)
            completion_obj = OpenAIChatCompletionChunk.model_validate_json(payload_str)
            yield completion_obj.model_dump_json()


class GeriDescriptor(
    TaskExecutionDescriptor[GeneratorTask, GeneratorInput, GeneratorOutput]
):
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
        return f"{base}/{self.task.modality.value}/generate"

    def to_request(
        self, task_input: GeneratorInput, task_output: GeneratorOutput
    ) -> dict[str, Any]:
        """Convert TaskInput to a request object for the task executor."""
        req = ImageGeriRequest(
            embedding_data_id=task_input.embeddings.id,
            height=task_input.height,
            width=task_input.width,
            num_inference_steps=task_input.num_inference_steps,
            skip_tokens=task_input.skip_tokens,
        )
        return req.model_dump()

    async def from_response(
        self, task_output: GeneratorOutput, response: aiohttp.ClientResponse
    ) -> GeneratorOutput:
        """Convert the task executor response to TaskOutput."""
        response_data = await response.json()
        resp = BatchGeriResponse.model_validate(response_data)
        if resp.status == Status.SUCCESS:
            if resp.generated is None:
                raise RuntimeError("No generated content received from Geri")

            return GeneratorOutput(generated=resp.generated)
        else:
            raise RuntimeError(f"Error in generator task: {resp.error_message}")


class AudioGeriDescriptor(
    TaskExecutionDescriptor[
        AudioGeneratorTask, AudioGeneratorInput, Stream[OpenAIChatCompletionChunk]
    ]
):
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
        return f"{base}/{self.task.modality.value}/generate"

    def to_request(
        self,
        task_input: AudioGeneratorInput,
        task_output: Stream[OpenAIChatCompletionChunk],
    ) -> dict[str, Any]:
        """Convert TaskInput to a request object for the task executor."""
        req = AudioGeriRequest(
            embedding_data_id=task_input.embeddings.id,
            chunk_size=task_input.chunk_size,
            left_context_size=task_input.left_context_size,
        )
        return req.model_dump()

    async def from_response(
        self,
        task_output: Stream[OpenAIChatCompletionChunk],
        response: aiohttp.ClientResponse,
    ) -> Stream[OpenAIChatCompletionChunk]:
        """Convert the task executor response to TaskOutput."""
        return Stream[OpenAIChatCompletionChunk](
            async_iterator=parse_geri_chunks(response),
            response=response,
        )
