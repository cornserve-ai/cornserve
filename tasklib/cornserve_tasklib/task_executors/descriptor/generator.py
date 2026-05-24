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
    DummyAudioGeneratorInput,
    DummyAudioGeneratorTask,
    DummyGeneratorOutput,
    DummyImageGeneratorInput,
    DummyImageGeneratorTask,
    ImageGeneratorInput,
    ImageGeneratorOutput,
    ImageGeneratorTask,
    QwenImageTextGeneratorInput,
    QwenImageTextGeneratorTask,
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
            logger.info("One iter through parse_geri_chunks")
            yield completion_obj.model_dump_json()


class ImageGeriDescriptor(
    TaskExecutionDescriptor[
        ImageGeneratorTask, ImageGeneratorInput, ImageGeneratorOutput
    ]
):
    """Task execution descriptor for Image Generator tasks.

    This descriptor handles launching Geri (multimodal generator) tasks and converting between
    the external task API types and internal executor types.
    """

    def get_container_envs(self, gpus: list[GPU]) -> list[tuple[str, str]]:
        """Get the additional environment variables for the task executor."""
        envs = super().get_container_envs(gpus)
        envs.append(("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"))
        return envs

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
        # Only rank 0 receives embeddings via sidecar; other SP ranks
        # get data via NCCL broadcast from rank 0.
        # fmt: off
        cmd = [
            "--model.id", self.task.model_id,
            "--model.modality", self.task.modality.value.upper(),
            "--model.sp-size", str(self.task.sp_size),
            "--server.port", str(port),
            "--server.max-batch-size", str(self.task.max_batch_size),
            "--sidecar.ranks", str(gpus[0].global_rank),
        ]
        # fmt: on
        return cmd

    def get_api_url(self, base: str) -> str:
        """Get the task executor's base URL for API calls."""
        return f"{base}/{self.task.modality.value}/generate"

    def to_request(
        self,
        task_input: ImageGeneratorInput,
        task_output: ImageGeneratorOutput,
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
        self, task_output: ImageGeneratorOutput, response: aiohttp.ClientResponse
    ) -> ImageGeneratorOutput:
        """Convert the task executor response to TaskOutput."""
        response_data = await response.json()
        resp = BatchGeriResponse.model_validate(response_data)
        if resp.status == Status.SUCCESS:
            if resp.generated is None:
                raise RuntimeError("No generated content received from Geri")

            return ImageGeneratorOutput(generated=resp.generated)
        else:
            raise RuntimeError(f"Error in generator task: {resp.error_message}")


class QwenImageTextGeriDescriptor(
    TaskExecutionDescriptor[
        QwenImageTextGeneratorTask,
        QwenImageTextGeneratorInput,
        ImageGeneratorOutput,
    ]
):
    """Task execution descriptor for direct-text Qwen-Image generation in Geri."""

    def get_container_envs(self, gpus: list[GPU]) -> list[tuple[str, str]]:
        """Get additional environment variables for the task executor."""
        envs = super().get_container_envs(gpus)
        envs.append(("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"))
        # envs.append(("CORNSERVE_GERI_QWEN_IMAGE_MAX_DOUBLE_BLOCKS", "24"))
        # envs.append(("CORNSERVE_GERI_QWEN_IMAGE_MAX_SINGLE_BLOCKS", "1"))
        return envs

    def create_executor_name(self) -> str:
        """Create a name for the task executor."""
        model_name = self.task.model_id.split("/")[-1].lower()
        return "-".join(["geri", self.task.modality, "text", model_name]).lower()

    def get_container_image(self) -> str:
        """Get the container image name for the task executor."""
        return constants.CONTAINER_IMAGE_GERI

    def get_container_args(self, gpus: list[GPU], port: int) -> list[str]:
        """Get the container command for the task executor."""
        # Only rank 0 receives sidecar inputs when embedding mode is used.
        # Direct-text mode bypasses sidecar for request data, but the rank is still required by config.
        # fmt: off
        cmd = [
            "--model.id", self.task.model_id,
            "--model.modality", self.task.modality.value.upper(),
            "--model.sp-size", str(self.task.sp_size),
            "--server.port", str(port),
            "--server.max-batch-size", str(self.task.max_batch_size),
            "--sidecar.ranks", str(gpus[0].global_rank),
        ]
        # fmt: on
        return cmd

    def get_api_url(self, base: str) -> str:
        """Get the task executor's base URL for API calls."""
        return f"{base}/{self.task.modality.value}/generate"

    def to_request(
        self,
        task_input: QwenImageTextGeneratorInput,
        task_output: ImageGeneratorOutput,
    ) -> dict[str, Any]:
        """Convert TaskInput to a request object for the task executor."""
        return {
            "prompt": task_input.prompt,
            "height": task_input.height,
            "width": task_input.width,
            "num_inference_steps": task_input.num_inference_steps,
        }

    async def from_response(
        self, task_output: ImageGeneratorOutput, response: aiohttp.ClientResponse
    ) -> ImageGeneratorOutput:
        """Convert the task executor response to TaskOutput."""
        response_data = await response.json()
        resp = BatchGeriResponse.model_validate(response_data)
        if resp.status == Status.SUCCESS:
            if resp.generated is None:
                raise RuntimeError("No generated content received from Geri")

            return ImageGeneratorOutput(generated=resp.generated)

        raise RuntimeError(f"Error in generator task: {resp.error_message}")


class AudioGeriDescriptor(
    TaskExecutionDescriptor[
        AudioGeneratorTask, AudioGeneratorInput, Stream[OpenAIChatCompletionChunk]
    ]
):
    """Task execution descriptor for Audio Generator tasks.
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


class DummyImageGeriDescriptor(
    TaskExecutionDescriptor[
        DummyImageGeneratorTask, DummyImageGeneratorInput, DummyGeneratorOutput
    ]
):
    """Task execution descriptor for dummy image generator tasks.

    Passes `--dummy` to Geri's container args so the executor generates random
    tensors instead of waiting on the Sidecar.
    """

    def create_executor_name(self) -> str:
        """Create a name for the task executor."""
        model_name = self.task.model_id.split("/")[-1].lower()
        return "-".join(
            ["dummy-geri", self.task.modality, model_name, self.task.to_profile_str()]
        ).lower()

    def get_container_image(self) -> str:
        """Get the container image name for the task executor."""
        return constants.CONTAINER_IMAGE_GERI

    def get_container_args(self, gpus: list[GPU], port: int) -> list[str]:
        """Get the container command for the task executor."""
        # Only rank 0 receives embeddings via sidecar; other SP ranks
        # get data via NCCL broadcast from rank 0.
        # fmt: off
        cmd = [
            "--model.id", self.task.model_id,
            "--model.modality", self.task.modality.value.upper(),
            "--model.sp-size", str(self.task.sp_size),
            "--server.port", str(port),
            "--server.max-batch-size", str(self.task.max_batch_size),
            "--sidecar.ranks", str(gpus[0].global_rank),
            "--dummy",
        ]
        # fmt: on
        return cmd

    def get_api_url(self, base: str) -> str:
        """Get the task executor's base URL for API calls."""
        return f"{base}/{self.task.modality.value}/generate"

    def to_request(
        self,
        task_input: DummyImageGeneratorInput,
        task_output: DummyGeneratorOutput,
    ) -> dict[str, Any]:
        """Convert TaskInput to a request object for the task executor."""
        req = ImageGeriRequest(
            embedding_data_id="dummy",
            height=task_input.height,
            width=task_input.width,
            num_inference_steps=task_input.num_inference_steps,
            skip_tokens=task_input.skip_tokens,
            dummy_seq_len=task_input.dummy_seq_len,
        )
        return req.model_dump()

    async def from_response(
        self, task_output: DummyGeneratorOutput, response: aiohttp.ClientResponse
    ) -> DummyGeneratorOutput:
        """Convert the task executor response to TaskOutput."""
        response_data = await response.json()
        resp = BatchGeriResponse.model_validate(response_data)
        if resp.status == Status.SUCCESS:
            return DummyGeneratorOutput()
        else:
            raise RuntimeError(f"Error in generator task: {resp.error_message}")


class DummyAudioGeriDescriptor(
    TaskExecutionDescriptor[
        DummyAudioGeneratorTask,
        DummyAudioGeneratorInput,
        Stream[OpenAIChatCompletionChunk],
    ]
):
    """Task execution descriptor for dummy audio generator tasks.

    Passes `--dummy` to Geri's container args so the executor generates random
    tensors instead of waiting on the Sidecar.
    """

    def create_executor_name(self) -> str:
        """Create a name for the task executor."""
        model_name = self.task.model_id.split("/")[-1].lower()
        return "-".join(
            ["dummy-geri", self.task.modality, model_name, self.task.to_profile_str()]
        ).lower()

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
            "--dummy",
        ]
        # fmt: on
        return cmd

    def get_api_url(self, base: str) -> str:
        """Get the task executor's base URL for API calls."""
        return f"{base}/{self.task.modality.value}/generate"

    def to_request(
        self,
        task_input: DummyAudioGeneratorInput,
        task_output: Stream[OpenAIChatCompletionChunk],
    ) -> dict[str, Any]:
        """Convert TaskInput to a request object for the task executor."""
        req = AudioGeriRequest(
            embedding_data_id="dummy",
            chunk_size=task_input.chunk_size,
            left_context_size=task_input.left_context_size,
            dummy_seq_len=task_input.dummy_seq_len,
        )
        return req.model_dump()

    async def from_response(
        self,
        task_output: Stream[OpenAIChatCompletionChunk],
        response: aiohttp.ClientResponse,
    ) -> Stream[OpenAIChatCompletionChunk]:
        """Convert the task executor response to TaskOutput.

        Consumes the entire generated stream from Geri (discarding audio chunks),
        then returns a Stream with an empty async iterator.
        """
        async for _ in parse_geri_chunks(response):
            pass

        async def _empty() -> AsyncGenerator[str]:
            return  # The immediate return makes this an empty function.
            yield  # The yield makes this an async generator.

        return Stream[OpenAIChatCompletionChunk](async_iterator=_empty())
