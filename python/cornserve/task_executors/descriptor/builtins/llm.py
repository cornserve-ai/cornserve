"""Built-in task execution descriptor for OpenAI-compatible tasks."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import httpx

from cornserve import constants
from cornserve.logging import get_logger
from cornserve.services.resource_manager.resource import GPU
from cornserve.task.base import Stream
from cornserve.task.builtins.llm import (
    LLMUnitTask,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequest,
    extract_multimodal_content,
)
from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor
from cornserve.task_executors.descriptor.registry import DESCRIPTOR_REGISTRY

logger = get_logger(__name__)


async def parse_stream_to_completion_chunks(response: httpx.Response) -> AsyncGenerator[str]:
    """Parse the response stream to OpenAIChatCompletionChunk objects."""
    aiter = response.aiter_text()
    try:
        async for line_ in aiter:
            line = line_.strip()
            if not line:
                continue

            if not line.startswith("data: "):
                logger.warning("Skipping unexpected line in OpenAI chat completion stream: %s", line)
                continue

            line = line[6:].strip()

            if line.startswith("[DONE]"):
                break

            # Test validation
            try:
                _ = OpenAIChatCompletionChunk.model_validate_json(line)
            except Exception as e:
                logger.error("Failed to parse OpenAIChatCompletionChunk from line: %s", line)
                logger.exception(e)
                break

            yield line

    finally:
        # Ensure the iterator has been fully consumed
        async for _ in aiter:
            pass
        await response.aclose()


class VLLMDescriptor(
    TaskExecutionDescriptor[LLMUnitTask, OpenAIChatCompletionRequest, Stream[OpenAIChatCompletionChunk]]
):
    """Task execution descriptor using vLLM."""

    def create_executor_name(self) -> str:
        """Create a name for the task executor."""
        return "-".join(["vllm", self.task.model_id.split("/")[-1]]).lower()

    def get_container_image(self) -> str:
        """Get the container image name for the task executor."""
        return constants.CONTAINER_IMAGE_VLLM

    def get_container_args(self, gpus: list[GPU], port: int) -> list[str]:
        """Get the container command for the task executor."""
        args = [
            self.task.model_id,
            "--tensor-parallel-size",
            str(len(gpus)),
            "--port",
            str(port),
            "--limit-mm-per-prompt",
            "image=5",  # TODO: Is this still needed?
        ]

        # If `receive_embeddings` is True, this task will be connected to an
        # `EncoderTask` to offload multimodal embedding computation and receive
        # embeddings through sidecars.
        if self.task.receive_embeddings:
            args.extend(
                [
                    "--cornserve-sidecar-ranks",
                    *[str(gpu.global_rank) for gpu in gpus],
                ]
            )

        return args

    def get_api_url(self, base: str) -> str:
        """Get the task executor's base URL for API calls."""
        return f"{base}/v1/chat/completions"

    def to_request(
        self,
        task_input: OpenAIChatCompletionRequest,
        task_output: Stream[OpenAIChatCompletionChunk],
    ) -> dict[str, Any]:
        """Convert TaskInput to a request object for the task executor."""
        # If `cornserve_embeddings` is empty, the request will be sent to vLLM as is.
        # If not, we inspect the request's messages and replace multimodal data URLs
        # with Cornserve sidecar-compatible URIs (using data IDs in `DataForward`).
        # The expectation is that the number of multimodal data is the same as the
        # length of `cornserve_embeddings`.
        multimodal_data = extract_multimodal_content(task_input.messages)
        if len(multimodal_data) != len(task_input.cornserve_embeddings):
            logger.error(
                "The number of multimodal data in messages (%d) does not match "
                "the number of embeddings provided (%d). Multimodal data: %s, Embeddings: %s",
                len(multimodal_data),
                len(task_input.cornserve_embeddings),
                multimodal_data,
                task_input.cornserve_embeddings,
            )
            raise ValueError(
                "The number of multimodal data in messages does not match the number of embeddings provided."
            )
        for multimodal_content, forward in zip(multimodal_data, task_input.cornserve_embeddings, strict=True):
            modality = multimodal_content.type.split("_")[0]  # e.g., "audio", "image", "video"
            data_url = getattr(multimodal_content, multimodal_content.type)
            data_uri = f"data:{modality}/uuid;data_id={forward.id};url={data_url},"
            setattr(multimodal_content, multimodal_content.type, data_uri)

        request = task_input.model_dump(exclude={"cornserve_embeddings"})
        request["stream"] = True
        return request

    def from_response(
        self,
        task_output: Stream[OpenAIChatCompletionChunk],
        response: httpx.Response,
    ) -> Stream[OpenAIChatCompletionChunk]:
        """Convert the response from the task executor to TaskOutput."""
        return Stream[OpenAIChatCompletionChunk](
            async_iterator=parse_stream_to_completion_chunks(response),
            response=response,
        )


DESCRIPTOR_REGISTRY.register(LLMUnitTask, VLLMDescriptor, default=True)
