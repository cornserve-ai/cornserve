"""Built-in task execution descriptor for Omni tasks."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import aiohttp
from cornserve import constants
from cornserve.logging import get_logger
from cornserve.services.resource import GPU
from cornserve.task.base import Stream
from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor

from cornserve_tasklib.task.unit.llm import OpenAIChatCompletionChunk
from cornserve_tasklib.task.unit.omni import (
    OmniTalkerEmbeddingResponse,
    OmniTalkerEmbeddingTask,
    OmniTalkerVocoderInput,
    OmniTalkerVocoderTask,
)

logger = get_logger(__name__)


async def parse_stream_to_audio_chunks(
    response: aiohttp.ClientResponse,
) -> AsyncGenerator[str]:
    """Parse the streaming response into audio chunks."""
    assert not response.closed, "Response must not be closed when parsing."
    try:
        buffer = b""
        # Read in larger chunks to avoid "Chunk too big" error with large base64-encoded audio
        async for chunk in response.content.iter_chunked(1024 * 1024):  # 1MB chunks
            buffer += chunk
            # Process complete lines from buffer
            while b"\n" in buffer:
                line_bytes, buffer = buffer.split(b"\n", 1)
                line = line_bytes.decode().strip()

                if not line:
                    continue

                if not line.startswith("data: "):
                    logger.warning(
                        "Skipping unexpected line in OpenAI chat completion stream: %s",
                        line[:100],
                    )
                    continue

                line = line[6:].strip()

                if line.startswith("[DONE]"):
                    return

                # each line is a chat completion chunk where some times it has a wav in delta message
                chunk = OpenAIChatCompletionChunk.model_validate_json(line)
                yield chunk.model_dump_json()
    finally:
        response.close()


class OmniTalkerVocoderDescriptor(
    TaskExecutionDescriptor[
        OmniTalkerVocoderTask, OmniTalkerVocoderInput, Stream[OpenAIChatCompletionChunk]
    ]
):
    """Task execution descriptor for Omni Talker tasks.

    This descriptor handles launching Omni Talker tasks and converting between
    the external task API types and internal executor types.
    """

    def create_executor_name(self) -> str:
        """Create a name for the task executor."""
        return "-".join(
            ["vllm", self.task.model_id.split("/")[-1].replace(".", "-"), "talker"]
        ).lower()

    def get_container_image(self) -> str:
        """Get the container image name for the task executor."""
        return constants.CONTAINER_IMAGE_VLLM_OMNI_TALKER

    def get_container_args(self, gpus: list[GPU], port: int) -> list[str]:
        """Get the container command for the task executor."""
        # fmt: off
        if len(gpus) > 1:
            raise NotImplementedError("TP not supported for Omni Talker Vocoder yet.")
        cmd = [
            self.task.model_id,
            "--port", str(port),
            "--enforce-eager",
            "--cornserve-sidecar-ranks", *[str(gpu.global_rank) for gpu in gpus],
            "--run-talker",
            "--run-vocoder",
            "--no-enable-prefix-caching",
            "--no-enable-chunked-prefill",
            "--max-num-batched-tokens", "124000",
            "--disable-chunked-mm-input",
        ]
        # fmt: on
        return cmd

    def get_api_url(self, base: str) -> str:
        """Get the task executor's base URL for API calls."""
        return f"{base}/v1/chat/completions"

    def to_request(
        self,
        task_input: OmniTalkerVocoderInput,
        task_output: Stream[OpenAIChatCompletionChunk],
    ) -> dict[str, Any]:
        """Convert TaskInput to a request object for the task executor."""
        request = task_input.model_dump(
            exclude={
                "thinker_hidden_states",
                "cornserve_embeddings",
                "cornserve_kv_transfer_params",
                "encoder_fission",
            }
        )
        request["stream"] = True
        vllm_xargs = {
            "cornserve_hidden_states_recv_id": task_input.thinker_hidden_states.id,
        }
        request["vllm_xargs"] = vllm_xargs
        # cornserve specific field for wiping out the text content
        request["cornserve_return_audio"] = True
        logger.info("Omni Talker Vocoder request: %s", request)
        return request

    async def from_response(
        self,
        task_output: Stream[OpenAIChatCompletionChunk],
        response: aiohttp.ClientResponse,
    ) -> Stream[OpenAIChatCompletionChunk]:
        """Convert the task executor response to TaskOutput."""
        return Stream[OpenAIChatCompletionChunk](
            async_iterator=parse_stream_to_audio_chunks(response),
            response=response,
        )


class OmniTalkerEmbeddingDescriptor(
    TaskExecutionDescriptor[
        OmniTalkerEmbeddingTask, OmniTalkerVocoderInput, OmniTalkerEmbeddingResponse
    ]
):
    """Task execution descriptor for Omni Talker embedding tasks.
    This descriptor handles launching Omni Talker tasks and converting between
    the external task API types and internal executor types.
    """

    def create_executor_name(self) -> str:
        """Create a name for the task executor."""
        return "-".join(
            ["vllm", self.task.model_id.split("/")[-1].replace(".", "-"), "talker"]
        ).lower()

    def get_container_image(self) -> str:
        """Get the container image name for the task executor."""
        return constants.CONTAINER_IMAGE_VLLM_OMNI_TALKER

    def get_container_args(self, gpus: list[GPU], port: int) -> list[str]:
        """Get the container command for the task executor."""
        # fmt: off
        if len(gpus) > 1:
            raise NotImplementedError("TP not supported for Omni Talker Vocoder yet.")
        cmd = [
            self.task.model_id,
            "--port", str(port),
            "--enforce-eager",
            "--cornserve-sidecar-ranks", *[str(gpu.global_rank) for gpu in gpus],
            "--run-talker",
            "--no-enable-prefix-caching",
            "--no-enable-chunked-prefill",
            "--max-num-batched-tokens", "124000",
            "--disable-chunked-mm-input",
        ]
        # fmt: on
        return cmd

    def get_api_url(self, base: str) -> str:
        """Get the task executor's base URL for API calls."""
        return f"{base}/v1/chat/completions"

    def to_request(
        self,
        task_input: OmniTalkerVocoderInput,
        task_output: OmniTalkerEmbeddingResponse,
    ) -> dict[str, Any]:
        """Convert TaskInput to a request object for the task executor."""
        request = task_input.model_dump(
            exclude={
                "thinker_hidden_states",
                "cornserve_embeddings",
                "cornserve_kv_transfer_params",
                "encoder_fission",
            }
        )
        request["stream"] = True
        vllm_xargs = {
            "cornserve_hidden_states_recv_id": task_input.thinker_hidden_states.id,
            "cornserve_residual_codes_forward_id": task_output.embeddings.id,
            "cornserve_residual_codes_forward_ranks": str(
                task_output.embeddings.dst_sidecar_ranks
            ),
        }
        request["vllm_xargs"] = vllm_xargs
        # cornserve specific field for wiping out the text content
        request["cornserve_return_audio"] = True
        logger.info("Omni Talker Vocoder embedding request: %s", request)
        return request

    async def from_response(
        self,
        task_output: OmniTalkerEmbeddingResponse,
        response: aiohttp.ClientResponse,
    ) -> OmniTalkerEmbeddingResponse:
        """Convert the task executor response to TaskOutput."""
        return OmniTalkerEmbeddingResponse(embeddings=task_output.embeddings)
