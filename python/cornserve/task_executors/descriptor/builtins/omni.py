"""Built-in task execution descriptor for Omni tasks."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import aiohttp

from cornserve import constants
from cornserve.logging import get_logger
from cornserve.services.resource import GPU
from cornserve.task.base import Stream
from cornserve.task.builtins.omni import (
    OmniOutputChunk,
    OmniTalkerVocoderInput,
    OmniTalkerVocoderTask,
)
from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor
from cornserve.task_executors.descriptor.registry import DESCRIPTOR_REGISTRY

logger = get_logger(__name__)


async def parse_stream_to_audio_chunks(response: aiohttp.ClientResponse) -> AsyncGenerator[str]:
    """Parse the streaming response into audio chunks."""
    assert not response.closed, "Response must not be closed when parsing."
    try:
        async for line in response.content:
            line = line.decode().strip()
            if not line:
                continue

            if not line.startswith("data: "):
                logger.warning("Skipping unexpected line in OpenAI chat completion stream: %s", line)
                continue

            line = line[6:].strip()

            if line.startswith("[DONE]"):
                break

            yield OmniOutputChunk(audio_chunk=line).model_dump_json()
    finally:
        response.close()


class OmniTalkerVocoderDescriptor(
    TaskExecutionDescriptor[OmniTalkerVocoderTask, OmniTalkerVocoderInput, Stream[OmniOutputChunk]]
):
    """Task execution descriptor for Omni Talker tasks.

    This descriptor handles launching Omni Talker tasks and converting between
    the external task API types and internal executor types.
    """

    def create_executor_name(self) -> str:
        """Create a name for the task executor."""
        return "-".join(["vllm", self.task.model_id.split("/")[-1].replace(".", "-"), "talker"]).lower()

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
            "--talker-devices", *[str(gpu.local_rank) for gpu in gpus],
            "--code2wav-devices", *[str(gpu.local_rank) for gpu in gpus],
        ]
        # fmt: on
        return cmd

    def get_api_url(self, base: str) -> str:
        """Get the task executor's base URL for API calls."""
        return f"{base}/v1/chat/completions"

    def to_request(
        self,
        task_input: OmniTalkerVocoderInput,
        task_output: Stream[OmniOutputChunk],
    ) -> dict[str, Any]:
        """Convert TaskInput to a request object for the task executor."""
        request = task_input.model_dump()
        request["stream"] = True
        request["cornserve_hidden_states_forward_data_id"] = task_input.embeddings.id
        return request

    async def from_response(
        self,
        task_output: Stream[OmniOutputChunk],
        response: aiohttp.ClientResponse,
    ) -> Stream[OmniOutputChunk]:
        """Convert the task executor response to TaskOutput."""
        return Stream[OmniOutputChunk](
            async_iterator=parse_stream_to_audio_chunks(response),
            response=response,
        )


DESCRIPTOR_REGISTRY.register(OmniTalkerVocoderTask, OmniTalkerVocoderDescriptor, default=True)
