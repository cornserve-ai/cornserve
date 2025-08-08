"""Built-in task execution descriptor for OpenAI-compatible tasks."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, ClassVar

import aiohttp
import kubernetes_asyncio.client as kclient

from cornserve import constants
from cornserve.logging import get_logger
from cornserve.services.resource import GPU
from cornserve.task.base import Stream
from cornserve.task.builtins.llm import (
    URL,
    DecodeLLMUnitTask,
    LLMUnitTask,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequest,
    PrefillChatCompletionResponse,
    PrefillLLMUnitTask,
    extract_multimodal_content,
)
from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor
from cornserve.task_executors.descriptor.registry import DESCRIPTOR_REGISTRY

logger = get_logger(__name__)


async def parse_stream_to_completion_chunks(response: aiohttp.ClientResponse) -> AsyncGenerator[str]:
    """Parse the response stream to OpenAIChatCompletionChunk objects."""
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

            # Test validation
            try:
                _ = OpenAIChatCompletionChunk.model_validate_json(line)
            except Exception as e:
                logger.error("Failed to parse OpenAIChatCompletionChunk from line: %s", line)
                logger.exception(e)
                break

            yield line

    finally:
        response.close()


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
            "--cornserve-sidecar-ranks",
            *[str(gpu.global_rank) for gpu in gpus],
        ]
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
        if self.task.receive_embeddings:
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
                data_url: URL = getattr(multimodal_content, multimodal_content.type)
                data_url.url = f"data:{modality}/uuid;data_id={forward.id};url={data_url.url},"

        request = task_input.model_dump(exclude={"cornserve_embeddings"})
        request["stream"] = True
        return request

    async def from_response(
        self,
        task_output: Stream[OpenAIChatCompletionChunk],
        response: aiohttp.ClientResponse,
    ) -> Stream[OpenAIChatCompletionChunk]:
        """Convert the response from the task executor to TaskOutput."""
        return Stream[OpenAIChatCompletionChunk](
            async_iterator=parse_stream_to_completion_chunks(response),
            response=response,
        )


DESCRIPTOR_REGISTRY.register(LLMUnitTask, VLLMDescriptor, default=True)


class PrefillVLLMDescriptor(
    TaskExecutionDescriptor[
        PrefillLLMUnitTask,
        OpenAIChatCompletionRequest,
        PrefillChatCompletionResponse,
    ]
):
    """Task execution descriptor using vLLM in prefill mode."""

    NIXL_BASE_PORT: ClassVar[int] = 5565

    def create_executor_name(self) -> str:
        """Create a name for the task executor."""
        return "-".join(["prefill", self.task.model_id.split("/")[-1]]).lower()

    def get_container_image(self) -> str:
        """Get the container image name for the task executor."""
        return constants.CONTAINER_IMAGE_VLLM

    def get_service_ports(self, gpus: list[GPU]) -> list[tuple[str, int]]:
        """Get the additional service ports for the task executor."""
        return [
            ("nixl", self.NIXL_BASE_PORT + gpus[0].global_rank),
        ]

    def get_container_envs(self, gpus: list[GPU]) -> list[tuple[str, str]]:
        """Get the additional environment variables for the task executor."""
        return [
            ("UCX_TLS", "cuda,rc,ib,tcp"),
            # ("UCX_TLS", "cuda,rc,ib"),
            ("UCX_NET_DEVICES", "mlx5_0:1"),
            # ("UCX_LOG_LEVEL", "debug"),
            ("CUDA_VISIBLE_DEVICES", ",".join(str(gpu.local_rank) for gpu in gpus)),
            # ("VLLM_LOGGING_LEVEL", "DEBUG"),
            ("VLLM_NIXL_SIDE_CHANNEL_PORT", str(self.NIXL_BASE_PORT + gpus[0].global_rank)),
        ]

    def get_kubernetes_envs(self, gpus: list[GPU]) -> list[kclient.V1EnvVar]:
        """Get the kubernetes environment variables for the task executor."""
        envs = [kclient.V1EnvVar(name=n, value=v) for n, v in self.get_container_envs(gpus)]
        envs.append(
            kclient.V1EnvVar(
                name="VLLM_NIXL_SIDE_CHANNEL_HOST",
                value_from=kclient.V1EnvVarSource(field_ref=kclient.V1ObjectFieldSelector(field_path="status.podIP")),
            )
        )
        return envs

    def get_container_args(self, gpus: list[GPU], port: int) -> list[str]:
        """Get the container command for the task executor."""
        args = [
            self.task.model_id,
            "--tensor-parallel-size",
            str(len(gpus)),
            "--port",
            str(port),
            "--kv-transfer-config",
            '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}',
            # need to forward KV transfer parameters to a decode instance
            "--cornserve-sidecar-ranks",
            *[str(gpu.global_rank) for gpu in gpus],
        ]
        return args

    def get_container_volumes(self) -> list[tuple[str, str, str]]:
        """Get the container volumes for the task manager.

        Returns:
            A list of tuples: name, host path, container path.
        """
        return [
            ("infiniband-class", "/sys/class/infiniband", "/sys/class/infiniband"),
            ("infiniband-dev", "/dev/infiniband", "/dev/infiniband"),
            ("hf-cache", constants.VOLUME_HF_CACHE, "/root/.cache/huggingface"),
            ("shm", constants.VOLUME_SHM, "/dev/shm"),
        ]

    def get_api_url(self, base: str) -> str:
        """Get the task executor's base URL for API calls."""
        return f"{base}/v1/chat/completions"

    def to_request(
        self,
        task_input: OpenAIChatCompletionRequest,
        task_output: PrefillChatCompletionResponse,
    ) -> dict[str, Any]:
        """Convert TaskInput to a request object for the task executor."""
        # If `cornserve_embeddings` is empty, the request will be sent to vLLM as is.
        # If not, we inspect the request's messages and replace multimodal data URLs
        # with Cornserve sidecar-compatible URIs (using data IDs in `DataForward`).
        # The expectation is that the number of multimodal data is the same as the
        # length of `cornserve_embeddings`.
        if self.task.receive_embeddings:
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
                data_url: URL = getattr(multimodal_content, multimodal_content.type)
                data_url.url = f"data:{modality}/uuid;data_id={forward.id};url={data_url.url},"

        # force non-streaming
        request = task_input.model_dump(exclude={"cornserve_embeddings", "stream_options"})
        # overwrite max_completion_tokens
        request["max_completion_tokens"] = 1
        request["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
        request["cornserve_kv_transfer_send_params"] = {
            "id": task_output.kv_transfer_params.id,
            "receiver_sidecar_ranks": task_output.kv_transfer_params.dst_sidecar_ranks,
        }
        return request

    async def from_response(
        self,
        task_output: PrefillChatCompletionResponse,
        response: aiohttp.ClientResponse,
    ) -> PrefillChatCompletionResponse:
        """Convert the response from the task executor to TaskOutput."""
        resp_data = await response.json()
        if "kv_transfer_params" not in resp_data:
            raise ValueError("Response does not contain kv_transfer_params.")
        return PrefillChatCompletionResponse(kv_transfer_params=task_output.kv_transfer_params)


DESCRIPTOR_REGISTRY.register(PrefillLLMUnitTask, PrefillVLLMDescriptor, default=True)


class DecodeVLLMDescriptor(
    TaskExecutionDescriptor[DecodeLLMUnitTask, OpenAIChatCompletionRequest, Stream[OpenAIChatCompletionChunk]]
):
    """Task execution descriptor using vLLM in decode mode."""

    NIXL_BASE_PORT: ClassVar[int] = 5665

    def create_executor_name(self) -> str:
        """Create a name for the task executor."""
        return "-".join(["decode", self.task.model_id.split("/")[-1]]).lower()

    def get_container_image(self) -> str:
        """Get the container image name for the task executor."""
        return constants.CONTAINER_IMAGE_VLLM

    def get_service_ports(self, gpus: list[GPU]) -> list[tuple[str, int]]:
        """Get the additional service ports for the task executor."""
        return [
            ("nixl", self.NIXL_BASE_PORT + gpus[0].global_rank),
        ]

    def get_container_envs(self, gpus: list[GPU]) -> list[tuple[str, str]]:
        """Get the additional environment variables for the task executor."""
        return [
            ("UCX_TLS", "cuda,rc,ib,tcp"),
            # ("UCX_TLS", "cuda,rc,ib"),
            ("UCX_NET_DEVICES", "mlx5_0:1"),
            # ("UCX_LOG_LEVEL", "debug"),
            ("CUDA_VISIBLE_DEVICES", ",".join(str(gpu.local_rank) for gpu in gpus)),
            # ("VLLM_LOGGING_LEVEL", "DEBUG"),
            ("VLLM_NIXL_SIDE_CHANNEL_PORT", str(self.NIXL_BASE_PORT + gpus[0].global_rank)),
        ]

    def get_kubernetes_envs(self, gpus: list[GPU]) -> list[kclient.V1EnvVar]:
        """Get the kubernetes environment variables for the task executor."""
        envs = [kclient.V1EnvVar(name=n, value=v) for n, v in self.get_container_envs(gpus)]
        envs.append(
            kclient.V1EnvVar(
                name="VLLM_NIXL_SIDE_CHANNEL_HOST",
                value_from=kclient.V1EnvVarSource(field_ref=kclient.V1ObjectFieldSelector(field_path="status.podIP")),
            )
        )
        return envs

    def get_container_args(self, gpus: list[GPU], port: int) -> list[str]:
        """Get the container command for the task executor."""
        args = [
            self.task.model_id,
            "--tensor-parallel-size",
            str(len(gpus)),
            "--port",
            str(port),
            "--kv-transfer-config",
            '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}',
            # need to receive KV transfer parameters from a decode instance
            "--cornserve-sidecar-ranks",
            *[str(gpu.global_rank) for gpu in gpus],
        ]

        return args

    def get_container_volumes(self) -> list[tuple[str, str, str]]:
        """Get the container volumes for the task manager.

        Returns:
            A list of tuples: name, host path, container path.
        """
        return [
            ("infiniband-class", "/sys/class/infiniband", "/sys/class/infiniband"),
            ("infiniband-dev", "/dev/infiniband", "/dev/infiniband"),
            ("hf-cache", constants.VOLUME_HF_CACHE, "/root/.cache/huggingface"),
            ("shm", constants.VOLUME_SHM, "/dev/shm"),
        ]

    def get_api_url(self, base: str) -> str:
        """Get the task executor's base URL for API calls."""
        return f"{base}/v1/chat/completions"

    def to_request(
        self,
        task_input: OpenAIChatCompletionRequest,
        task_output: Stream[OpenAIChatCompletionChunk],
    ) -> dict[str, Any]:
        """Convert TaskInput to a request object for the task executor."""
        if self.task.receive_embeddings:
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
                data_url: URL = getattr(multimodal_content, multimodal_content.type)
                data_url.url = f"data:{modality}/uuid;data_id={forward.id};url={data_url.url},"

        request = task_input.model_dump(exclude={"cornserve_embeddings", "cornserve_kv_transfer_params"})
        if task_input.cornserve_kv_transfer_params is None:
            raise ValueError("Task input must contain cornserve_kv_transfer_params for decode tasks.")
        request["cornserve_kv_transfer_recv_params"] = {
            "id": task_input.cornserve_kv_transfer_params.id,
        }
        request["stream"] = True
        return request

    async def from_response(
        self,
        task_output: Stream[OpenAIChatCompletionChunk],
        response: aiohttp.ClientResponse,
    ) -> Stream[OpenAIChatCompletionChunk]:
        """Convert the response from the task executor to TaskOutput."""
        return Stream[OpenAIChatCompletionChunk](
            async_iterator=parse_stream_to_completion_chunks(response),
            response=response,
        )


DESCRIPTOR_REGISTRY.register(DecodeLLMUnitTask, DecodeVLLMDescriptor, default=True)
