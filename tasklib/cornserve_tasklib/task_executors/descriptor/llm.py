"""Built-in task execution descriptor for OpenAI-compatible tasks."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from secrets import token_hex
from typing import Any, ClassVar

import aiohttp
import kubernetes_asyncio.client as kclient
from cornserve import constants
from cornserve.logging import get_logger
from cornserve.services.resource import GPU
from cornserve.task.base import Stream, TaskOutput
from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor

from cornserve_tasklib.task.unit.llm import (
    URL,
    DecodeLLMUnitTask,
    DummyMLLMUnitTask,
    LLMBaseUnitTask,
    LLMEmbeddingResponse,
    OmniMLLMUnitTask,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequest,
    PrefillChatCompletionResponse,
    PrefillLLMUnitTask,
    extract_multimodal_content,
    llm_executor_name,
)

logger = get_logger(__name__)


def _base_vllm_container_args(
    model_id: str,
    gpus: list[GPU],
    port: int,
    gpu_memory_utilization: float,
    max_num_seqs: int,
) -> list[str]:
    """Return the base vLLM container args shared across descriptors."""
    args = [
        model_id,
        "--tensor-parallel-size",
        str(len(gpus)),
        "--port",
        str(port),
        "--trust-remote-code",
        "--cornserve-sidecar-ranks",
        *[str(gpu.global_rank) for gpu in gpus],
        # XXX: Sending hidden states from vLLM to the sidecar fases device pointer errors
        # when compilation is enabled. Unsure if it's CUDA graph, torch.compile, or something else.
        "--enforce-eager",
        # XXX: When prefix caching is enabled, hidden states of the prefix that hit the cache
        # are never computed and thus never sent to the sidecar. Ideally, we want to include the
        # hidden states in the prefix cache, which V1 doesn't support yet.
        "--no-enable-prefix-caching",
        "--mm-processor-cache-type",
        "shm",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-num-seqs",
        str(max_num_seqs),
        "--api-server-count",
        str(3),
    ]
    # Qwen3-Omni's default max_model_len is too large for single-GPU deployments,
    # causing vLLM to fail during KV cache allocation.
    if "qwen3-omni" in model_id.lower():
        args.extend(["--max-model-len", str(constants.QWEN3_OMNI_MAX_MODEL_LEN)])
    if "qwen3-vl" in model_id.lower():
        args.extend(["--max-model-len", str(constants.QWEN3_VL_MAX_MODEL_LEN)])
    return args


async def parse_stream_to_completion_chunks(
    response: aiohttp.ClientResponse,
) -> AsyncGenerator[str]:
    """Parse the response stream to OpenAIChatCompletionChunk objects."""
    try:
        async for line in response.content:
            line = line.decode().strip()
            if not line:
                continue

            if not line.startswith("data: "):
                logger.warning(
                    "Skipping unexpected line in OpenAI chat completion stream: %s",
                    line,
                )
                continue

            line = line[6:].strip()

            if line.startswith("[DONE]"):
                break

            # Test validation
            try:
                _ = OpenAIChatCompletionChunk.model_validate_json(line)
            except Exception:
                logger.exception(
                    "Failed to parse OpenAIChatCompletionChunk from line: %s", line
                )
                break

            yield line

    finally:
        response.close()


class DummyVLLMDescriptor(
    TaskExecutionDescriptor[DummyMLLMUnitTask, OpenAIChatCompletionRequest, TaskOutput]
):
    """Task execution descriptor using vLLM."""

    def create_executor_name(self) -> str:
        """Create a name for the task executor."""
        return llm_executor_name(
            "dummy-vllm",
            self.task.model_id,
            self.task.receive_embeddings,
            self.task.to_profile_str(),
        )

    def get_container_image(self) -> str:
        """Get the container image name for the task executor."""
        return constants.CONTAINER_IMAGE_VLLM

    def get_container_envs(self, gpus: list[GPU]) -> list[tuple[str, str]]:
        """Get the container environment variables for the task executor."""
        envs = super().get_container_envs(gpus)
        envs.append(("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"))
        envs.append(
            ("CORNSERVE_PROFILING", "1"),
        )
        envs.append(
            ("VLLM_LOGGING_LEVEL", "DEBUG"),
        )
        # Always re-run the encoder for each request
        envs.append(
            ("CORNSERVE_NO_ENCODER_CACHE", "1"),
        )
        if self.task.receive_embeddings:
            envs.append(
                ("CORNSERVE_VLLM_DISABLE_MULTIMODAL", "1"),
            )
        return envs

    def get_container_args(self, gpus: list[GPU], port: int) -> list[str]:
        """Get the container command for the task executor."""
        args = _base_vllm_container_args(
            self.task.model_id,
            gpus,
            port,
            self.task.gpu_memory_utilization,
            self.task.max_num_seqs,
        )
        if self.task.receive_embeddings:
            args.append("--skip-mm-profiling")
        return args

    def get_api_url(self, base: str) -> str:
        """Get the task executor's base URL for API calls."""
        return f"{base}/v1/chat/completions"

    def to_request(
        self,
        task_input: OpenAIChatCompletionRequest,
        task_output: TaskOutput,
    ) -> dict[str, Any]:
        """Convert TaskInput to a request object for the task executor."""
        # If `cornserve_embeddings` is empty, the request will be sent to vLLM as is.
        # If not, we inspect the request's messages and replace multimodal data URLs
        # with Cornserve sidecar-compatible URIs (using data IDs in `DataForward`).
        # The expectation is that the number of multimodal data is the same as the
        # length of `cornserve_embeddings`.
        if self.task.receive_embeddings:
            multimodal_data = extract_multimodal_content(task_input.messages)
            for multimodal_content in multimodal_data:
                modality = multimodal_content.type.split("_")[
                    0
                ]  # e.g., "audio", "image", "video"
                data_url: URL = getattr(multimodal_content, multimodal_content.type)
                data_url.url = (
                    f"data:{modality}/uuid;data_id=xxxxxx;url={data_url.url},"
                )

        request = task_input.model_dump(
            exclude={
                "cornserve_embeddings",
                "cornserve_kv_transfer_params",
                "encoder_fission",
            }
        )
        request["stream"] = True
        return request

    async def from_response(
        self,
        task_output: TaskOutput,
        response: aiohttp.ClientResponse,
    ) -> TaskOutput:
        """Convert the response from the task executor to TaskOutput."""
        if isinstance(task_output, Stream):
            return Stream[OpenAIChatCompletionChunk](
                async_iterator=parse_stream_to_completion_chunks(response),
                response=response,
            )
        if isinstance(task_output, LLMEmbeddingResponse):
            return LLMEmbeddingResponse(embeddings=task_output.embeddings)
        raise ValueError(
            f"Expected task output to be Stream or LLMEmbeddingResponse, got {type(task_output)}"
        )

    def get_container_volumes(self) -> list[tuple[str, str, str]]:
        """Get the container volumes for the task manager.

        Returns:
            A list of tuples: name, host path, container path.
        """
        return [
            ("hf-cache", constants.VOLUME_HF_CACHE, "/root/.cache/huggingface"),
            ("shm", constants.VOLUME_SHM, "/dev/shm"),
            (
                "torch-compile-cache",
                constants.VOLUME_VLLM_EXECUTOR_CACHE,
                "/root/.cache/vllm/torch_compile_cache",
            ),
        ]


class OmniVLLMDescriptor(
    TaskExecutionDescriptor[OmniMLLMUnitTask, OpenAIChatCompletionRequest, TaskOutput]
):
    """Task execution descriptor for Omni MLLM with selective encoder disabling.

    Sets CORNSERVE_PROFILING but not CORNSERVE_VLLM_DISABLE_MULTIMODAL.
    For disabled modalities, rewrites data URLs so gpu_model_runner treats
    them as remote inputs and uses torch.randn via CORNSERVE_PROFILING.
    Non-disabled modalities keep native URLs and run through the real encoder.
    """

    def create_executor_name(self) -> str:
        """Create a name for the task executor."""
        return "-".join(
            [
                "omni-vllm",
                self.task.model_id.split("/")[-1],
                self.task._enc_flags_str(),
                self.task.to_profile_str(),
            ]
        ).lower()

    def get_container_image(self) -> str:
        """Get the container image name for the task executor."""
        return constants.CONTAINER_IMAGE_VLLM

    def get_container_envs(self, gpus: list[GPU]) -> list[tuple[str, str]]:
        """Get the container environment variables for the task executor."""
        envs = super().get_container_envs(gpus)
        envs.append(("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"))
        envs.append(("CORNSERVE_PROFILING", "1"))
        envs.append(("VLLM_LOGGING_LEVEL", "DEBUG"))
        envs.append(("CORNSERVE_NO_ENCODER_CACHE", "1"))
        # Do NOT set CORNSERVE_VLLM_DISABLE_MULTIMODAL — we want native
        # encoders to run for non-disabled modalities.
        return envs

    def get_container_args(self, gpus: list[GPU], port: int) -> list[str]:
        """Get the container command for the task executor."""
        return _base_vllm_container_args(
            self.task.model_id,
            gpus,
            port,
            self.task.gpu_memory_utilization,
            self.task.max_num_seqs,
        )

    def get_api_url(self, base: str) -> str:
        """Get the task executor's base URL for API calls."""
        return f"{base}/v1/chat/completions"

    def to_request(
        self,
        task_input: OpenAIChatCompletionRequest,
        task_output: TaskOutput,
    ) -> dict[str, Any]:
        """Convert TaskInput to a request object for the task executor.

        For disabled modalities, rewrite data URLs with a fake data_id so
        gpu_model_runner classifies them as 'remote' and serves torch.randn
        via CORNSERVE_PROFILING.  Non-disabled modalities keep their native
        URLs and run through the real encoder.
        """
        multimodal_data = extract_multimodal_content(task_input.messages)
        for multimodal_content in multimodal_data:
            modality = multimodal_content.type.split("_")[
                0
            ]  # "audio", "image", "video"
            should_disable = (
                (modality == "audio" and self.task.disable_audio_enc)
                or (modality == "image" and self.task.disable_image_enc)
                or (modality == "video" and self.task.disable_video_enc)
            )
            if should_disable:
                data_url: URL = getattr(multimodal_content, multimodal_content.type)
                data_url.url = (
                    f"data:{modality}/uuid;data_id=xxxxxx;url={data_url.url},"
                )

        request = task_input.model_dump(
            exclude={
                "cornserve_embeddings",
                "cornserve_kv_transfer_params",
                "encoder_fission",
            }
        )
        request["stream"] = True
        return request

    async def from_response(
        self,
        task_output: TaskOutput,
        response: aiohttp.ClientResponse,
    ) -> TaskOutput:
        """Convert the response from the task executor to TaskOutput."""
        if isinstance(task_output, Stream):
            return Stream[OpenAIChatCompletionChunk](
                async_iterator=parse_stream_to_completion_chunks(response),
                response=response,
            )
        raise ValueError(f"Expected task output to be Stream, got {type(task_output)}")

    def get_container_volumes(self) -> list[tuple[str, str, str]]:
        """Get the container volumes for the task manager."""
        return [
            ("hf-cache", constants.VOLUME_HF_CACHE, "/root/.cache/huggingface"),
            ("shm", constants.VOLUME_SHM, "/dev/shm"),
            (
                "torch-compile-cache",
                constants.VOLUME_VLLM_EXECUTOR_CACHE,
                "/root/.cache/vllm/torch_compile_cache",
            ),
        ]


class VLLMDescriptor(
    TaskExecutionDescriptor[LLMBaseUnitTask, OpenAIChatCompletionRequest, TaskOutput]
):
    """Task execution descriptor using vLLM."""

    def create_executor_name(self) -> str:
        """Create a name for the task executor."""
        profile_str = self.task.to_profile_str()
        if self.task.enable_prefix_caching:
            profile_str = profile_str + "+pc1"
        return llm_executor_name(
            "vllm", self.task.model_id, self.task.receive_embeddings, profile_str
        )

    def get_container_image(self) -> str:
        """Get the container image name for the task executor."""
        return constants.CONTAINER_IMAGE_VLLM

    def get_container_envs(self, gpus: list[GPU]) -> list[tuple[str, str]]:
        """Get the container environment variables for the task executor."""
        envs = super().get_container_envs(gpus)
        envs.append(
            (
                "VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME",
                f"VLLM_OBJECT_STORAGE_SHM_BUFFER_NAME_{token_hex(3)}",
            )
        )
        envs.append(("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"))
        # Always set CORNSERVE_BENCHMARKING so that time-sharing partial
        # disagg thinkers (receive_embeddings=False) tolerate encoder-thinker
        # embedding size mismatches when they receive sidecar data for
        # offloaded modalities.
        envs.append(
            ("CORNSERVE_BENCHMARKING", "1"),
        )
        if self.task.receive_embeddings:
            envs.append(
                ("CORNSERVE_VLLM_DISABLE_MULTIMODAL", "1"),
            )
        return envs

    def get_container_args(self, gpus: list[GPU], port: int) -> list[str]:
        """Get the container command for the task executor."""
        args = _base_vllm_container_args(
            self.task.model_id,
            gpus,
            port,
            self.task.gpu_memory_utilization,
            self.task.max_num_seqs,
        )
        if self.task.receive_embeddings:
            args.append("--skip-mm-profiling")
        if self.task.enable_prefix_caching:
            args.remove("--no-enable-prefix-caching")
        return args

    def get_api_url(self, base: str) -> str:
        """Get the task executor's base URL for API calls."""
        return f"{base}/v1/chat/completions"

    def to_request(
        self,
        task_input: OpenAIChatCompletionRequest,
        task_output: TaskOutput,
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
                    f"The number of multimodal data in messages {len(multimodal_data)} != "
                    f"{len(task_input.cornserve_embeddings)} the number of embeddings provided."
                )
            for multimodal_content, forward in zip(
                multimodal_data, task_input.cornserve_embeddings, strict=True
            ):
                modality = multimodal_content.type.split("_")[
                    0
                ]  # e.g., "audio", "image", "video"
                data_url: URL = getattr(multimodal_content, multimodal_content.type)
                data_url.url = (
                    f"data:{modality}/uuid;data_id={forward.id};url={data_url.url},"
                )

        request = task_input.model_dump(
            exclude={
                "cornserve_embeddings",
                "cornserve_kv_transfer_params",
                "encoder_fission",
            }
        )

        if isinstance(task_output, Stream):
            request["stream"] = True

        if isinstance(task_output, LLMEmbeddingResponse):
            vllm_xargs = {
                "cornserve_hidden_states_forward_id": task_output.embeddings.id,
                "cornserve_hidden_states_forward_ranks": str(
                    task_output.embeddings.dst_sidecar_ranks
                ),
            }
            request["vllm_xargs"] = vllm_xargs

        return request

    async def from_response(
        self,
        task_output: TaskOutput,
        response: aiohttp.ClientResponse,
    ) -> TaskOutput:
        """Convert the response from the task executor to TaskOutput."""
        if isinstance(task_output, Stream):
            return Stream[OpenAIChatCompletionChunk](
                async_iterator=parse_stream_to_completion_chunks(response),
                response=response,
            )
        if isinstance(task_output, LLMEmbeddingResponse):
            return LLMEmbeddingResponse(embeddings=task_output.embeddings)
        raise ValueError(
            f"Expected task output to be Stream or LLMEmbeddingResponse, got {type(task_output)}"
        )

    def get_container_volumes(self) -> list[tuple[str, str, str]]:
        """Get the container volumes for the task manager.

        Returns:
            A list of tuples: name, host path, container path.
        """
        return [
            ("hf-cache", constants.VOLUME_HF_CACHE, "/root/.cache/huggingface"),
            ("shm", constants.VOLUME_SHM, "/dev/shm"),
            (
                "torch-compile-cache",
                constants.VOLUME_VLLM_EXECUTOR_CACHE,
                "/root/.cache/vllm/torch_compile_cache",
            ),
        ]


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
        return llm_executor_name(
            "prefill",
            self.task.model_id,
            self.task.receive_embeddings,
            self.task.to_profile_str(),
        )

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
        envs = super().get_container_envs(gpus)
        envs.append(("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"))
        envs.extend(
            [
                # ("UCX_LOG_LEVEL", "debug"),
                # ("VLLM_LOGGING_LEVEL", "DEBUG"),
                (
                    "VLLM_NIXL_SIDE_CHANNEL_PORT",
                    str(self.NIXL_BASE_PORT + gpus[0].global_rank),
                ),
            ]
        )
        if self.task.receive_embeddings:
            envs.append(
                ("CORNSERVE_VLLM_DISABLE_MULTIMODAL", "1"),
            )
        return envs

    def get_kubernetes_envs(self, gpus: list[GPU]) -> list[kclient.V1EnvVar]:
        """Get the kubernetes environment variables for the task executor."""
        envs = [
            kclient.V1EnvVar(name=n, value=v) for n, v in self.get_container_envs(gpus)
        ]
        envs.append(
            kclient.V1EnvVar(
                name="VLLM_NIXL_SIDE_CHANNEL_HOST",
                value_from=kclient.V1EnvVarSource(
                    field_ref=kclient.V1ObjectFieldSelector(field_path="status.podIP")
                ),
            )
        )
        return envs

    def get_container_args(self, gpus: list[GPU], port: int) -> list[str]:
        """Get the container command for the task executor."""
        args = _base_vllm_container_args(
            self.task.model_id,
            gpus,
            port,
            self.task.gpu_memory_utilization,
            self.task.max_num_seqs,
        )
        if self.task.receive_embeddings:
            args.append("--skip-mm-profiling")
        args.extend(
            [
                "--kv-transfer-config",
                '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}',
            ]
        )
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
            (
                "torch-compile-cache",
                constants.VOLUME_VLLM_EXECUTOR_CACHE,
                "/root/.cache/vllm/torch_compile_cache",
            ),
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
            for multimodal_content, forward in zip(
                multimodal_data, task_input.cornserve_embeddings, strict=True
            ):
                modality = multimodal_content.type.split("_")[
                    0
                ]  # e.g., "audio", "image", "video"
                data_url: URL = getattr(multimodal_content, multimodal_content.type)
                data_url.url = (
                    f"data:{modality}/uuid;data_id={forward.id};url={data_url.url},"
                )

        # force non-streaming
        request = task_input.model_dump(
            exclude={"cornserve_embeddings", "stream_options"}
        )
        # overwrite max_completion_tokens
        request["max_completion_tokens"] = 1

        if (params := task_output.kv_transfer_params) is not None:
            request["kv_transfer_params"] = {
                "do_remote_decode": True,
                "do_remote_prefill": False,
                "remote_engine_id": None,
                "remote_block_ids": None,
                "remote_host": None,
                "remote_port": None,
            }
            vllm_xargs = {
                "cornserve_kv_transfer_params_forward_id": params.id,
                "cornserve_kv_transfer_params_forward_ranks": str(
                    params.dst_sidecar_ranks
                ),
            }
            request["vllm_xargs"] = vllm_xargs

        if (hidden_states := task_output.hidden_states) is not None:
            vllm_xargs = request.setdefault("vllm_xargs", {})
            vllm_xargs.update(
                {
                    "cornserve_hidden_states_forward_id": hidden_states.id,
                    "cornserve_hidden_states_forward_ranks": str(
                        hidden_states.dst_sidecar_ranks
                    ),
                }
            )

        return request

    async def from_response(
        self,
        task_output: PrefillChatCompletionResponse,
        response: aiohttp.ClientResponse,
    ) -> PrefillChatCompletionResponse:
        """Convert the response from the task executor to TaskOutput."""
        resp_data = await response.json()
        if "kv_transfer_params" in resp_data:
            return PrefillChatCompletionResponse(
                kv_transfer_params=task_output.kv_transfer_params
            )
        return PrefillChatCompletionResponse(hidden_states=task_output.hidden_states)


class DecodeVLLMDescriptor(
    TaskExecutionDescriptor[
        DecodeLLMUnitTask,
        OpenAIChatCompletionRequest,
        Stream[OpenAIChatCompletionChunk],
    ]
):
    """Task execution descriptor using vLLM in decode mode."""

    NIXL_BASE_PORT: ClassVar[int] = 5665

    def create_executor_name(self) -> str:
        """Create a name for the task executor."""
        return llm_executor_name(
            "decode",
            self.task.model_id,
            self.task.receive_embeddings,
            self.task.to_profile_str(),
        )

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
        envs = super().get_container_envs(gpus)
        envs.append(("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"))
        envs.extend(
            [
                # ("UCX_LOG_LEVEL", "debug"),
                # ("VLLM_LOGGING_LEVEL", "DEBUG"),
                (
                    "VLLM_NIXL_SIDE_CHANNEL_PORT",
                    str(self.NIXL_BASE_PORT + gpus[0].global_rank),
                ),
            ]
        )
        envs.append(
            ("CORNSERVE_VLLM_DISABLE_MULTIMODAL", "1"),
        )
        return envs

    def get_kubernetes_envs(self, gpus: list[GPU]) -> list[kclient.V1EnvVar]:
        """Get the kubernetes environment variables for the task executor."""
        envs = [
            kclient.V1EnvVar(name=n, value=v) for n, v in self.get_container_envs(gpus)
        ]
        envs.append(
            kclient.V1EnvVar(
                name="VLLM_NIXL_SIDE_CHANNEL_HOST",
                value_from=kclient.V1EnvVarSource(
                    field_ref=kclient.V1ObjectFieldSelector(field_path="status.podIP")
                ),
            )
        )
        return envs

    def get_container_args(self, gpus: list[GPU], port: int) -> list[str]:
        """Get the container command for the task executor."""
        args = _base_vllm_container_args(
            self.task.model_id,
            gpus,
            port,
            self.task.gpu_memory_utilization,
            self.task.max_num_seqs,
        )
        if self.task.receive_embeddings:
            args.append("--skip-mm-profiling")
        args.extend(
            [
                "--kv-transfer-config",
                '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}',
            ]
        )
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
            (
                "torch-compile-cache",
                constants.VOLUME_VLLM_EXECUTOR_CACHE,
                "/root/.cache/vllm/torch_compile_cache",
            ),
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
            for multimodal_content, forward in zip(
                multimodal_data, task_input.cornserve_embeddings, strict=True
            ):
                modality = multimodal_content.type.split("_")[
                    0
                ]  # e.g., "audio", "image", "video"
                data_url: URL = getattr(multimodal_content, multimodal_content.type)
                data_url.url = (
                    f"data:{modality}/uuid;data_id={forward.id};url={data_url.url},"
                )

        request = task_input.model_dump(
            exclude={"cornserve_embeddings", "cornserve_kv_transfer_params"}
        )
        if task_input.cornserve_kv_transfer_params is None:
            raise ValueError(
                "Task input must contain cornserve_kv_transfer_params for decode tasks."
            )
        vllm_xargs = {
            "cornserve_kv_transfer_params_recv_id": task_input.cornserve_kv_transfer_params.id,
        }
        request["vllm_xargs"] = vllm_xargs
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
