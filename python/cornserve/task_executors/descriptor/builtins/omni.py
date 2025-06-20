"""Built-in task execution descriptor for Omni tasks."""

from __future__ import annotations

from typing import Any

from cornserve import constants
from cornserve.services.resource_manager.resource import GPU
from cornserve.task.builtins.omni import (
    OmniTalkerInput,
    OmniTalkerLLMTask,
    OmniTalkerOutput,
    OmniThinkerInput,
    OmniThinkerLLMTask,
    OmniThinkerOutput,
)
from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor
from cornserve.task_executors.descriptor.registry import DESCRIPTOR_REGISTRY


class OmniThinkerDescriptor(
    TaskExecutionDescriptor[
        OmniThinkerLLMTask,
        OmniThinkerInput,
        OmniThinkerOutput,
    ],
):
    """Task execution descriptor for Omni Thinker tasks.

    This descriptor handles launching Omni Thinker tasks and converting between
    the external task API types and internal executor types.
    """

    def create_executor_name(self) -> str:
        """Create a name for the task executor."""
        return "-".join(["vllm", self.task.model_id.split("/")[-1].replace(".", "-"), "thinker"]).lower()

    def get_container_image(self) -> str:
        """Get the container image name for the task executor."""
        return constants.CONTAINER_IMAGE_VLLM

    def get_container_args(self, gpus: list[GPU], port: int) -> list[str]:
        """Get the container command for the task executor."""
        # fmt: off
        cmd = [
            self.task.model_id,
            "--tensor-parallel-size", str(len(gpus)),
            "--no-enable-prefix-caching",
            # Jeff: this is required for now bc vLLM V1 cannot cache prompt embeddings,
            # resulting the direct reuse of matched prefix, leaving no prompt embeddings
            # to stream to the talker
            "--enforce-eager",
            "--port", str(port),
            "--cornserve-sidecar-ranks", *[str(gpu.global_rank) for gpu in gpus],
        ]
        # fmt: on
        return cmd

    def get_api_url(self, base: str) -> str:
        """Get the task executor's base URL for API calls."""
        return f"{base}/v1/chat/completions"

    def to_request(self, task_input: OmniThinkerInput, task_output: OmniThinkerOutput) -> dict[str, Any]:
        """Convert TaskInput to a request object for the task executor."""
        # XXX: `DataForward[str]` not supported yet.
        # Compatibility with OpenAI Chat Completion API is kept.
        content: list[dict[str, Any]] = [dict(type="text", text=task_input.prompt)]

        for (modality, data_url), forward in zip(task_input.multimodal_data, task_input.embeddings, strict=True):
            data_uri = f"data:{modality}/uuid;data_id={forward.id};url={data_url},"
            content.append({"type": f"{modality}_url", f"{modality}_url": {"url": data_uri}})

        request: dict[str, Any] = dict(
            model=self.task.model_id,
            messages=[dict(role="user", content=content)],
            max_completion_tokens=512,
        )
        if task_input.return_audio:
            assert task_output.embeddings is not None, "Audio response requested but no embeddings provided"
            request["request_id"] = task_output.embeddings.id
            request["talker_sidecar_ranks"] = task_output.embeddings.dst_sidecar_ranks
        return request

    def from_response(
        self,
        task_output: OmniThinkerOutput,
        response: dict[str, Any],
    ) -> OmniThinkerOutput:
        """Convert the task executor response to TaskOutput."""
        return OmniThinkerOutput(
            response=response["choices"][0]["message"]["content"], embeddings=task_output.embeddings
        )


DESCRIPTOR_REGISTRY.register(OmniThinkerLLMTask, OmniThinkerDescriptor, default=True)


class OmniTalkerDescriptor(TaskExecutionDescriptor[OmniTalkerLLMTask, OmniTalkerInput, OmniTalkerOutput]):
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
        cmd = [
            self.task.model_id,
            "--port", str(port),
            "--enforce-eager",
            "--cornserve-sidecar-ranks", *[str(gpu.global_rank) for gpu in gpus],
        ]
        # fmt: on
        return cmd

    def get_api_url(self, base: str) -> str:
        """Get the task executor's base URL for API calls."""
        return f"{base}/v1/chat/completions"

    def to_request(self, task_input: OmniTalkerInput, task_output: OmniTalkerOutput) -> dict[str, Any]:
        """Convert TaskInput to a request object for the task executor."""
        # XXX: `DataForward[str]` not supported yet.
        # Compatibility with OpenAI Chat Completion API is kept.
        content: list[dict[str, Any]] = [dict(type="text", text=task_input.prompt)]

        for modality, data_url in task_input.multimodal_data:
            content.append(
                {
                    "type": f"{modality}_url",
                    f"{modality}_url": data_url,
                }
            )

        request = dict(
            model=self.task.model_id,
            messages=[dict(role="user", content=content)],
            request_id=task_input.embeddings.id,
        )
        return request

    def from_response(self, task_output: OmniTalkerOutput, response: dict[str, Any]) -> OmniTalkerOutput:
        """Convert the task executor response to TaskOutput."""
        return OmniTalkerOutput(response=response["content"])


DESCRIPTOR_REGISTRY.register(OmniTalkerLLMTask, OmniTalkerDescriptor, default=True)
