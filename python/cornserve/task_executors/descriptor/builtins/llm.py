"""Built-in task execution descriptor for LLM tasks."""

from __future__ import annotations

from typing import Any

from cornserve import constants
from cornserve.services.resource_manager.resource import GPU
from cornserve.task.builtins.llm import LLMInput, LLMOutput, LLMTask
from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor
from cornserve.task_executors.descriptor.registry import DESCRIPTOR_REGISTRY


class VLLMDescriptor(TaskExecutionDescriptor[LLMTask, LLMInput, LLMOutput]):
    """Task execution descriptor for Encoder tasks.

    This descriptor handles launching Eric (multimodal encoder) tasks and converting between
    the external task API types and internal executor types.
    """

    def create_executor_name(self) -> str:
        """Create a name for the task executor."""
        return "-".join(["llm", self.task.model_id.split("/")[-1]]).lower()

    def get_container_image(self) -> str:
        """Get the container image name for the task executor."""
        return constants.CONTAINER_IMAGE_VLLM

    def get_container_args(self, gpus: list[GPU], port: int) -> list[str]:
        """Get the container command for the task executor."""
        # fmt: off
        cmd = [
            self.task.model_id,
            "--tensor-parallel-size", str(len(gpus)),
            "--port", str(port),
            "--limit-mm-per-prompt", "image=5",  # TODO: Is this still needed?
            "--cornserve-sidecar-ranks", *[str(gpu.global_rank) for gpu in gpus],
        ]
        # fmt: on
        return cmd

    def get_api_url(self, base: str) -> str:
        """Get the task executor's base URL for API calls."""
        return f"{base}/v1/chat/completions"

    def to_request(self, task_input: LLMInput, task_output: LLMOutput) -> dict[str, Any]:
        """Convert TaskInput to a request object for the task executor."""
        # XXX: `DataForward[str]` not supported yet.
        # Compatibility with OpenAI Chat Completion API is kept.
        content: list[dict[str, Any]] = [dict(type="text", text=task_input.prompt)]
        for (modality, data_url), forward in zip(task_input.multimodal_data, task_input.embeddings, strict=True):
            data_uri = f"data:{modality}/uuid;data_id={forward.id};url={data_url},"
            content.append({"type": modality, f"{modality}_url": {"url": data_uri}})

        request = dict(
            model=self.task.model_id,
            messages=[dict(role="user", content=content)],
            max_completion_tokens=512,
        )

        return request

    def from_response(self, task_output: LLMOutput, response: dict[str, Any]) -> LLMOutput:
        """Convert the task executor response to TaskOutput."""
        # XXX: `DataForward[str]` not supported yet.
        return LLMOutput(response=response["choices"][0]["message"]["content"])


DESCRIPTOR_REGISTRY.register(LLMTask, VLLMDescriptor, default=True)
