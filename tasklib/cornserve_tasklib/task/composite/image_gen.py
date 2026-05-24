"""Composite tasks for Qwen-Image generation with weighted generator routing."""

from __future__ import annotations

from typing import cast

from cornserve.task.base import Task, TaskInput, TaskOutput

from cornserve_tasklib.task.composite.router import RouterApp
from cornserve_tasklib.task.unit.generator import (
    ImageGeneratorInput,
    ImageGeneratorOutput,
    ImageGeneratorTask,
    QwenImageTextGeneratorInput,
    QwenImageTextGeneratorTask,
)
from cornserve_tasklib.task.unit.llm import (
    LLMEmbeddingUnitTask,
    OpenAIChatCompletionRequest,
)


class QwenImageInput(TaskInput):
    """Input model for Qwen-Image generation tasks."""

    prompt: str
    height: int
    width: int
    num_inference_steps: int


class QwenImageOutput(TaskOutput):
    """Output model for Qwen-Image generation tasks."""

    image: str


class MixedQwenImageTask(Task[QwenImageInput, QwenImageOutput]):
    """Disaggregated Qwen-Image with shared LLM encoder and weighted generator routing.

    The LLM encoder (Qwen2.5-VL) computes text embeddings, which are then
    routed to one of several ``ImageGeneratorTask`` instances (potentially with
    different ``sp_size``) via ``RouterApp`` weighted routing.

    This mirrors ``MixedMLLMTask`` for MLLMs: shared encoder pool + weighted
    routing across heterogeneous backend replicas.

    Attributes:
        model_id: Image generator model (e.g. ``"Qwen/Qwen-Image"``).
        encoder_model_id: LLM encoder model (e.g. ``"Qwen/Qwen2.5-VL-7B-Instruct"``).
        generator_routing_tasks: Pre-constructed ``ImageGeneratorTask`` instances
            with potentially different ``sp_size``.
        routing_weights: Per-generator routing weights (capacity-proportional).
        encoder_tp_size: Tensor parallel degree for the LLM encoder.
        encoder_max_num_seqs: Max batch size for the LLM encoder.
        encoder_gpu_memory_utilization: GPU memory utilization for the encoder.
        generator_max_batch_size: Max batch size for generator tasks.
        num_prefix_tokens_to_slice: Tokens to skip from encoder embeddings.
    """

    model_id: str = "Qwen/Qwen-Image"
    encoder_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"

    generator_routing_tasks: list[ImageGeneratorTask]
    routing_weights: list[float]

    encoder_tp_size: int = 1
    encoder_max_num_seqs: int = 256
    encoder_gpu_memory_utilization: float = 0.9
    generator_max_batch_size: int = 1
    num_prefix_tokens_to_slice: int = 34

    def post_init(self) -> None:
        """Initialize the LLM encoder and generator router."""
        self.text_encoder = LLMEmbeddingUnitTask(
            model_id=self.encoder_model_id,
            receive_embeddings=False,
            tp_size=self.encoder_tp_size,
            max_num_seqs=self.encoder_max_num_seqs,
            gpu_memory_utilization=self.encoder_gpu_memory_utilization,
        )
        self.generator_router = RouterApp[ImageGeneratorInput, ImageGeneratorOutput](
            routing_tasks=cast(list[Task], self.generator_routing_tasks),
            routing_weights=self.routing_weights,
        )

        self.system_prompt = (
            "Describe the image by detailing the color, shape, size, texture, "
            "quantity, text, spatial relationships of the objects and background:"
        )

    def invoke(self, task_input: QwenImageInput) -> QwenImageOutput:
        """Encode text prompt, then route to a weighted generator."""
        encoder_input = OpenAIChatCompletionRequest.model_validate(
            {
                "model": self.encoder_model_id,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": task_input.prompt},
                ],
                "max_completion_tokens": 1,
            }
        )
        encoder_output = self.text_encoder.invoke(encoder_input)

        generator_input = ImageGeneratorInput(
            height=task_input.height,
            width=task_input.width,
            num_inference_steps=task_input.num_inference_steps,
            skip_tokens=self.num_prefix_tokens_to_slice,
            embeddings=encoder_output.embeddings,
            request_id=task_input.model_dump_json(),
        )
        generator_output = self.generator_router.invoke(generator_input)
        return QwenImageOutput(image=generator_output.generated)


class MixedMonoQwenImageTask(Task[QwenImageInput, QwenImageOutput]):
    """Monolithic Qwen-Image with weighted routing across different SP sizes.

    Each ``QwenImageTextGeneratorTask`` handles text encoding + image
    generation internally.  The ``RouterApp`` distributes requests across
    instances with different ``sp_size`` proportionally to their throughput.

    Attributes:
        mono_routing_tasks: Pre-constructed ``QwenImageTextGeneratorTask``
            instances with potentially different ``sp_size``.
        routing_weights: Per-task routing weights (capacity-proportional).
    """

    mono_routing_tasks: list[QwenImageTextGeneratorTask]
    routing_weights: list[float]

    def post_init(self) -> None:
        """Initialize the mono generator router."""
        self.router = RouterApp[QwenImageTextGeneratorInput, ImageGeneratorOutput](
            routing_tasks=cast(list[Task], self.mono_routing_tasks),
            routing_weights=self.routing_weights,
        )

    def invoke(self, task_input: QwenImageInput) -> QwenImageOutput:
        """Route to a weighted mono generator."""
        gen_input = QwenImageTextGeneratorInput(
            prompt=task_input.prompt,
            height=task_input.height,
            width=task_input.width,
            num_inference_steps=task_input.num_inference_steps,
        )
        gen_output = self.router.invoke(gen_input)
        return QwenImageOutput(image=gen_output.generated)
