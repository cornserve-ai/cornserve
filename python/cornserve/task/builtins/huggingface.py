"""Built-in HuggingFace tasks."""

from __future__ import annotations

import uuid
from typing import Any

from pydantic import Field

from cornserve.task.base import Stream, TaskInput, TaskOutput, UnitTask
from cornserve.task.builtins.llm import (
    ChatCompletionMessageParam,
    OpenAIChatCompletionChunk,
    StreamOptions,
)


class HuggingFaceQwenImageInput(TaskInput):
    """Input model for Qwen-Image generation task.

    Matches QwenImageInput from examples/qwen_image.py.

    Attributes:
        prompt: Text prompt for image generation.
        height: Height of generated image in pixels.
        width: Width of generated image in pixels.
        num_inference_steps: Number of denoising steps to perform.
    """

    prompt: str
    height: int
    width: int
    num_inference_steps: int


class HuggingFaceQwenImageOutput(TaskOutput):
    """Output model for Qwen-Image generation task.

    Matches QwenImageOutput from examples/qwen_image.py.

    Attributes:
        image: Generated image as a base64-encoded PNG bytes.
    """

    image: str


class HuggingFaceQwenImageTask(UnitTask[HuggingFaceQwenImageInput, HuggingFaceQwenImageOutput]):
    """A task that invokes the Qwen-Image model via HuggingFace diffusers.

    Attributes:
        model_id: The ID of the Qwen-Image model to use.
        max_batch_size: Maximum batch size for inference.
    """

    model_id: str = "Qwen/Qwen-Image"
    max_batch_size: int = 1

    def make_record_output(self, task_input: HuggingFaceQwenImageInput) -> HuggingFaceQwenImageOutput:
        """Create a mock task output object for invocation recording."""
        return HuggingFaceQwenImageOutput(image="")

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        return f"hf-qwen-image-{self.model_id.split('/')[-1].lower()}"


class HuggingFaceQwenOmniInput(TaskInput):
    """Input model for Qwen 2.5 Omni tasks.

    Matches OmniInput from omni.py (inherits OpenAIChatCompletionRequest pattern).

    Attributes:
        model: The model ID to use.
        messages: List of chat messages.
        return_audio: Whether to return audio response.
        frequency_penalty: Frequency penalty for generation.
        max_completion_tokens: Maximum completion tokens.
        presence_penalty: Presence penalty for generation.
        seed: Random seed for generation.
        stream_options: Streaming options.
        temperature: Temperature for generation.
        top_p: Top-p for generation.
        request_id: Unique request identifier.
        ignore_eos: Whether to ignore end-of-sequence tokens.
    """

    model: str = "Qwen/Qwen2.5-Omni-7B"
    messages: list[ChatCompletionMessageParam]
    return_audio: bool = True
    frequency_penalty: float | None = 0.0
    max_completion_tokens: int | None = None
    presence_penalty: float | None = 0.0
    seed: int | None = None
    stream_options: StreamOptions | None = None
    temperature: float | None = None
    top_p: float | None = None
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    ignore_eos: bool = False

    def model_post_init(self, context: Any, /) -> None:
        """Validate the model."""
        if self.model != "Qwen/Qwen2.5-Omni-7B":
            raise ValueError("Only Qwen/Qwen2.5-Omni-7B is supported.")


class HuggingFaceQwenOmniOutput(TaskOutput):
    """Output chunk for Qwen 2.5 Omni tasks.

    Matches OmniOutputChunk from omni.py.

    Either should be present but not both.

    Attributes:
        audio_chunk: Base64-encoded audio chunk of np.float32 raw waveform.
        text_chunk: Text chunk from the LLM.
    """

    audio_chunk: str | None = None
    text_chunk: OpenAIChatCompletionChunk | None = None


class HuggingFaceQwenOmniTask(UnitTask[HuggingFaceQwenOmniInput, Stream[HuggingFaceQwenOmniOutput]]):
    """A task that invokes the Qwen 2.5 Omni model via HuggingFace transformers.

    Attributes:
        model_id: The ID of the Qwen 2.5 Omni model to use.
        max_batch_size: Maximum batch size for inference.
    """

    model_id: str = "Qwen/Qwen2.5-Omni-7B"
    max_batch_size: int = 1

    def make_record_output(self, task_input: HuggingFaceQwenOmniInput) -> Stream[HuggingFaceQwenOmniOutput]:
        """Create a mock task output object for invocation recording."""
        return Stream[HuggingFaceQwenOmniOutput]()

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        return f"hf-qwen-omni-{self.model_id.split('/')[-1].lower().replace('.', '-')}"
