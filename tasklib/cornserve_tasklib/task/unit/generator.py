"""Built-in task for multimodal content generation."""

from __future__ import annotations

import enum

from cornserve.task.base import (
    Stream,
    TaskInput,
    TaskOutput,
    TaskProfileConfig,
    UnitTask,
)
from cornserve.task.forward import DataForward, Tensor

from cornserve_tasklib.task.unit.llm import OpenAIChatCompletionChunk


class Modality(enum.StrEnum):
    """Supported modalities for generator tasks."""

    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


class ImageGeriProfileConfig(TaskProfileConfig):
    """Image generator-specific profiling configuration fields."""

    sp_size: int
    max_batch_size: int = 1

    def to_profile_str(self) -> str:
        """Return a string representation of the profile configuration."""
        return f"maxbs{self.max_batch_size}+sp{self.sp_size}"


class AudioGeriProfileConfig(TaskProfileConfig):
    """Audio generator-specific profiling configuration fields."""

    max_batch_size: int = 1

    def to_profile_str(self) -> str:
        """Return a string representation of the profile configuration."""
        return f"maxbs{self.max_batch_size}"


class ImageGeneratorInput(TaskInput):
    """Input model for image generator tasks.

    Attributes:
        height: Height of the generated content in pixels.
        width: Width of the generated content in pixels.
        num_inference_steps: Number of denoising steps to perform.
        embeddings: Text (e.g., prompt) embeddings from the encoder.
        skip_tokens: Number of initial tokens to skip from the embeddings.
        request_id: Optional stable ID for deterministic routing in RouterApp.
    """

    height: int
    width: int
    num_inference_steps: int
    embeddings: DataForward[Tensor]
    skip_tokens: int = 0
    request_id: str = ""


class ImageGeneratorOutput(TaskOutput):
    """Output model for image generator tasks.

    Attributes:
        generated: Generated content as bytes, encoded in base64 (PNG).
    """

    generated: str


class ImageGeneratorTask(UnitTask[ImageGeneratorInput, ImageGeneratorOutput]):
    """A task that invokes an image content generator.

    Attributes:
        model_id: The ID of the model to use for the task.
        max_batch_size: Maximum batch size to use for the serving system.
        sp_size: Sequence parallel degree (number of GPUs for SP).
    """

    model_id: str
    max_batch_size: int = 1
    modality: Modality = Modality.IMAGE
    sp_size: int = 1

    def make_record_output(
        self, task_input: ImageGeneratorInput
    ) -> ImageGeneratorOutput:
        """Create a task output for task invocation recording."""
        return ImageGeneratorOutput(generated="")

    def validate_input(self, task_input: ImageGeneratorInput) -> None:
        """Validate the input for the generator task."""
        if task_input.height <= 0 or task_input.width <= 0:
            raise ValueError("Height and width must be positive integers.")

        if task_input.num_inference_steps <= 0:
            raise ValueError("Number of inference steps must be positive.")

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        model_name = self.model_id.split("/")[-1].lower()
        profile = f"maxbs{self.max_batch_size}+sp{self.sp_size}"
        return f"generator-{self.modality.lower()}-{model_name}-{profile}"


class QwenImageTextGeneratorInput(TaskInput):
    """Input model for direct-text Qwen-Image generator tasks.

    In this mode, Geri runs Qwen2.5-VL text encoding internally and feeds the
    resulting prompt embeddings directly into Qwen-Image DiT.

    Attributes:
        prompt: Text prompt to render.
        height: Height of the generated content in pixels.
        width: Width of the generated content in pixels.
        num_inference_steps: Number of denoising steps to perform.
    """

    prompt: str
    height: int
    width: int
    num_inference_steps: int


class QwenImageTextGeneratorTask(
    UnitTask[QwenImageTextGeneratorInput, ImageGeneratorOutput]
):
    """A task that invokes Qwen-Image in direct-text mode inside Geri.

    Attributes:
        model_id: The ID of the Qwen-Image model to use for generation.
        max_batch_size: Maximum batch size to use for the serving system.
        sp_size: Sequence parallel degree (number of GPUs for SP).
    """

    model_id: str = "Qwen/Qwen-Image"
    max_batch_size: int = 1
    modality: Modality = Modality.IMAGE
    sp_size: int = 1

    def make_record_output(
        self, task_input: QwenImageTextGeneratorInput
    ) -> ImageGeneratorOutput:
        """Create a task output for task invocation recording."""
        return ImageGeneratorOutput(generated="")

    def validate_input(self, task_input: QwenImageTextGeneratorInput) -> None:
        """Validate the input for direct-text Qwen-Image generation."""
        if not task_input.prompt.strip():
            raise ValueError("Prompt must be non-empty.")

        if task_input.height <= 0 or task_input.width <= 0:
            raise ValueError("Height and width must be positive integers.")

        if task_input.num_inference_steps <= 0:
            raise ValueError("Number of inference steps must be positive.")

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        model_name = self.model_id.split("/")[-1].lower()
        profile = f"maxbs{self.max_batch_size}+sp{self.sp_size}"
        return f"generator-{self.modality.lower()}-text-{model_name}-{profile}"


class AudioGeneratorInput(TaskInput):
    """Input model for audio generator tasks."""

    embeddings: DataForward[Tensor]
    chunk_size: int | None = None
    left_context_size: int | None = None


class AudioGeneratorTask(
    UnitTask[AudioGeneratorInput, Stream[OpenAIChatCompletionChunk]]
):
    """A task that invokes an audio content generator.
    Attributes:
        model_id: The ID of the model to use for the task.
        max_batch_size: Maximum batch size to use for the serving system.
    """

    model_id: str
    max_batch_size: int = 1
    modality: Modality = Modality.AUDIO

    def make_record_output(
        self, task_input: AudioGeneratorInput
    ) -> Stream[OpenAIChatCompletionChunk]:
        """Create a task output for task invocation recording."""
        return Stream[OpenAIChatCompletionChunk]()

    def validate_input(self, task_input: AudioGeneratorInput) -> None:
        """Validate the input for the generator task."""
        pass

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        model_name = self.model_id.split("/")[-1].lower()
        return f"generator-{self.modality.lower()}-{model_name}"


class DummyImageGeneratorInput(TaskInput):
    """Input for dummy image generator tasks.

    Attributes:
        dummy_seq_len: Number of embedding tokens to simulate. Dummy Generator tasks do not
            send and receive tensors via sidecar, and instead simulate their own data by
            instantiating random tensors. However, for Geri to know the sequence length of the
            data it should simulate receiving, this field is necessary.
        height: Height of generated image in pixels.
        width: Width of generated image in pixels.
        num_inference_steps: Number of denoising steps.
        skip_tokens: Initial tokens to skip from embeddings.
    """

    dummy_seq_len: int
    height: int
    width: int
    num_inference_steps: int
    skip_tokens: int = 0


class DummyAudioGeneratorInput(TaskInput):
    """Input for dummy audio generator tasks.

    Attributes:
        dummy_seq_len: Number of embedding tokens to simulate. Dummy Generator tasks do not
            send and receive tensors via sidecar, and instead simulate their own data by
            instantiating random tensors. However, for Geri to know the sequence length of the
            data it should simulate receiving, this field is necessary.
        chunk_size: Audio chunk size.
        left_context_size: Audio left context size.
    """

    dummy_seq_len: int
    chunk_size: int | None = None
    left_context_size: int | None = None


class DummyGeneratorOutput(TaskOutput):
    """Empty output for dummy generator tasks."""

    pass


class DummyImageGeneratorTask(
    UnitTask[DummyImageGeneratorInput, DummyGeneratorOutput],
    ImageGeriProfileConfig,
):
    """A dummy task that invokes an image generator.

    Similar to a real generator task, this will invoke a generator task executor
    to perform work. However, real data will not be forwarded to the task executor
    via sidecar; dummy data will be randomly initialized by the generator. Also,
    the task will not return its results either, dropping them instead.

    Attributes:
        model_id: The Hugging Face model ID.
        max_batch_size: Maximum batch size for the serving system.
        sp_size: Sequence parallel degree (number of GPUs for SP).
    """

    model_id: str
    modality: Modality = Modality.IMAGE

    def make_record_output(
        self, task_input: DummyImageGeneratorInput
    ) -> DummyGeneratorOutput:
        """Create a task output for task invocation recording."""
        return DummyGeneratorOutput()

    def validate_input(self, task_input: DummyImageGeneratorInput) -> None:
        """Validate the input for the dummy image generator task."""
        if task_input.dummy_seq_len <= 0:
            raise ValueError("Embedding sequence length must be positive.")
        if task_input.height <= 0 or task_input.width <= 0:
            raise ValueError("height and width must be positive integers.")
        if task_input.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be a positive integer.")

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        model_name = self.model_id.split("/")[-1].lower()
        return f"dummy-generator-image-{model_name}-{self.to_profile_str()}"


class DummyAudioGeneratorTask(
    UnitTask[DummyAudioGeneratorInput, Stream[OpenAIChatCompletionChunk]],
    AudioGeriProfileConfig,
):
    """A dummy task that invokes an audio generator.

    Similar to a real generator task, this will invoke a generator task executor
    to perform work. However, real data will not be forwarded to the task executor
    via sidecar; dummy data will be randomly initialized by the generator. Also,
    the task will not return its results either, dropping them instead.

    Attributes:
        model_id: The Hugging Face model ID.
        max_batch_size: Maximum batch size for the serving system.
    """

    model_id: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    modality: Modality = Modality.AUDIO

    def make_record_output(
        self, task_input: DummyAudioGeneratorInput
    ) -> Stream[OpenAIChatCompletionChunk]:
        """Create a task output for task invocation recording."""
        return Stream[OpenAIChatCompletionChunk]()

    def validate_input(self, task_input: DummyAudioGeneratorInput) -> None:
        """Validate the input for the dummy audio generator task."""
        if task_input.dummy_seq_len <= 0:
            raise ValueError("Embedding sequence length must be positive.")

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        model_name = self.model_id.split("/")[-1].lower()
        return f"dummy-generator-audio-{model_name}-{self.to_profile_str()}"
