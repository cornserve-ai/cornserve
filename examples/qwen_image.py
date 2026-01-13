"""An app that generates images using Qwen-Image model.

```console
$ cornserve register examples/qwen_image.py

$ cornserve invoke qwen_image --data - <<EOF
prompt: "A beautiful landscape with mountains and a lake at sunset"
height: 512
width: 512
num_inference_steps: 20
EOF

$ cornserve invoke qwen_image --data - <<EOF
prompt: "A cute robot playing with a cat in a futuristic city"
height: 1024
width: 1024
num_inference_steps: 50
EOF
```
"""

from __future__ import annotations

from cornserve_tasklib.task.unit.generator import (
    ImageGeneratorInput,
    ImageGeneratorTask,
)
from cornserve_tasklib.task.unit.llm import (
    LLMEmbeddingUnitTask,
    OpenAIChatCompletionRequest,
)

from cornserve.app.base import AppConfig
from cornserve.task.base import Task, TaskInput, TaskOutput


class QwenImageInput(TaskInput):
    """Input model for Qwen-Image generation task."""

    prompt: str
    height: int
    width: int
    num_inference_steps: int


class QwenImageOutput(TaskOutput):
    """Output model for Qwen-Image generation task.

    Attributes:
        image: Generated image as a base64-encoded PNG bytes.
    """

    image: str


class QwenImageTask(Task[QwenImageInput, QwenImageOutput]):
    """A task that invokes the Qwen-Image model for image generation."""

    def post_init(self) -> None:
        """Initialize subtasks."""
        self.text_encoder = LLMEmbeddingUnitTask(model_id="Qwen/Qwen2.5-VL-7B-Instruct", receive_embeddings=False)
        self.generator = ImageGeneratorTask(model_id="Qwen/Qwen-Image")

        # These two parameters are specific to Qwen/Qwen-Image and should not be changed.
        self.system_prompt = (
            "Describe the image by detailing the color, shape, size, texture, quantity, "
            "text, spatial relationships of the objects and background:"
        )
        self.num_prefix_tokens_to_slice = 34

    def invoke(self, task_input: QwenImageInput) -> QwenImageOutput:
        """Invoke the task."""
        encoder_input = OpenAIChatCompletionRequest.model_validate(
            dict(
                model="Qwen/Qwen2.5-VL-7B-Instruct",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": task_input.prompt},
                ],
                max_completion_tokens=1,
            )
        )
        encoder_output = self.text_encoder.invoke(encoder_input)
        generator_input = ImageGeneratorInput(
            height=task_input.height,
            width=task_input.width,
            num_inference_steps=task_input.num_inference_steps,
            skip_tokens=self.num_prefix_tokens_to_slice,
            embeddings=encoder_output.embeddings,
        )
        generator_output = self.generator.invoke(generator_input)
        return QwenImageOutput(image=generator_output.generated)


qwen_image = QwenImageTask()


class Config(AppConfig):
    """App configuration model."""

    tasks = {"qwen_image": qwen_image}


async def serve(request: QwenImageInput) -> QwenImageOutput:
    """Main serve function for the app."""
    return await qwen_image(request)
