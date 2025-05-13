"""Built-in task for Multimodal LLMs."""

from __future__ import annotations

from cornserve.task.base import Task, TaskInput, TaskOutput
from cornserve.task.builtins.encoder import EncoderInput, EncoderTask, Modality
from cornserve.task.builtins.llm import LLMInput, LLMTask
from cornserve.task.forward import DataForward, Tensor


class MLLMInput(TaskInput):
    """Input model for Multimodal LLM tasks.

    Attributes:
        prompt: The prompt to send to the LLM.
        multimodal_data: List of tuples (modality, data URL).
    """

    prompt: str
    multimodal_data: list[tuple[str, str]] = []


class MLLMOutput(TaskOutput):
    """Output model for Multimodal LLM tasks.

    Attributes:
        response: The response from the LLM.
    """

    response: str


class MLLMTask(Task[MLLMInput, MLLMOutput]):
    """A task that invokes a Multimodal LLM.

    Attributes:
        model_id: The ID of the model to use for the task.
        modalities: List of input modalities other than text.
    """

    model_id: str
    modalities: list[Modality] = []

    def post_init(self) -> None:
        """Initialize subtasks."""
        if Modality.IMAGE in self.modalities:
            self.image_encoder = EncoderTask(model_id=self.model_id, modality=Modality.IMAGE)
        if Modality.VIDEO in self.modalities:
            self.video_encoder = EncoderTask(model_id=self.model_id, modality=Modality.VIDEO)
        self.llm = LLMTask(model_id=self.model_id)

    def invoke(self, task_input: MLLMInput) -> MLLMOutput:
        """Invoke the task.

        Given multimodal data and a text prompt, run the corresponding encoder
        for multimodal data and then pass the embeddings and text prompt to the LLM.
        """
        image_data = []
        video_data = []
        for modality, data in task_input.multimodal_data:
            if modality == Modality.IMAGE:
                image_data.append(data)
            elif modality == Modality.VIDEO:
                video_data.append(data)
            else:
                raise ValueError(f"Unsupported modality: {modality}")

        if image_data:
            if not hasattr(self, "image_encoder"):
                raise ValueError("Image modality is not supported.")
            image_task_input = EncoderInput(data_urls=image_data)
            image_embeddings = self.image_encoder.invoke(image_task_input).embeddings
        else:
            image_embeddings = []

        if video_data:
            if not hasattr(self, "video_encoder"):
                raise ValueError("Video modality is not supported.")
            video_task_input = EncoderInput(data_urls=video_data)
            video_embeddings = self.video_encoder.invoke(video_task_input).embeddings
        else:
            video_embeddings = []

        # Retain the order of multimodal data
        embeddings: list[DataForward[Tensor]] = []
        for modality, _ in task_input.multimodal_data:
            if modality == Modality.IMAGE:
                embeddings.append(image_embeddings.pop(0))
            elif modality == Modality.VIDEO:
                embeddings.append(video_embeddings.pop(0))

        llm_input = LLMInput(
            prompt=task_input.prompt,
            multimodal_data=task_input.multimodal_data,
            embeddings=embeddings,
        )
        llm_output = self.llm.invoke(llm_input)

        return MLLMOutput(response=llm_output.response)
