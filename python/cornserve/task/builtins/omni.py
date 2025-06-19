"""Built-in task for Qwen Omni Thinker and Talker."""

from __future__ import annotations

from cornserve.task.base import Task, TaskInput, TaskOutput, UnitTask
from cornserve.task.builtins.encoder import EncoderInput, EncoderTask, Modality
from cornserve.task.forward import DataForward, Tensor


class OmniThinkerInput(TaskInput):
    """Input model for Qwen Omni Thinker.

    Attributes:
        prompt: The prompt to send to the thinker.
        multimodal_data: List of tuples (modality, data URL).
            "image", "audio", "video", etc. for modality.
        embeddings: Multimodal embeddings to send to the LLM.
        return_audio: Whether to return audio response.
    """

    prompt: str
    multimodal_data: list[tuple[str, str]] = []
    embeddings: list[DataForward[Tensor]] = []
    return_audio: bool = False


class OmniThinkerOutput(TaskOutput):
    """Output model of Qwen Omni Thinker.

    Attributes:
        response: The text ouptut of the thinker.
        embeddings: The embeddings from the thinker.
    """

    response: str
    embeddings: DataForward[Tensor] | None


class OmniThinkerLLMTask(UnitTask[OmniThinkerInput, OmniThinkerOutput]):
    """A task that represents the Qwen Omni Thinker.

    Attributes:
        model_id: The ID of the model to use for the task.
    """

    model_id: str = "Qwen/Qwen2.5-Omni-7B"

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        return f"llm-{self.model_id.split('/')[-1].lower().replace('.', '-')}-thinker"

    def make_record_output(self, task_input: OmniThinkerInput) -> OmniThinkerOutput:
        """Create a task output for task invocation recording."""
        return OmniThinkerOutput(response="", embeddings=DataForward[Tensor]() if task_input.return_audio else None)


class OmniTalkerInput(TaskInput):
    """Input model for Qwen Omni Talker.

    Attributes:
        prompt: The prompt to send to the LLM.
        multimodal_data: List of tuples (modality, data URL).
            "image", "audio", "video", etc. for modality.
        embeddings: Thinker embeddings to send to the Talker.
    """

    prompt: str
    multimodal_data: list[tuple[str, str]] = []
    embeddings: DataForward[Tensor]


class OmniTalkerOutput(TaskOutput):
    """Output model of Qwen Omni Talker.

    Attributes:
        response: The audio response from the Talker.
    """

    response: bytes


class OmniTalkerLLMTask(UnitTask[OmniTalkerInput, OmniTalkerOutput]):
    """A task that represents the Qwen Omni Talker.

    Attributes:
        model_id: The ID of the model to use for the task.
    """

    model_id: str = "Qwen/Qwen2.5-Omni-7B"

    def make_name(self) -> str:
        """Create a concise string representation of the task."""
        return f"llm-{self.model_id.split('/')[-1].lower().replace('.', '-')}-talker"

    def make_record_output(self, task_input: OmniTalkerInput) -> OmniTalkerOutput:
        """Create a task output for task invocation recording."""
        return OmniTalkerOutput(response=b"")


class OmniInput(TaskInput):
    """Input model for Qwen Omni tasks.

    Attributes:
        prompt: The prompt to send to the Omni model.
        multimodal_data: List of tuples (modality, data URL).
            "image", "audio", "video", etc. for modality.
        embeddings: Multimodal embeddings to send to the LLM.
        return_audio: Whether to return audio response.
    """

    prompt: str
    multimodal_data: list[tuple[str, str]] = []
    embeddings: list[DataForward[Tensor]] = []
    return_audio: bool  # same as huggingface parameter
    # use_audio_in_video: not supported yet


class OmniOutput(TaskOutput):
    """Output model of Qwen Omni tasks.

    Attributes:
        response: The thinker text response or talker audio response.
    """

    response: str | bytes


class OmniTask(Task[OmniInput, OmniOutput]):
    """A task that invokes a Multimodal LLM.

    Attributes:
        model_id: The ID of the model to use for the task.
        modalities: List of input modalities other than text.
    """

    model_id: str
    modalities: list[Modality] = []
    enable_talker: bool = True

    def post_init(self) -> None:
        """Initialize subtasks."""
        if self.model_id != "Qwen/Qwen2.5-Omni-7B":
            raise ValueError("OmniTask currently only supports native Qwen2.5 Omni model.")
        if Modality.IMAGE in self.modalities:
            self.image_encoder = EncoderTask(model_id=self.model_id, modality=Modality.IMAGE)
        if Modality.VIDEO in self.modalities:
            self.video_encoder = EncoderTask(model_id=self.model_id, modality=Modality.VIDEO)
        if Modality.AUDIO in self.modalities:
            self.audio_encoder = EncoderTask(model_id=self.model_id, modality=Modality.AUDIO)
        self.thinker = OmniThinkerLLMTask(model_id=self.model_id)
        if self.enable_talker:
            self.talker = OmniTalkerLLMTask(model_id=self.model_id)

    def invoke(self, task_input: OmniInput) -> OmniOutput:
        """Invoke the task.

        Given multimodal data and a text prompt, run the corresponding encoder
        for multimodal data and then pass the embeddings and text prompt to the LLM.
        """
        image_data = []
        video_data = []
        audio_data = []
        for modality, data in task_input.multimodal_data:
            if modality == Modality.IMAGE:
                image_data.append(data)
            elif modality == Modality.VIDEO:
                video_data.append(data)
            elif modality == Modality.AUDIO:
                audio_data.append(data)
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

        if audio_data:
            if not hasattr(self, "audio_encoder"):
                raise ValueError("Audio modality is not supported.")
            audio_task_input = EncoderInput(data_urls=audio_data)
            audio_embeddings = self.audio_encoder.invoke(audio_task_input).embeddings
        else:
            audio_embeddings = []

        # Retain the order of multimodal data
        embeddings: list[DataForward[Tensor]] = []
        for modality, _ in task_input.multimodal_data:
            if modality == Modality.IMAGE:
                embeddings.append(image_embeddings.pop(0))
            elif modality == Modality.VIDEO:
                embeddings.append(video_embeddings.pop(0))
            elif modality == Modality.AUDIO:
                embeddings.append(audio_embeddings.pop(0))

        if task_input.return_audio and not self.enable_talker:
            raise ValueError("Audio response is requested but talker is not enabled.")

        thinker_input = OmniThinkerInput(
            prompt=task_input.prompt,
            multimodal_data=task_input.multimodal_data,
            embeddings=embeddings,
            return_audio=task_input.return_audio,
        )

        thinker_output = self.thinker.invoke(thinker_input)

        if self.enable_talker and task_input.return_audio:
            assert thinker_output.embeddings is not None, "Thinker output must have embeddings for talker."
            talker_input = OmniTalkerInput(
                prompt=task_input.prompt,
                multimodal_data=task_input.multimodal_data,
                embeddings=thinker_output.embeddings,
            )
            talker_output = self.talker.invoke(talker_input)
            return OmniOutput(response=talker_output.response)

        return OmniOutput(response=thinker_output.response)
