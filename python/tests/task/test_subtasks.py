from __future__ import annotations

from cornserve.task.base import Task, TaskInput, TaskOutput
from cornserve.task.builtins.encoder import EncoderInput, EncoderTask, Modality
from cornserve.task.builtins.llm import LLMInput, LLMTask
from cornserve.task.builtins.mllm import MLLMTask


class ArenaInput(TaskInput):
    """App request model for the Arena task.

    Attributes:
        prompt: The prompt to send to the LLM.
        multimodal_data: List of tuples (modality, data URL).
        max_completion_tokens: Max number of tokens to generate in the response.
        seed: Optional random seed.
    """

    prompt: str
    multimodal_data: list[tuple[str, str]] = []
    max_completion_tokens: int | None = None
    seed: int | None = None


class ArenaOutput(TaskOutput):
    """App response model for the Arena task.

    Attributes:
        responses: Dictionary mapping model IDs to their responses.
    """

    responses: dict[str, str]


class ArenaTask(Task[ArenaInput, ArenaOutput]):
    """A task that invokes multiple LLMs for comparison.

    Attributes:
        modality: Input modality other than text.
        models: Dictionary mapping model nicknames to their model IDs.
    """

    modality: Modality
    models: dict[str, str]

    def post_init(self) -> None:
        """Initialize subtasks.

        Normally we would save the task instances somewhere, but for testing, we let them go.
        """
        EncoderTask(
            modality=self.modality,
            model_id=list(self.models.values())[0],
            adapter_model_ids=list(self.models.values())[1:],
        )

        MLLMTask(
            model_id=list(self.models.values())[1],
            modalities=[self.modality],
        )

        for model_id in self.models.values():
            LLMTask(model_id=model_id)

    def invoke(self, task_input: ArenaInput) -> ArenaOutput:
        """Invoke the Arena task with the given input."""
        return ArenaOutput(responses={"nice": "yeah"})


def test_arena_task_registration():
    """Test whether subtasks are registered correctly."""
    task = ArenaTask(
        modality=Modality.IMAGE,
        models={
            "llama": "llama-7B",
            "gemma": "gemma-4B",
        },
    )

    # Encoder, MLLM, and two LLM tasks
    assert len(task.subtask_attr_names) == 4

    encoder = getattr(task, "__subtask_0__")
    assert isinstance(encoder, EncoderTask)
    assert encoder.model_id == "llama-7B"
    assert encoder.modality == Modality.IMAGE

    mllm = getattr(task, "__subtask_1__")
    assert isinstance(mllm, MLLMTask)
    assert mllm.model_id == "gemma-4B"
    assert mllm.modalities == [Modality.IMAGE]

    # Encoder and LLM
    assert len(mllm.subtask_attr_names) == 2

    mllm_encoder = getattr(mllm, "__subtask_0__")
    assert isinstance(mllm_encoder, EncoderTask)
    assert mllm_encoder.model_id == "gemma-4B"
    assert mllm_encoder.modality == Modality.IMAGE

    mllm_llm = getattr(mllm, "__subtask_1__")
    assert isinstance(mllm_llm, LLMTask)
    assert mllm_llm.model_id == "gemma-4B"

    llm0 = getattr(task, "__subtask_2__")
    assert isinstance(llm0, LLMTask)
    assert llm0.model_id == "llama-7B"

    llm1 = getattr(task, "__subtask_3__")
    assert isinstance(llm1, LLMTask)
    assert llm1.model_id == "gemma-4B"
