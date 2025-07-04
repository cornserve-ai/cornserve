"""An app that lets users compare different Gemma models."""

from __future__ import annotations

from cornserve.app.base import AppRequest, AppResponse, AppConfig
from cornserve.task.base import Task, TaskInput, TaskOutput
from cornserve.task.builtins.llm import LLMInput, LLMTask
from cornserve.task.builtins.encoder import EncoderInput, EncoderTask, Modality


class ArenaInput(TaskInput, AppRequest):
    """App request model.

    Attributes:
        prompt: The prompt to send to the LLM.
        multimodal_data: List of tuples (modality, data URL).
            "image", "video", etc. for modality.
        max_completion_tokens: Max number of tokens to generate in the response.
        seed: Optional random seed.
    """

    prompt: str
    multimodal_data: list[tuple[str, str]] = []
    max_completion_tokens: int | None = None
    seed: int | None = None


class ArenaOutput(TaskOutput, AppResponse):
    """App response model.

    Attributes:
        responses: Dictionary mapping model IDs to their responses.
    """

    responses: dict[str, str]


class ArenaTask(Task[ArenaInput, ArenaOutput]):
    """A task that invokes multiple LLMs for comparison.

    Attributes:
        modality: Input modality other than text.
        model_ids: Dictionary mapping model nicknames to their model IDs.
    """

    modality: Modality
    models: dict[str, str]

    def post_init(self) -> None:
        """Initialize subtasks."""
        model_ids = list(self.models.values())
        self.encoder = EncoderTask(
            modality=self.modality,
            model_id=model_ids[0],
            adapter_model_ids=model_ids[1:],
        )
        self.llms: list[tuple[str, str, LLMTask]] = []
        for name, model_id in self.models.items():
            task = LLMTask(model_id=model_id)
            self.llms.append((name, model_id, task))

    def invoke(self, task_input: ArenaInput) -> ArenaOutput:
        """Invoke the task with the given input."""
        responses = {}
        for model_name, model_id, llm in self.llms:
            embeddings = self.encoder.invoke(
                EncoderInput(
                    model_id=model_id,
                    data_urls=[url for _, url in task_input.multimodal_data],
                )
            ).embeddings
            response = llm.invoke(
                LLMInput(
                    prompt=task_input.prompt,
                    multimodal_data=task_input.multimodal_data,
                    max_completion_tokens=task_input.max_completion_tokens,
                    seed=task_input.seed,
                    embeddings=embeddings,
                )
            ).response
            responses[model_name] = response
        return ArenaOutput(responses=responses)


task = ArenaTask(
    modality=Modality.IMAGE,
    models={
        "4B": "google/gemma-3-4b-it",
        "12B": "google/gemma-3-12b-it",
        "27B": "google/gemma-3-27b-it",
    },
)

class Config(AppConfig):
    """App configuration model."""

    tasks = {"arena": task}


async def serve(request: ArenaInput) -> ArenaOutput:
    """Main serve function for the app."""
    return await task(request)
