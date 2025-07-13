"""An app that lets users compare different Gemma models."""

from __future__ import annotations
from collections.abc import AsyncGenerator

from cornserve.app.base import AppConfig
from cornserve.task.base import Task, TaskOutput, Stream
from cornserve.task.builtins.llm import OpenAIChatCompletionChunk, OpenAIChatCompletionRequest, LLMUnitTask, extract_multimodal_content
from cornserve.task.builtins.encoder import EncoderInput, EncoderTask, Modality


class ArenaOutput(TaskOutput):
    """App response model.

    Attributes:
        responses: Dictionary mapping model IDs to their responses.
    """

    responses: dict[str, str]


class ArenaTask(Task[OpenAIChatCompletionRequest, Stream[ArenaOutput]]):
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
        self.llms: list[tuple[str, str, LLMUnitTask]] = []
        for name, model_id in self.models.items():
            task = LLMUnitTask(model_id=model_id)
            self.llms.append((name, model_id, task))

    def invoke(self, task_input: OpenAIChatCompletionRequest) -> Stream[ArenaOutput]:
        """Invoke the task with the given input."""
        encoder_input_urls: list[str] = []
        for multimodal_content in extract_multimodal_content(task_input.messages):
            modality = Modality(multimodal_content.type.split("_")[0])
            if modality != self.modality:
                raise ValueError(
                    f"Got unexpected modality {modality.value} in input. "
                    f"Expected {self.modality.value}."
                )
            encoder_input_urls.append(
                getattr(multimodal_content, multimodal_content.type).url
            )

        streams: dict[str, Stream[OpenAIChatCompletionChunk]] = {}
        for model_name, model_id, llm in self.llms:
            if encoder_input_urls:
                encoder_input = EncoderInput(
                    model_id=model_id,
                    data_urls=encoder_input_urls,
                )
                embeddings = self.encoder.invoke(encoder_input).embeddings
            else:
                embeddings = []
            llm_input = task_input.model_copy(deep=True)
            llm_input.cornserve_embeddings = embeddings
            stream = llm.invoke(llm_input)
            streams[model_name] = stream
        
        # XXX: Aggergating multiple streams into a single stream.



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


async def serve(request: OpenAIChatCompletionRequest) -> Stream[ArenaOutput]:
    """Main serve function for the app."""
    return await task(request)
