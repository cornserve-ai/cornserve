# Building Apps

Cornserve has two layers of defining *execution*:

- **App**: This is the highest level of construct, which takes a request and returns a response. Apps are written in Python and can be submitted to the Cornserve Gateway for deployment.
- **Task**: This is a unit of work that is executed by the Cornserve data plane. There are two types of tasks:
    - **Unit Task**: Unit Tasks are the smallest and most basic type of task. They are executed in a single Kubernetes Pod and are the unit of scaling. For instance, there is the built-in modality embedding unit task which embeds specific modalities (e.g., image, video, audio), which is executed by our Eric server. There is also the built-in LLM text generation task, which generates text from input text prompts and any embedded modalities.
    - **Composite Task**: Composite Tasks are a composition of one or more Unit Tasks. They are defined by the user in Python. For instance, there is the built-in Multimodal LLM composite task which instantiates modality embedding unit tasks as needed, runs them on multimodal data to embed them, and passes them to the LLM text generation unit task to generate text. Intermediate data produced by unit tasks are forwarded directly to the next unit task in the graph.

## Example: Gemma Arena

Let's look at a real example app that demonstrates streaming responses from different Gemma models simultaneously. This `gemmarena.py` app lets users compare different Gemma models side by side, like an arena.

```python
import asyncio
from collections.abc import AsyncIterator

from cornserve_tasklib.task.composite.llm import MLLMTask
from cornserve_tasklib.task.unit.encoder import Modality
from cornserve_tasklib.task.unit.llm import OpenAIChatCompletionChunk, OpenAIChatCompletionRequest
from pydantic import RootModel

from cornserve.app.base import AppConfig
from cornserve.task.base import Stream

# Comment out any Gemma models you don't want to include in the arena.
gemma_model_ids = {
    "gemma3-4b": "google/gemma-3-4b-it",
    "gemma3-12b": "google/gemma-3-12b-it",
    "gemma3-27b": "google/gemma-3-27b-it",
}


# All Gemma MLLMTasks will share the same encoder task deployment.
gemma_tasks = {
    name: MLLMTask(
        modalities=[Modality.IMAGE],
        model_id=model_id,
        encoder_model_ids=set(gemma_model_ids.values()),
    )
    for name, model_id in gemma_model_ids.items()
}


class ArenaOutput(RootModel[dict[str, str]]):
    """Response model for the app.

    Wrapper around a dictionary that maps model names to generated text chunks.
    """


class Config(AppConfig):
    """App configuration model."""

    tasks = {**gemma_tasks}


async def serve(request: OpenAIChatCompletionRequest) -> AsyncIterator[ArenaOutput]:
    """Main serve function for the app."""
    # NOTE: Doing `await` for each task separately will make them run sequentially.
    tasks: list[asyncio.Task[Stream[OpenAIChatCompletionChunk]]] = []
    for task in gemma_tasks.values():
        # Overwrite the model ID in the request to match the task's model ID.
        request_ = request.model_copy(update={"model": task.model_id}, deep=True)
        tasks.append(asyncio.create_task(task(request_)))
    
    # An await is needed to actually dispatch the tasks.
    # Responses will be streamed after this.
    dispatch_responses = await asyncio.gather(*tasks)

    streams = {
        asyncio.create_task(anext(stream)): (stream, name)
        for stream, name in zip(dispatch_responses, gemma_model_ids.keys(), strict=True)
    }

    while streams:
        done, _ = await asyncio.wait(streams.keys(), return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            stream, name = streams.pop(task)

            try:
                chunk = task.result()
            except StopAsyncIteration:
                # This model is done responding.
                continue

            delta = chunk.choices[0].delta.content
            if delta is None:
                continue
            yield ArenaOutput({name: delta})

            streams[asyncio.create_task(anext(stream))] = (stream, name)
```

Let's break this down.

### Key Components

1. **Built-in Tasks**: The app uses Cornserve's built-in `MLLMTask` (Multimodal LLM Task) rather than defining custom composite tasks. This task handles both image encoding and text generation.

2. **Task Declaration**: Multiple Gemma models are configured with shared encoder deployments:
   ```python
   gemma_tasks = {
       name: MLLMTask(
           modalities=[Modality.IMAGE],
           model_id=model_id,
           encoder_model_ids=set(gemma_model_ids.values()),
       )
       for name, model_id in gemma_model_ids.items()
   }
   ```

3. **App Configuration**: The `Config` class registers all the tasks with the Cornserve platform:
   ```python
   class Config(AppConfig):
       tasks = {**gemma_tasks}
   ```

4. **The Async `serve` function**: This is the main entry point of the app, taking a Pydantic model as input (request). It handles incoming requests and orchestrates parallel execution of multiple unit/composite tasks.
   ```python
   async def serve(
       request: OpenAIChatCompletionRequest,
   ) -> AsyncIterator[ArenaOutput]:
       # Invoke all Gemma tasks concurrently
       # Wait for all three concurrently, and yield chunks as they arrive
       ...
   ```

5. **App Responses**: The app's `serve` function can either return a single response object (Pydantic model), or instead return an async iterator that yields response chunks (each chunk being a Pydantic model) for streaming responses. The Gemma Arena app demonstrates streaming responses.

### Benefits

- **Flexibility**: The app's asynchronous `serve` function allows the full flexibility of Python programming, including concurrency with `asyncio`. This enables complex orchestration patterns, such as parallel execution and streaming responses.
- **Automatic Sharing**: In the app, we defined three Gemma `MLLMTask`s. In fact, the three Gemma models share the exact same vision encoder, so we can share a single vision encoder for all three Gemma tasks. Cornserve automatically detects this (by checking whether the `EncoderTask` inside the three `MLLMTask`s are identical) and only deploys a single vision encoder for all three tasks.


## Next Steps

Now that you've learned how to build apps with Cornserve, the next step is to deploy and run your app. Head over to [Deploying Apps to Cornserve](registering_apps.md) to learn how to register your `gemmarena.py` app and invoke it with real requests.
