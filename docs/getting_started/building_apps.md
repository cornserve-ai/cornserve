# Building and Deploying Your App

Cornserve as two layers of defining *execution*:

- **App**: This is the highest level of construct, which takes a request and returns a response. Apps are written in Python and can be submitted to the Cornserve Gateway for deployment.
- **Task**: This is a unit of work that is executed by the Cornserve data plane. There are two types of tasks:
    - **Unit Task**: Unit Tasks are the smallest and most basic type of task. They are executed in a single Kubernetes Pod and are the unit of scaling. For instance, there is the built-in modality embedding unit task which embeds specific modalities (e.g., image, video, audio), which is executed by our Eric server. There is also the built-in LLM text generation task, which generates text from input text prompts and any embedded modalities.
    - **Composite Task**: Composite Tasks are a composition of one or more Unit Tasks. They are defined by the user in Python. For instance, there is the built-in Multimodal LLM composite task which instantiates modality embedding unit tasks as needed, runs then on multimodal data to embeds them, and passes them to the LLM text generation unit task to generate text. Intermediate data produced by unit tasks are forwarded directly to the next unit task in the graph.

## Example: Writing an Image Understanding App

Apps are written in Python and use Tasks to process requests.
Let's build a simple example app that takes an image and a text prompt, and generates a response based on the image and the prompt.

### Composite Task

First, let's see how to build a composite task out of built-in unit tasks for this: `#!python ImageChatTask`.

```python
from cornserve.task import Task, TaskInput, TaskOutput
from cornserve.task.builtins.encoder import EncoderTask, Modality, EncoderInput
from cornserve.task.buildins.llm import LLMTask, LLMInput
from cornserve.app.base import AppRequest, AppResponse, AppConfig


class ImageChatInput(TaskInput):
    prompt: str
    image_url: str


class ImageChatOutput(TaskOutput):
    response: str


class ImageChatTask(Task[ImageChatInput, ImageChatOutput]):
    model_id: str

    def post_init(self) -> None:
        """Initialize subtasks."""
        self.image_encoder = EncoderTask(
            model_id=self.model_id,
            modality=Modality.IMAGE,
        )
        self.llm = LLMTask(model_id=self.model_id)

    def invoke(self, task_input: ImageChatInput) -> ImageChatOutput:
        """Invoke the task."""
        encoder_input = EncoderInput(data_urls=[task_input.image_url])
        image_embedding = self.image_encoder.invoke(encoder_input)
        llm_input = LLMInput(
            prompt=task_input.prompt,
            multimodal_data=[("image", task_input.image_url)],
            embeddings=[image_embedding],
        )
        llm_output = self.llm.invoke(llm_input)
        return ImageChatOutput(response=llm_output.response)
```

It was a handful of code, so let's break it down:

1. **Input/Output Models**: We define `ImageChatInput` and `ImageChatOutput` using Pydantic. This allows us to define clear input and output models for our task. These should inherit from `TaskInput` and `TaskOutput`, respectively.
2. **Task Class**: We define a new composite task class called `ImageChatTask` that inherits from `Task[ImageChatInput, ImageChatOutput]`. This class specifies two things:
    - **Subtasks**, namely the built-in `EncoderTask` and `LLMTask`, which are instantiated in the `post_init()` method. This is where we define the subtasks that will be used in the task.
    - **Task logic**. Each unit task (e.g., `EncoderTask`) expects its input data to be an instance of its `TaskInput` (e.g., `EncoderInput`), and returns an instance of its `TaskOutput` (e.g., `EncoderOutput`). The `invoke` method is where we define the logic of how the subtasks are composed together to produce the final output.

### App

With `#!python ImageChatTask` defined, we can now use it in our app:

```python
from cornserve.app.base import AppRequest, AppResponse, AppConfig

image_chat = ImageChatTask(model_id="Qwen/Qwen2-VL-7B-Instruct")


class Request(AppRequest):
    image_url: str
    prompt: str


class Response(AppResponse):
    response: str


class Config(AppConfig):
    tasks: {"image_chat": image_chat}


async def serve(request: Request) -> Response:
    """App's main entry point that serves a request."""
    image_chat_input = ImageChatInput(
        prompt=request.prompt,
        image_url=request.image_url,
    )
    image_chat_output = await image_chat(image_chat_input)
    return Response(response=image_chat_output.response)
```

This app only uses a single composite task, `#!python ImageChatTask`, but it should be easy to see that you can use arbitrary numbers of unit and composite tasks in your app.

Another thing to note is that **the app's main entry point is an async function called `serve`**.
This is the function that will be called by the Cornserve Gateway when a request is received.

Finally, notice that when you compose tasks inside composite tasks, you called the `invoke` method of tasks synchronously.
However, in the context of apps, **you call the `__call__` method of tasks asynchronously**.
This allows you to run multiple tasks in parallel with usual Python asynchronous programming patterns like `#!python asyncio.gather`.

## Debugging

We've just showed how to build a simple app.
However, having the build the entire thing in one shot is not the most convenient.
In the [next page](jupyter.ipynb), we'll show how you can interactively build and debug your task and app logic in Jupyter Notebook!
