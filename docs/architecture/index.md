# Cornserve Architecture

Cornserve is a disaggregated ML serving platform that allows you to implement and deploy ML applications on your infrastructure.

!!! Important
    This document includes descriptions of Cornserve that has not been developed yet.
    Please track [issues on GitHub](https://github.com/cornserve-ai/cornserve/issues) for the latest updates.


## Task and App

Applications are written by developers using the Cornserve frontend library in `cornserve.frontend`.
Currently, a Cornserve app is a single Python file that implements three classes and an async function:
- `Request` (inherits from `cornserve.app.base.AppRequest`): A single input request for the app.
- `Response` (inherits from `cornserve.app.base.AppResponse`): A single output response for the app.
- `Config` (inherits from `cornserve.app.base.AppConfig`): Configuration parameters and task definitions for the app.
- `async def serve(request: Request) -> Response`: The main function that handles the request and returns a response.

```python
from cornserve.app.base import AppRequest, AppResponse, AppConfig
from cornserve.task.builtins.mllm import MLLMInput, MLLMTask, Modality


class Request(AppRequest):
    """App request model.

    Attributes:
        prompt: The prompt to send to the LLM.
        multimodal_data: List of tuples (modality, data URL).
    """

    prompt: str
    multimodal_data: list[tuple[str, str]] = []


class Response(AppResponse):
    """App response model.

    Attributes:
        response: The response from the LLM.
    """

    response: str


mllm = MLLMTask(
    model_id="Qwen/Qwen2-VL-7B-Instruct",
    modalities=[Modality.IMAGE],
)


class Config(AppConfig):
    """App configuration model."""

    tasks = {"mllm": mllm}


async def serve(request: Request) -> Response:
    """The function that will be run when a request hits the app."""
    mllm_input = MLLMInput(prompt=request.prompt, multimodal_data=request.multimodal_data)
    mllm_output = await mllm(mllm_input)
    return Response(response=mllm_output.response)
```

Importantly, in app configurations, apps specify the tasks that they intend to invoke.
These tasks are *dispatched* to be executed on the data plane.
There are built-in tasks under `cornserve.task.builtins`, such as `MLLMTask` for multimodal LLM inference, `LLMTask` for LLM inference, and `EncoderTask` for multimodal data embedding, and users can build their own tasks using components from `cornserve.task.base`.
All other inline Python code is executed in place by the Cornserve Gateway.

See also the dedicated page on [tasks](task.md).


## Control Plane

The control plane manages numerous registered apps and handles incoming requests to each app.

Control plane components generally send and receive control signals using gRPC (`proto/v1/*.proto`).
On the other hand, application requests and task invocations are sent and received using HTTP.

### Gateway and App Manager

Package: `cornserve.services.gateway`

The gateway is the entry point for all apps and incoming requests to each app.

An app is registered with Cornserve by the cluster admin by sending a request to the gateway, including the app's Python source code as string.
The gateway then validates the app definition (primarily whether it has all the required classes and the `serve` function) and registers it with the App Manager singleton class.

When a new app is registered, the App Manager will read in the tasks that the app intends to invoke and instruct the Resource Manager to deploy Task Managers in the data plane such that all tasks invoked by the new app is available for execution in the data plane.
There is only a single Task Manager per task, so multiple apps that invoke the same task will share a single Task Manager.

When a request for a registered app is received, the gateway will spawn a new App Driver for the app to handle the request.
The App Driver will execute the app's `serve` function, and invoked tasks will be sent to the Task Dispatcher, which handles actually executing the task in the data plane and retrieving results back to the App Driver.

### Resource Manager

Package: `cornserve.services.resource_manager`

The Resource Manager is primarily responsible for allocating cluster resources (primarily GPUs) to Task Managers.

There are two primary events that trigger the Resource Manager to allocate resources:

1. **New app registration.** When a new app is registered and its required tasks are sent to the Resource Manager, the Resource Manager figures out the Task Managers that need to be deployed in the data plane. If there was already an app that required some of the tasks, those Task Managers will not be deployed again, but rather shared by those apps.
2. **App unregistration.** When an app is unregistered, the Resource Manager will check if there are any other apps that require the same tasks. If not, those unnecessary Task Managers will be killed.

Beyond app registration and unregistration, the Resource Manager also dynamically adjusts the amount of resources given to each Task Manager.
Say, if a certain Task Manager receives more requests than others, or if it is computationally heavy and cannot serve as many requests per second compared to other tasks, the Resource Manager will dynamically provision more resources for it.
This will happen at the cost of taking away resources from other Task Managers, if need be.
The goal would be to balance the request throughput of the whole system over time given a fixed amount of resource.

### Task Manager

Package: `cornserve.services.task_manager`

A Task Manager is responsible for executing a single unit task given a subset of the cluster's resources and exposing information about their performance characteristics.
A task, for instance, can be an LLM inference task with a particular model; an LLM inference task with a different model, for instance, is considered a different task.

Task Managers spawn one or more Task Executors that will actually perform task execution on GPUs on the data plane.
The Task Manager is responsible for managing the lifecycle of the Task Executors, including spawning and killing them as needed.
When there are more than one Task Executors deployed under a Task Manager, the Task Manager will also route and load balance requests across the Task Executors.

For multimodal data embedding tasks, the Task Manager will use [Eric](eric.md) as the Task Executor by default.
For LLM inference tasks, the Task Manager will use [our fork of vLLM](https://github.com/cornserve-ai/vllm) as the Task Executor.

The Task Manager also exposes performance characteristics of the Task Executors.
For instance, given $N$ GPUs, the Task Manager will profile the Task Executor's throughput and latency and expose the throughput--latency Pareto frontier.
The Resource Manager can make better resource allocation decisions based on this information.

### Task Dispatcher

Package: `cornserve.services.task_dispatcher`

App Drivers or interactive Jupyter notebook users send task invocation requests to the Task Dispatcher, which is responsible for dispatching the requests to appropriate Task Executors and retrieving the results back to the App Driver.

When a composite task is invoked, the following happens:
1. The composite task's `__call__` method records the unit task invocations and dispatches all of them to the Task Dispatcher.
    - For instance, if the composite task is a Vision-Language Model task, its `__call__` method will record two unit task invocations: one for the image encoder and one for the LLM text generation.
2. (For each unit task invocation) The Task Dispatcher queries the Task Manager for the Task Executor that is best suited to handle the request.
3. (For each unit task invocation) The Task Dispatcher translates the `TaskInput` object of the unit task invocation into JSON, dispatches the HTTP request to the selected Task Executor, waits for the result, and translates the result back to a `TaskOutput` object.
4. Finally, the Task Dispatcher aggregates all unit task invocation results and response with the final result.

How does the Task Dispatcher know how to contact the Task Manager?
Whenever there is a change to Task Managers (spawning new ones or killing existing ones), the Resource Manager will inform the Task Dispatcher of the mapping between the unit task definition and its corresponding Task Manager's endpoint.


## Data Plane

The data plane is where the actual task execution happens on GPUs.

### Task Executor

Package: `cornserve.task_executors`

As detailed in the [Task](task.md) page, a single unit task class is associated with a Task execution descriptor, which provides information about how to spin up the Task Executor and how to execute the task, among other things.

Refer to [Eric](eric.md) and [vLLM](https://github.com/cornserve-ai/vllm) for more information about built-in Task Executors. Cornserve currently uses vLLM v0.9.0.1.

### Sidecar

Package: `cornserve.services.sidecar`

Data plane Task Executors need to communicate tensor data between each other.
A concrete example would be Eric sending the encoded image/video tensor to vLLM for text generation.
See [Sidecar](sidecar.md) for more about the Sidecar service.
