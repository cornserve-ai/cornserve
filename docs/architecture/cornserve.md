# CornServe Architecture

CornServe is a distributed ML serving platform that allows you to implement and deploy ML apps at scale.


## App

Applications are written by developers using the CornServe frontend library in `cornserve.frontend`.
Currently, a CornServe app is a single Python file that implements three classes an a function:
- `Request` (inherits from `cornserve.frontend.app.AppRequest`): A single input request for the app.
- `Response` (inherits from `cornserve.frontend.app.AppResponse`): A single output response for the app.
- `Config` (inherits from `cornserve.frontend.app.AppConfig`): Configuration parameters and task definitions for the app.
- `async def serve(request: Request, config: Config) -> Response`: The main function that handles the request and returns a response.

```python
from cornserve.frontend.app import AppConfig, AppRequest, AppResponse
from cornserve.frontend.tasks import LLMTask


class Request(AppRequest):
    """Request class for the VLM app."""
    
    image_url: str
    prompt: str


class Response(AppResponse):
    """Response class for the VLM app."""
    
    text: str


vlm = LLMTask(
    modalities={"text", "image"},
    model_id="Qwen/Qwen2-VL-7B-Instruct",
)


class Config(AppConfig):
    """Config class for the VLM app."""
    
    tasks = {"vlm": vlm}


async def serve(request: Request) -> Response:
    """Serve a single VLM request."""
    response = await vlm.invoke(
        prompt=request.prompt,
        multimodal_data=[("image", request.image_url)],
    )
    return Response(text=response)
```

Importantly, in app configurations, app specify the tasks that they intend to invoke.
Available tasks, e.g., `LLMTask`, are defined in `cornserve.frontend.tasks`, and tasks are *dispatched* to be executed by the data plane GPUs.
All other inline Python code is executed in place by the control plane App Driver.


## Control Plane

The control plane manages numerous registered apps and handle incoming requests to each app.

Control plane components generally send and receive control signals using gRPC (`proto/v1/*.proto`).
On the other hand, application requests and task invocations are sent and received using HTTP.

### Gateway and App Manager (`cornserve.services.gateway`)

The gateway is the entry point for all apps and incoming requests to each app.

An app is registered with CornServe by the cluster admin by sending a request to the gateway, including the app's Python source code.
The gateway then validates the app definition and registers it to with the App Manager.

When a new app is registered, the App Manager will read in the tasks that the app intends to invoke and instruct the Resource Manager to deploy Task Managers in the data plane such that all tasks invoked by the new app is available for execution in the data plane.
There is only a single Task Manager per task, so multiple apps that invoke the same task will share a single Task Manager.

When a request for a registered app is received, the gateway will spawn a new App Driver for the app to handle the request.
The App Driver will invoke the app's `serve` function, and invoked tasks will be sent to the Task Dispatcher, which handles actually executing the task in the data plane and retrieving results back to the App Driver.

### Resource Manager (`cornserve.services.resource_manager`)

The Resource Manager is primarily responsible for allocating cluster resources (primarily GPUs) to Task Managers.

There are two primary events that trigger the Resource Manager to allocate resources:
1. **New app registration.** When a new app is registered and its required tasks are sent to the Resource Manaager, the Resource Manager figures out the Task Managers that need to be deployed in the data plane. If there was already an app that required some of the tasks, those Task Managers will not be deployed again, but rather shared by those apps.
2. **App unregistration.** When an app is unregistered, the Resource Manager will check if there are any other apps that require the same tasks. If not, those unnecessary Task Managers will be killed.

Beyond app registration and unregistration, the Resource Manager also dynamically adjusts the amount of resources given to each Task Manager.
Say, if a certain Task Manager receives more requests than others, or if it is computationally heavy and cannot serve as much requests per second compared to other tasks, the Resource Manager will dynamically provision more resources for it.
This will happen at the cost of taking away resources from other Task Managers, if need be.
The goal would be to balance the request throughput of the whole system over time given a fixed amount of resource.

### Task Manager (`cornserve.services.task_manager`)

A Task Manager is responsible for executing a single task given a subset of the cluster's resources and exposing information about their performance characteristics.
A task, for instance, can be an LLM inference task with a particular model; an LLM inference task with a different model, for instance, is considered a different task.

Task Managers spawn one or more Task Executors that will actually perform task execution on GPUs in the data plane.
The Task Manager is responsible for managing the lifecycle of the Task Executors, including spawning and killing them as needed.
When there are more than one Task Executors deployed under a Task Manager, the Task Manager will also load balance the requests across the Task Executors.

For multimodal data embedding tasks, the Task Manager will use [Eric](eric.md) as the Task Executor.
For LLM inference tasks, the Task Manager will use vLLM as the Task Executor.

The Task Manager also profiles and exposes performance characteristics of the Task Executors.
For instance, given $N$ GPUs, the Task Manager will profile the Task Executor's throughput and latency and expose the throughput--latency tradeoff curve.
The Resource Manager can make better resource allocation decisions based on this information.

### Task Dispatcher (`cornserve.services.task_dispatcher`)

App Drivers send task invocation requests to the Task Dispatcher, which is responsible for dispatching the requests to appropriate Task Executors and retrieving the results back to the App Driver.

For a given task invocation request, the Task Dispatcher:
1. Rewrites the HTTP request to match what is expected by each Task Executor. In this process, a multimodal LLM inference request will be broken down into one multimodal embedding request (for Eric) and one LLM text generation request (for vLLM).
2. (For each sub-request) Queries the Task Manager for the Task Executor that is best suited to handle the request.
3. (For each sub-request) Sends the request to the Task Executor and waits for the result.
4. Aggregates sub-request results and response to the App Driver with the final result.

Whenever there is a change to Task Managers (spawning new ones or killing existing ones), the Resource Manager will inform the Task Dispatcher.


## Data Plane

The data plane is where the actual task execution happens on GPUs.

### Task Executor (`cornserve.task_executors`)

There is one canonical Task Executor for each task type.
Note that the task type here (`cornserve.services.task_manager.models.TaskManagerType`) is different from the task types available in the app frontend library (`cornserve.frontend.tasks`).
That is, the former is a more fine-grained task type that maps to specific task executors (e.g., separated multimodal data encoding and LLM text generation), whereas the latter is more coarse-grained and maps the unit of task execution that our developer would be familiar with.

The Task Executor for `TaskManagerType.ENCODER` is [Eric](eric.md), and the Task Executor for `TaskManagerType.LLM` is [vLLM](vllm.md).
All task executors are implemented in `cornserve.task_executors`, and `cornserve.task_executors.launch` provides information on how to launch the task executors.

### Tensor Sidecar (`cornserve.services.sidecar`)

Data plane Task Executors sometimes have to communicate tensor data between each other.
A concrete example would be Eric sending the encoded image/video tensor to vLLM for text generation.

We may consider using NCCL within those Task Executors for high-performance tensor communication.
However, there are notable roadblocks to this approach:
1. Task Executors are designed to be ephemeral (with somewhat frequent killing and spawning as resource is adjusted), but NCCL communicators are designed to be static and long-lived. This means that we would have to re-establish NCCL communicators every time a Task Executor is killed and spawned, interrupting all communication across the cluster.
2. NCCL spins up a high-speed polling CUDA kernel that takes up the GPU's SM, leading to potential performance degradation for actual computation tasks that should be running on the GPU.

To avoid these issues, we use a sidecar process that runs alongside each Task Executor.
Sidecar processes are bound with each other using GLOO, a CPU-based communication library.

Task Executors write generated tensors to a shared memory (`/dev/shm`) buffer that is shared by both the Task Executor and the sidecar, and signals the sidecar to send the tensor to a specific receiver sidecar.
The sender sidecar will then transfer the tensor to the receiver sidecar's shared memory buffer, which can be read by the receiver Task Executor.
