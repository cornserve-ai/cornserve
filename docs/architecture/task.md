# Task Abstraction

This page explains the definition of tasks as a unit of work and how the system executes them.

## `Task`

A `Task` is a reusable and composable unit of work that is executed within the data plane.
Applications define and invoke one or more Tasks to achieve their goals.
A `Task` logically describes what should be done.

A task can be either a Unit Task, or a Composite Task.
The former is a single atomic unit of work, while the latter is a composition of multiple Unit Tasks.

### Examples

A `Task` can be:  

- a single inference of a neural network on GPUs (e.g., Text encoder, Vision encoder, Audio encoder, LLM, DiT, VAE decoder, Vocoder)
- a composition of the above inference units (e.g., a Vision-Language model, a Thinker-Talker architecture model)

Concretely, `MLLMTask` is a composition of `EncoderTask` and `LLMTask`.

### Properties

#### Recursive Composition

Tasks are recursively composed; a Task can be a single inference of a neural network on GPUs or a DAG of other Tasks that make up a larger chunk of coherent work.

#### Data Forwarding

The inputs and outputs to a Task are defined by the Task itself, and both are stored in the App Driver.
However, intermediate data (particularly tensors) are forwarded to next Tasks within the data plane via the [Sidecar](sidecar.md).
That is, when a Task is composed of multiple sub-Tasks, the output of one sub-Task is forwarded to the next sub-Task in the DAG.

#### Static Graph Given Task Input

The concrete execution DAG of a Task must be statically determined at the time of invocation by a request.
That is, the DAG must be completely determined by the App Driver given the request to the app.
In other words, there can not be any dynamic control flow that depends on unmaterialized intermediate data.
For instance, the input to a Task may hold a field `image_url: str | None`, and whether the execution DAG includes the image encoder can be determined by inspecting whether the `image_url` field is `None` or not.

### Specification

The core specification of a Task is by its execution DAG.
Each node is a Task instance that has:

- **Task execution descriptor**: Descriptor instance that describes how the Task is executed.
- **The `invoke` method**: The method that executes the Task. Input and output are Pydantic models. The Python code in `invoke` puts together the Tasks invocation to implicitly define the execution DAG.

## `TaskExecutionDescriptor`

A `TaskExecutionDescriptor` strategy class that describes how a Task is executed.
Each concrete `Task` subclass is associated with one `TaskExecutionDescriptor` subclasses and takes an instance of the descriptor as an argument to its constructor.

### Examples

The `LLMTask` is compatible with the `VLLMDescriptor`, which describes how to execute the LLM task using vLLM.
Currently, only vLLM is implemented, but other executors like TensorRT-LLM or Dynamo can be implemented in the future.
Similarly, the `EncoderTask` is compatible with the `EricDescriptor`, which describes how to execute the encoder task using Eric.

## Task Lifecycle

### Registration

Unit Tasks classes (e.g., `LLMTask`) are registered with the whole system.
Their source code (concrete class definition) should be available to all services in the system.
At the moment, we create multiple built-in Unit Task classes under `cornserve.task.builtins`.

### Deployment

A Unit Task class that is registered in the system can be deployed on the data plane as a Unit Task instance (e.g., `LLMTask(model_id="llama")`).

1. The Unit Task object is instantiated externally, and then serialized into JSON via Pydantic.
2. The name of the Unit Task (as registered in the system) and the serialized JSON are sent to the Gateway service.
3. The Gateway service sends the unit task instance to the Resource Manager, which ensures that the Task Manager for the Unit Task is running on the data plane.
4. When the Task Manager is running, the Resource Manager notifies the Task Dispatcher with the unit task instance and Task Manager deployment information.

Deployed Unit Tasks become invocable, either as part of Composite Tasks or directly.
Invocation can be driven by a static App driver registered in the Gateway service, a human user via our Jupyter Notebook interace.

### Invocation

Task invocations go to the Task Dispatcher by calling and awaiting on the async `__call__` method of the Task.
This internally calls all `invoke` methods of Tasks in the DAG, where each unit Task constructs a `TaskInvocation` object (task, input, and output) to add to a task-specific `TaskContext` object.
The list of `TaskInvocation` objects are sent to the Taks Dispatcher.

The Task Dispatcher is responsible for actually constructing requests, dispatching them to Task Executors, waiting for the results to come back, and then returning task outputs to the App Driver.  

### Deregistration

When an App is unregistered and if there are no other active Apps that require the Task, the Resource Manager will kill the Task Manager and free up the resources.
