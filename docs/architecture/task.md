# Task Abstraction

The definition of tasks as a unit of work and how the system executes them.

Task and GiG (Graph in Graph) are alternative term for the same concept.

## `Task`

A `Task` is a reusable and composable unit of work that is executed within the data plane.
Applications define and invoke one or more Tasks to achieve their goals.
A `Task` logically describes what should be done.

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
However, intermediate data (particularly tensors) are forwarded to next Tasks within the data plane.
That is, when a Task is composed of multiple sub-Tasks, the output of one sub-Task is forwarded to the next sub-Task in the DAG.
Data forwarding is managed by the network of sidecars that are deployed in the data plane.

#### Multi-Entry and Multi-Exit

The DAG defined by a Task can have multiple entry and exit points.
As a concrete example, a Task may define a multimodal Thinker-Talker LLM architecture (e.g., Qwen2.5 Omni).
Requests that come with different modalities (e.g., image-only, image + video, audio + video) are routed to only the multimodal encoders that are needed for the request.
Naturally, requests that do not have any multimodal data are routed straight to the Thinker LLM, without going through any multimodal encoders.
Also, a request that does not ask for speech generation will immediately return after the Thinker LLM, instead of going to the Talker and Vocoder.

The concrete execution DAG of a Task must be statically defined at the time of invocation by a request.
That is, the DAG must be completely determined by the App Driver given the request to the app.
For instance, the input to a Task may hold a field `image_url: str | None`, and whether the execution DAG includes the image encoder can be determined by inspecting whether the `image_url` field is `None` or not.

### Specification

The core specification of a Task is by its execution DAG.

#### Node

A node is a Task instance. Tasks has:
- **Task execution descriptor**: Descriptor instance that describes how the Task is executed.
- **Chunking and pipelining semantics**: Whether certain inputs can be received in a chunking fashion, and whether certain outputs can be materialized in chunking fashion. If chunking is supported, the possible chunk sizes.
- **The `invoke` method**: The method that executes the Task. Input and output are Pydantic models. The Python code in `invoke` puts together the Tasks invocation to implicitly define the execution DAG.

The entry and exit data of the DAG are also Pydantic models.

#### Edge

An edge represents the data flow between two nodes in the DAG. An edge has:
- **Data type**: The type of data that is passed between nodes. Either arbitrary small bytes or structured tensors.
- **Chunking and pipelining semantics**: The reconciled chunking and pipelining semantics between the two nodes.

## `TaskExecutionDescriptor`

A `TaskExecutionDescriptor` strategy class that describes how a Task is executed.
Each concrete `Task` subclass is associated with one or more `TaskExecutionDescriptor` subclasses and takes an instance of the descriptor as an argument to its constructor.

### Examples

The `LLMTask` is compatible with descriptors that are subclasses of `LLMTaskExecutionDescriptor`.
Currently, only vLLM is implemented, but other executors like TensorRT-LLM or Dynamo can be implemented in the future.

Concretely, `MLLMTask` is composed of modality encoders (`EncoderTask`) and a LLM text generation task (`LLMTask`).

### Specification

- **Dominant resource**: The dominant resource that the node uses, e.g., GPU, CPU, memory, or disk.
- **Chunking and pipelining semantics**: In what granularity and pattern does this node allow supplying nodes to chunk and pipeline input for a single invocation?
- **Launch information**: How to launch the Task executor. This includes the command line arguments, environment variables, and other information needed to launch the Task executor on K8s.
- **Request and response schema**: The request and response API schema of the Task executor as Pydantic models.


## Task Lifecycle

### Registration

When an App is registered, the App Manager constructs the task DAG and sends it to the Resource Manager, which ensures Task Managers for each required Task are deployed in the data plane.

### Invocation

When an App Driver receives a request, it invokes the Tasks defined in the app by calling and awaiting on their async `__call__`.
This internally calls all `invoke` methods of Tasks in the DAG to populate a task-specific `TaskContext` object.
This context object is then used to dispatch task invocations to the Task Dispatcher.

The Task Dispatcher is responsible for actually constructing requests, dispatching them to Task Executors, and then waiting for the results to come back.  
1. For each task invocation:
  - Call `GetRoute` on each Task's Task Manager to get the route to the Task Executor.
  - Find all `DataForward` objects in the Task's input and output, and add them to a dictionary of `DataForward` ID → `ForwardInfo` object.
    - `ForwardInfo` contains the producer sidecar ranks (if `DataForward` was the output) and the consumer sidecar ranks (if `DataForward` was the input).
2. Inspect the `ForwardInfo` objects to see if any of them are without producers or consumers. If any, it's an error.
3. For each task invocation:
  - For all `DataForward` objects in the Task's input and output, populate their sidecar ranks from `ForwardInfo` objects.
4. For each task invocation:
  - Translate the Task's input model to the Task Executor's request using `TaskExecutionDescriptor`.
    - This requires the Task's input and output objects (`DataForward` objects filled in). The descriptor will fill in the receiver sidecar ranks of the `DataForward` objects in the Task's output.
  - Upon translation, create a new `asyncio.Task` that sends the request to the central task dispatch scheduler.
5. Wait for the list of `asyncio.Task` objects:
  - For each finished Task, translate the Task Executor's response Pydantic model to the Task's output Pydantic model using `TaskExecutionDescriptor`.
    - This requires the Task Executor's response object and Task's input and output objects (`DataForward` objects filled in). The descriptor will fill in relevant fields in the Task's output object from the Task Executor's response. At this point, `DataForward` objects are not important; they can actually be anything.
  - Upon translation, add it to the `TaskResponse` object.
6. Return the `TaskResponse` object to the App Driver.

When the App Driver receives the `TaskResponse` object, it will set a flag in the `TaskContext` object to indicate that the Task has finished.
Then, the `invoke` method will be called again, and this time, each unit Task's `invoke` method will return the actual Task output instead of a placeholder object, and the top-level `invoke` will construct the final result.
This result goes back to the App Driver.

### Deregistration

When an App is unregistered and if there are no other active Apps that require the Task, the Resource Manager will kill the Task Manager and free up the resources.


## Implementation of `Task`

```python
class EncoderInput(BaseModel):
    embedding_data: list[str]

class EncoderOutput(BaseModel):
    embeddings: list[Forward[Tensor]] | None

class LLMInput(BaseModel):
    embeddings: list[Forward[Tensor]] | None
    prompt: str

class LLMOutput(BaseModel):
    response: str

class TaskInput(BaseModel):
    multimodal_items: list[tuple[str, str]]
    prompt: str

class TaskOutput(BaseModel):
    response: str

# Generic `DataT` is actually not needed because both the sender and receiver know exactly what type of data they are sending and receiving.
# Based on `DataT`, we can decide whether or not to use chunking. Basically, we only do that for tensors.
# Note that if something can be either forwarded or passed as is into the receiver request, it can be `Forward[DataT] | DataT`.
class Forward(BaseModel, Generic[DataT]):  # Fields are private. To convey necessary data transmission info to sidecars.
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    # chunk_size: int | None = None  # Filled in later by the TD by looking at the src–dst edge info
    src_sidecar_ranks: list[int] | None = None  # Filled in later by the TD
    dst_sidecar_ranks: list[int] | None = None  # Filled in later by the TD

def invoke(task_input: TaskInput) -> TaskOutput:
    image_items = [d for t, d in task_input.multimodal_items if t == "image"]
    if image_items:
        encoder_input = EncoderInput(embedding_data=image_items)
        encoder_output = self.image_encoder.invoke(encoder_input).embeddings
    else:
        encoder_output = None

    llm_input = LLMInput(embeddings=encoder_output, prompt=task_input.prompt)
    llm_output = self.llm.invoke(llm_input)

    return TaskOutput(response=llm_output.response)
```

The task graph is created via AST static analysis of the `invoke` method.

The `invoke` method runs immediately until completion, and each `invoke` method appends a task invocation to a Context Variable `app_context`.

The `invoke` method of a unit task (e.g., `LLMTask`) takes in the input object and sends it to the Task Dispatcher.

```python
task_context: ContextVar[TaskContext] = ContextVar("task_context")

class EncoderTask(Task):
    def make_emulated_output(self, task_input: EncoderInput) -> EncoderOutput:
        return EncoderOutput(
            embeddings=[Forward[torch.Tensor]() for _ in task_input.embedding_data],
        )

    def invoke(self, task_input: EncoderInput) -> EncoderOutput:
        ctx = task_context.get()

        if ctx.is_dispatching:
            task_output = self.make_emulated_output(task_input)
            ctx.invocations.append(
                TaskInvocation(
                    task=self,
                    input=task_input,
                    output=task_output,
                )
            )
        else:
            task_output = ctx.task_outputs[self.id]

        return task_output
```

The base class `Task` has an async `run` method that is intended to be called by the App Driver.
This method is what does the heavy lifting of actually driving the Task's execution.

```python
async def __call__(self, task_input: TaskInput) -> TaskOutput:
    # Create new task context
    task_context.set(TaskContext(task_id=self.id))
    return await asyncio.create_task(self._call_impl(task_input))

async def _call_impl(self, task_input: TaskInput) -> TaskOutput:
    # Fetch the task context.
    ctx = task_context.get()

    # Run the invoke method to trace and record task invocations.
    # The context manager will have all task invocations record their invocations inside the context.
    with ctx.record():
        _ = self.invoke(task_input)

    # Dispatch all tasks to the Task Dispatcher and wait.
    # Wait for all tasks to finish
    await ctx.dispatch_tasks_and_wait()

    # Re-run the invoke method to produce the final result of the task.
    # The context manager will have all tasks directly use actual task outputs.
    with ctx.replay():
        return self.invoke(task_input)
```


## Execution of multiple Tasks

Cornserve is essentially a system that executes numerous Tasks on a shared pool of resources.

### Performance profile

Performance profiles treat chunking as a first class citizen.
The unit of throughput and latency is for a single chunk of data as allowed by the Task's chunking semantics.

For a node (`Task`), its performance profile is determined as a function of:
- the amount of dominant resource given to the Task (e.g., number of GPUs)
- the application (different applications may have different input/output data statistics)
- the `TaskExecutionDescriptor` instance (how the Task is executed has significant impact)

For an edge, its performance profile is determined as a function of:
- the communication medium (e.g., local shared memory, NVLink, Infiniband, Ethernet)
- the application

### Logical cluster task graph

The cluster task graph is a DAG that was constructed by merging all unique Task DAGs of all active apps, which provides a global view of ongoing tasks and data flow.
*Composite* Tasks are broken down into *unit* Tasks that do not contain other Tasks in them.
These unit Tasks are the unit of horizontal scaling.

Importantly, nodes (Tasks) can be shared across apps either when they are identical, or special sharing mechanisms (e.g., S-LoRA) exist.

### Resource allocation

Resource: Number of GPUs, data transmission medium (NVLink > IB > Ethernet), and data transmission bandwidth  
Allocation: On the cluster task graph, nodes receive GPUs, edges receive data transmission medium and bandwidth  

Potential goals
- No bottleneck in the system. This does not take into account application-level latency SLO or fairness.  
- When multiple edges (in the task graph) share a single data transmission medium, bandwidth should be allocated while taking each edge’s data transmission volume and chunking pattern into account. For instance, when chunking & pipelining is happening, a significant delay in the final chunk (e.g., due to suboptimal communication volume allocation) delays the whole task invocation, so avoiding this makes sense. Or, if a certain application sends in images that are on average larger than another app that shares that edge, some kind of appropriate fair allocation or volume-proportional bandwidth allocation could be done.

### Physical cluster task graph

The physical graph is created from the logical graph after allocating and placing resources to all nodes and edges.
Nodes are physical instantiations of task executors and GPUs (nodes) and communication medium assignments for data forwarding between task executors (edges).

### Reallocation events

Specific events trigger reallocation of resources:
- **New app registration**: The list of tasks invoked by the new app is merged into the current logical cluster task graph. This may not change the graph at all, but since more applications exert that execution path in the DAG, a reallocation may be desirable.
- **App deregistration**: When an app is unregistered, tasks that no longer have apps that invoke them are removed from the logical cluster task graph and their resources are collected. Remaining resources are then reallocated to existing Tasks.
- **Periodic reallocation**: The load monitor continuously monitors the performance profile changes of each node and edge, and periodically triggers reallocation in a slow loop to optimize allocation based on dynamic application behavior.

### Task dispatch

After spinning up all necessary Subtask Managers, the RM sends the Task DAG to the Task Dispatcher.  

The Task Dispatcher is actually not a normal DAG executor (i.e., child dispatches after all parents are done).
Recall, we dispatch requests to Eric and vLLM at the same time, and waiting for data is done *inside* vLLM and the sidecar!
When a Task in invoked, the Task Dispatcher dispatches the **whole DAG** to the data plane, and waiting for data to be forwarded is done inside the data plane.
This asynchronous waiting is actually enabled by the decoupled and non-blocking nature of our sidecar.
