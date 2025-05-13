"""Task Execution Descriptor.

The `TaskExecutionDescriptor` class describes *how* to execute a `Task`.
The same `Task` may have multiple compatible `TaskExecutionDescriptor`s.
For instance, the builtin `LLMTask` can be executed with a monolithic
vLLM instance, but can also be executed with prefill-decode disaggregation.

A descriptor is compatible with a `Task` when it inherits from the base
descriptor class annotated in the `Task` class's `execution_descriptor` field.

The descriptor exposes the following information:
- Dominant resource type: GPU, CPU, memory, disk, etc.
    This is not implemented yet; all task executors consume GPU resources.
- Chunking and pipelining semantics: Information about what kind of chunking
    and pipelining is supported by the task executor for its input and output.
- Launch information: Information about how to launch the task executor.
- Request and response transformation: How to transform TaskInput to the actual
    Task Executor request object, and the Task Executor response object to the
    TaskOutput object.
"""
