### Pyright error guide (cornserve)

This document categorizes the current Pyright errors and provides concrete fixes with exact file references. Apply the config changes first, then address the few code tweaks.

- Missing imports: `cornserve_tasklib` not resolved
  - Root cause: Editor/type checker cannot see the sibling package `python-tasklib`.
  - Fix one:
    - Install both local packages in the environment:
      ```bash
      pip install -e /home/yizhuoliang/cornserve/python-tasklib && pip install -e /home/yizhuoliang/cornserve/python
      ```
    - Or add Pyright search paths in `python/pyproject.toml`:
      ```toml
      [tool.pyright]
      extraPaths = ["../python-tasklib", "../python-tasklib/build/lib"]
      ```
  - Examples:
    ```42:47:/home/yizhuoliang/cornserve/examples/mllm.py
    from cornserve_tasklib.task.composite.llm import MLLMTask
    from cornserve_tasklib.task.unit.encoder import Modality
    from cornserve_tasklib.task.unit.llm import OpenAIChatCompletionChunk, OpenAIChatCompletionRequest
    ```
    ```34:37:/home/yizhuoliang/cornserve/examples/gemmarena.py
    from cornserve_tasklib.task.composite.llm import MLLMTask
    from cornserve_tasklib.task.unit.encoder import Modality
    from cornserve_tasklib.task.unit.llm import OpenAIChatCompletionChunk, OpenAIChatCompletionRequest
    ```
    ```7:9:/home/yizhuoliang/cornserve/python/cornserve/task_executors/huggingface/api.py
    from cornserve_tasklib.task.unit.llm import StreamOptions
    from pydantic import BaseModel
    ```

- Generated protobuf stubs are incompatible with Pyright
  - Symptoms: "tuple[Literal['...']] is not assignable to tuple[()]" on `__slots__` in `*_pb2.pyi` files.
  - Fix: Exclude all generated protobuf `.py` and `.pyi` from Pyright (you already exclude the `.py`):
    ```toml
    [tool.pyright]
    exclude = [
      "**/*_pb2.py",
      "**/*_pb2_grpc.py",
      "**/*_pb2.pyi",
      "**/*_pb2_grpc.pyi",
      "cornserve/task_executors/eric/models/*.py",
    ]
    ```
  - Examples:
    ```9:19:/home/yizhuoliang/cornserve/python/cornserve/services/pb/resource_manager_pb2.pyi
    class DeployUnitTaskRequest(_message.Message):
        __slots__ = ("task_instance_name",)
    ```
    ```19:27:/home/yizhuoliang/cornserve/python/cornserve/services/pb/sidecar_pb2.pyi
    class RegisterRequest(_message.Message):
        __slots__ = ("rank", "group", "dtype", "send_slot_numel", "recv_slot_numel", "concurrent_copy")
    ```

- Incorrect use of ExceptionGroup handling in dispatcher
  - Symptoms: "Cannot access attribute 'exceptions' for class 'Exception'".
  - Fix: Catch `ExceptionGroup` separately and stop using `e.exceptions` on plain `Exception` in `TaskDispatcher.invoke`:
    ```267:277:/home/yizhuoliang/cornserve/python/cornserve/services/task_dispatcher/dispatcher.py
    try:
        async with asyncio.TaskGroup() as tg:
            ...
    except ExceptionGroup as eg:
        logger.exception("Error while invoking task")
        raise RuntimeError(f"Task invocation failed: {eg.exceptions}") from eg
    except Exception as e:
        logger.exception("Error while invoking task")
        raise RuntimeError(f"Task invocation failed: {e}") from e
    ```

- K8s client response typing too broad in task registry
  - Symptoms: Many "Cannot access attribute 'get'" and unbound variable reports in `TaskRegistry`.
  - Fixes:
    - Narrow dynamic objects from the Kubernetes API with a cast/annotation before `.get()` usage.
    - Predefine variables used in `except` messages so they are always bound.
    ```181:211:/home/yizhuoliang/cornserve/python/cornserve/services/task_registry/registry.py
    cr_object: dict[str, Any] = await self._custom_api.get_namespaced_custom_object(...)
    spec: dict[str, Any] = cr_object.get("spec", {})
    metadata: dict[str, Any] = cr_object.get("metadata", {})
    ...
    task_cls, _, _ = TASK_CLASS_REGISTRY.get_unit_task(definition_ref)
    task_instance = task_cls.model_validate(config)
    ```
    ```244:306:/home/yizhuoliang/cornserve/python/cornserve/services/task_registry/registry.py
    task_class_name: str | None = None
    module_name: str | None = None
    source_code: str | None = None
    is_unit_task: bool | None = None
    descriptor_class_name: str | None = None
    # then assign inside try and use safe fallbacks in the except logger
    ```

- Type narrowing in `TaskClassRegistry`
  - Symptoms: Union type mismatches when registering tasks and restoring modules.
  - Fixes:
    - Assert/cast `task_cls` prior to `_register` for unit tasks.
    - Guard `previous_module` restore with `is not None`.
    ```140:176:/home/yizhuoliang/cornserve/python/cornserve/services/task_registry/task_class_registry.py
    if is_unit_task:
        assert issubclass(task_cls, UnitTask)
        self._register(task_cls, task_input_cls, task_output_cls, task_class_name)
    ...
    if preexisting and previous_module is not None:
        sys.modules[module_name] = previous_module
    ```

- Optional misuse in `UnitTaskProfile`
  - Symptoms: "Argument of type 'None' cannot be assigned to parameter 'task' of type 'UnitTask'".
  - Fix one:
    - Make `task` optional in the dataclass, or
    - Don’t return a profile with `task=None`; instead, raise or use a separate default method.
    ```78:83:/home/yizhuoliang/cornserve/python/cornserve/task_executors/profile.py
    # Avoid returning cls(task=None, ...). Make task: UnitTask | None or return via get_default_profile.
    ```

- Possible None/typing on shared memory buffers
  - Symptoms: "Object of type 'None' is not subscriptable" around `memoryview(self.shared_memory.buf[...])`.
  - Fixes:
    - Ensure `self.shared_memory` is always initialized or guard its uses when open-by-name fails.
    - Cast the buffer to `memoryview` variable before slicing.
    ```170:175:/home/yizhuoliang/cornserve/python/cornserve/task_executors/eric/distributed/shm_broadcast.py
    buf_mem = self.shared_memory.buf  # ensure not None, then slice or guard
    with memoryview(buf_mem[start:end]) as buf:
        yield buf
    ```

### Minimal config to apply now

Add (or merge) into `python/pyproject.toml`:

```toml
[tool.pyright]
extraPaths = ["../python-tasklib", "../python-tasklib/build/lib"]
exclude = [
  "**/*_pb2.py",
  "**/*_pb2_grpc.py",
  "**/*_pb2.pyi",
  "**/*_pb2_grpc.pyi",
  "cornserve/task_executors/eric/models/*.py",
]
```

Then either install tasklib or keep extraPaths:

```bash
pip install -e /home/yizhuoliang/cornserve/python-tasklib && pip install -e /home/yizhuoliang/cornserve/python
```

After these, remaining type errors should be limited to the few code tweaks listed above.




