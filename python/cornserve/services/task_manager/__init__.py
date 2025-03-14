"""The Task Manager manages task executors.

A Task Manager handles exactly one type of task, for instance,
multimodal data embedding (Eric) or LLM inference (vLLM).
It spawns and kills task executors given the resource (GPUs) allocated to it by
the resource manager.

It's primarily responsible for
1. Spawning and killing task executors
2. Routing requests to the appropriate task executor
"""
