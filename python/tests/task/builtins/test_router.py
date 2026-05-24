"""Tests for weighted composite routing task."""

from __future__ import annotations

from collections import Counter
from typing import Any, cast

import pytest
from cornserve_tasklib.task.composite.llm import MLLMTask
from cornserve_tasklib.task.composite.router import RouterApp
from cornserve_tasklib.task.unit.encoder import EncoderTask, Modality
from cornserve_tasklib.task.unit.llm import LLMUnitTask
from pydantic import PrivateAttr

from cornserve.task.base import Task, TaskInput, TaskOutput, discover_unit_tasks


class RouterInput(TaskInput):
    """Simple input with deterministic routing key."""

    request_id: str
    payload: str = ""


class RouterOutput(TaskOutput):
    """Simple output carrying selected route label."""

    target: str


class EchoTask(Task[RouterInput, RouterOutput]):
    """Minimal task that returns a static route label."""

    target_name: str
    _request_ids: list[str] = PrivateAttr(default_factory=list)

    @property
    def call_count(self) -> int:
        """Total number of invocations observed by this task."""
        return len(self._request_ids)

    def invoke(self, task_input: RouterInput) -> RouterOutput:
        """Return the configured route label."""
        self._request_ids.append(task_input.request_id)
        return RouterOutput(target=self.target_name)


def test_router_validates_routing_config() -> None:
    """Router validates task/weight cardinality and values."""
    task_a = EchoTask(target_name="a")
    task_b = EchoTask(target_name="b")

    with pytest.raises(
        ValueError,
        match="routing_weights must have the same length as routing_tasks",
    ):
        RouterApp[RouterInput, RouterOutput](
            routing_tasks=[task_a, task_b],
            routing_weights=[1.0],
        )

    with pytest.raises(ValueError, match="routing_tasks must contain at least one task"):
        RouterApp[RouterInput, RouterOutput](
            routing_tasks=[],
            routing_weights=[],
        )

    with pytest.raises(ValueError, match="routing_weights must be non-negative"):
        RouterApp[RouterInput, RouterOutput](
            routing_tasks=[task_a, task_b],
            routing_weights=[1.0, -0.1],
        )

    with pytest.raises(ValueError, match="routing_weights must sum to a positive value"):
        RouterApp[RouterInput, RouterOutput](
            routing_tasks=[task_a, task_b],
            routing_weights=[0.0, 0.0],
        )


def test_router_routes_deterministically_by_request_id() -> None:
    """Same request_id routes to same task and follows weights globally."""
    router = RouterApp[RouterInput, RouterOutput](
        routing_tasks=[EchoTask(target_name="a"), EchoTask(target_name="b")],
        routing_weights=[0.2, 0.8],
    )

    out1 = router.invoke(RouterInput(request_id="stable-id"))
    out2 = router.invoke(RouterInput(request_id="stable-id"))
    assert out1.target == out2.target

    counts: Counter[str] = Counter()
    for i in range(300):
        out = router.invoke(RouterInput(request_id=f"req-{i}"))
        counts[out.target] += 1

    assert counts["b"] > counts["a"]


def test_router_discover_unit_tasks_with_mixed_mllm_configs() -> None:
    """Subtask discovery includes unit tasks from all routed MLLM configs."""
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

    task_tp2 = MLLMTask(
        model_id=model_id,
        modalities=[Modality.IMAGE],
        eric_max_batch_size=2,
        llm_tp_size=2,
        llm_max_num_seqs=32,
        llm_gpu_memory_utilization=0.9,
    )
    task_tp4 = MLLMTask(
        model_id=model_id,
        modalities=[Modality.IMAGE],
        eric_max_batch_size=2,
        llm_tp_size=4,
        llm_max_num_seqs=64,
        llm_gpu_memory_utilization=0.9,
    )

    router = RouterApp(
        routing_tasks=[task_tp2, task_tp4],
        routing_weights=[0.5, 0.5],
    )

    discovered = discover_unit_tasks([router])

    llm_tasks = [task for task in discovered if isinstance(task, LLMUnitTask)]
    assert len(llm_tasks) == 2
    assert sorted((task.tp_size, task.max_num_seqs) for task in llm_tasks) == [
        (2, 32),
        (4, 64),
    ]

    encoder_tasks = [task for task in discovered if isinstance(task, EncoderTask)]
    assert len(encoder_tasks) == 2


@pytest.mark.asyncio
async def test_router_routes_same_index_in_record_and_replay() -> None:
    """Top-level `__call__` keeps one deterministic route for record/replay."""
    task_a = EchoTask(target_name="a")
    task_b = EchoTask(target_name="b")
    router = RouterApp[RouterInput, RouterOutput](
        routing_tasks=[task_a, task_b],
        routing_weights=[0.3, 0.7],
    )

    output = await router(RouterInput(request_id="record-replay-stable-id"))
    assert output.target in {"a", "b"}

    # The chosen route is invoked during both record and replay phases.
    assert sorted([task_a.call_count, task_b.call_count]) == [0, 2]

    # Router counters update once per top-level request, not once per phase.
    counts, total, percentages = cast(Any, router).get_routing_stats()
    assert total == 1
    assert sum(counts) == 1
    assert pytest.approx(sum(percentages), abs=1e-6) == 100.0
