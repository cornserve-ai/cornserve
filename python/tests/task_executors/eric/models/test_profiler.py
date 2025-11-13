"""Test PyTorch profiler integration with Eric."""

import os

from cornserve.task_executors.eric.executor.executor import ModelExecutor
from cornserve.task_executors.eric.schema import Status

from ..utils import batch_builder, param_tp_size

model_id = "Qwen/Qwen3-VL-4B-Instruct"
model_shorthand = "qwen3_vl"


@param_tp_size
def test_profiler_integration(test_images, tp_size: int, tmp_path) -> None:
    """Test that profiler can be started, used, and stopped."""
    executor = ModelExecutor(
        model_id=model_id,
        adapter_model_ids=[],
        tp_size=tp_size,
        sender_sidecar_ranks=None,
    )

    # Start profiling
    output_dir = str(tmp_path / "profiler_traces")
    trace_paths = executor.start_profile(output_dir=output_dir)

    # Verify trace paths were returned
    assert len(trace_paths) == tp_size
    for i, trace_path in enumerate(trace_paths):
        assert trace_path == os.path.join(output_dir, f"worker-rank-{i}-trace.json")

    # Execute model (this should be profiled)
    result = executor.execute_model(batch=batch_builder(model_id, model_shorthand, test_images[:1]))
    assert result.status == Status.SUCCESS

    # Stop profiling
    saved_paths = executor.stop_profile()

    # Verify traces were saved
    assert len(saved_paths) == tp_size
    for trace_path in saved_paths:
        assert os.path.exists(trace_path), f"Trace file not found: {trace_path}"
        # Verify the file is not empty
        assert os.path.getsize(trace_path) > 0, f"Trace file is empty: {trace_path}"

    executor.shutdown()
