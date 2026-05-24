"""The model executor manages generation operations.

ModelExecutor hierarchy:
- ``ModelExecutor`` (ABC): Base class, owns a GeriModel and provides generate().
- ``BatchExecutor``: Single-GPU batch generation (image).
- ``StreamExecutor``: Single-GPU streaming generation (audio).
- ``SPBatchExecutor``: Multi-GPU batch generation with Sequence Parallelism.
  Spawns SP worker processes, manages distributed init/shutdown, and delegates
  generation to the SP-patched model.  Mirrors Eric's ``ModelExecutor`` pattern
  where all worker lifecycle management lives in the executor, not the engine.
"""

from __future__ import annotations

import os
import pickle
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import suppress
from typing import Any, Generic, Protocol, TypeVar

import torch
import zmq

from cornserve.logging import get_logger
from cornserve.task_executors.geri.api import Status
from cornserve.task_executors.geri.models.base import (
    BatchGeriModel,
    GeriModel,
    StreamGeriModel,
)
from cornserve.task_executors.geri.schema import (
    BatchGenerationResult,
    StreamGenerationResult,
)

logger = get_logger(__name__)


ModelT = TypeVar("ModelT", bound=GeriModel)


class HasModel(Protocol, Generic[ModelT]):
    """Protocol to enforce that ModelExecutor subclasses initialize a GeriModel field."""

    model: ModelT


class ModelExecutor(HasModel[ModelT], Generic[ModelT], ABC):
    """A class to execute generation with a model.

    This is the base executor class.  Subclasses own the model instance and
    handle the generate() call.  For multi-GPU SP, see :class:`SPBatchExecutor`.
    """

    def shutdown(self) -> None:
        """Shutdown the executor and clean up resources."""
        logger.info("Shutting down ModelExecutor")

        if hasattr(self, "model"):
            del self.model

    @abstractmethod
    def generate(self, *args, **kwargs) -> Any:
        """Execute generation with the model.

        The ModelExecutor base class requires that subclasses implement this method, but
        parameters and return type will vary and therefore left to each subclass to decide.
        """


class BatchExecutor(ModelExecutor[BatchGeriModel]):
    """Executor for batched (i.e., non-streaming) generation requests."""

    def __init__(self, model: BatchGeriModel) -> None:
        """Initialize the batch executor."""
        self.model = model

    @torch.inference_mode()
    def generate(
        self,
        prompt_embeds: list[torch.Tensor],
        height: int,
        width: int,
        num_inference_steps: int,
    ) -> BatchGenerationResult:
        """Execute batched generation with the model.

        Currently, the primary use case for this class is image generation.

        Args:
            prompt_embeds: List of text embeddings from the LLM encoder, one per batch item.
            height: Height of the generated image in pixels.
            width: Width of the generated image in pixels.
            num_inference_steps: Number of denoising steps to perform.

        Returns:
            Generation result containing images or error information.
        """
        try:
            logger.info("Generating content with size %dx%d, %d inference steps", height, width, num_inference_steps)

            generated_bytes = self.model.generate(
                prompt_embeds=prompt_embeds,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
            )

            logger.info("Generation completed successfully, got %d images as PNG bytes", len(generated_bytes))
            return BatchGenerationResult(status=Status.SUCCESS, generated=generated_bytes)

        except Exception as e:
            logger.exception("Generation failed: %s", str(e))
            return BatchGenerationResult(status=Status.ERROR, error_message=f"Generation failed: {str(e)}")


class SPBatchExecutor(ModelExecutor[BatchGeriModel]):
    """Executor for batched generation with Sequence Parallelism (SP).

    This class mirrors Eric's ``ModelExecutor`` pattern: the executor owns the
    full lifecycle of worker processes, distributed init, model loading, and
    SP patching.  The engine simply creates an ``SPBatchExecutor`` and calls
    ``generate()`` — it does not need to know about workers or NCCL.

    Initialization:
    1. SP worker processes (ranks 1..sp_size-1) are spawned.
    2. Rank 0's ``torch.distributed`` is initialized, which unblocks workers.
    3. Workers load the model, patch it for SP, and signal readiness via ZMQ.
    4. Rank 0 loads its own model and patches it for SP.

    Executing a batch:
    1. The executor's ``generate()`` method is called.
    2. It delegates to ``model.generate()`` which internally broadcasts
       generation params to workers via NCCL broadcast and runs the denoising
       loop in lockstep across all ranks.
    3. After generation, the executor checks the ZMQ error channel for any
       worker errors.

    Shutdown:
    1. A shutdown command is broadcast to workers via NCCL.
    2. Worker processes are terminated and ZMQ sockets cleaned up.
    3. The distributed process group is destroyed.
    """

    def __init__(
        self,
        model_id: str,
        sp_size: int,
        torch_dtype: torch.dtype,
        torch_device: torch.device,
        registry_entry: Any,
        model_config: Any | None = None,
    ) -> None:
        """Initialize the SP executor, spawn workers, and load the model.

        Args:
            model_id: Hugging Face model ID.
            sp_size: Number of GPUs for sequence parallelism.
            torch_dtype: Data type for model weights.
            torch_device: Device for rank 0's model.
            registry_entry: Geri model registry entry.
            model_config: Optional HF model config.
        """
        from cornserve.task_executors.eric.utils.network import get_open_port
        from cornserve.task_executors.geri.distributed.parallel import (
            get_sp_group,
            init_sp_distributed,
        )
        from cornserve.task_executors.geri.executor.loader import load_model
        from cornserve.task_executors.geri.executor.worker import SPWorker, SPWorkerHandle

        self.sp_size = sp_size
        self.sp_workers: list[SPWorkerHandle] = []

        # ZMQ error polling infrastructure
        self._error_zmq_ctx: zmq.Context | None = None
        self._error_sockets: list[zmq.Socket] = []

        init_method = f"tcp://127.0.0.1:{get_open_port()}"
        logger.info("SPBatchExecutor: initializing SP with %d GPUs (init_method=%s)", sp_size, init_method)

        # Phase 1: Spawn all worker processes.
        # They will block on init_process_group until rank 0 also calls init_sp_distributed.
        for rank in range(1, sp_size):
            worker_handle = SPWorker.spawn_worker(
                model_id=model_id,
                sp_rank=rank,
                sp_size=sp_size,
                torch_dtype=torch_dtype,
                init_method=init_method,
            )
            self.sp_workers.append(worker_handle)

        # Phase 2: Initialize rank 0's distributed (unblocks all workers).
        init_sp_distributed(world_size=sp_size, rank=0, init_method=init_method)

        # Phase 3: Wait for all workers to finish initialization and model loading.
        for worker_handle in self.sp_workers:
            SPWorker.wait_for_worker_ready(worker_handle)

        # Phase 3.5: Set up ZMQ error PULL sockets (bind to worker error paths).
        self._error_zmq_ctx = zmq.Context(io_threads=1)
        for worker_handle in self.sp_workers:
            sock = self._error_zmq_ctx.socket(zmq.PULL)
            sock.setsockopt(zmq.RCVHWM, 0)
            sock.setsockopt(zmq.LINGER, 0)
            sock.bind(worker_handle.error_zmq_path)
            self._error_sockets.append(sock)

        # Phase 4: Load model on rank 0 and patch for SP.
        model = load_model(
            model_id=model_id,
            torch_device=torch_device,
            registry_entry=registry_entry,
            config=model_config,
        )

        if not isinstance(model, BatchGeriModel):
            raise TypeError(
                f"SPBatchExecutor requires a BatchGeriModel, got {type(model).__name__}"
            )

        sp_group = get_sp_group()
        if hasattr(model, "patch_for_sp"):
            model.patch_for_sp(sp_group)

        self.model = model

        # Profiler state for rank 0 (mirrors Eric's executor pattern)
        self._profiler_info: tuple[torch.profiler.profile, str] | None = None

        logger.info("SPBatchExecutor: initialization complete.")

    def _check_worker_errors(self) -> None:
        """Poll ZMQ error sockets for any worker errors.

        If a worker reported an error, raise it as a RuntimeError.
        Non-blocking: returns immediately if no errors.
        """
        for i, sock in enumerate(self._error_sockets):
            if sock.poll(timeout=0):
                try:
                    error_data = pickle.loads(sock.recv(zmq.NOBLOCK))
                    rank = error_data.get("rank", "?")
                    error_msg = error_data.get("error", "Unknown error")
                    tb = error_data.get("traceback", "")
                    raise RuntimeError(
                        f"SP worker rank {rank} reported error: {error_msg}\n{tb}"
                    )
                except zmq.Again:
                    pass

    @torch.inference_mode()
    def generate(
        self,
        prompt_embeds: list[torch.Tensor],
        height: int,
        width: int,
        num_inference_steps: int,
    ) -> BatchGenerationResult:
        """Execute SP-parallel batched generation.

        The model's ``generate()`` internally broadcasts params and latents
        to SP workers and runs denoising in lockstep.
        After generation, checks for any worker errors via ZMQ.

        Args:
            prompt_embeds: List of text embeddings from the LLM encoder.
            height: Height of the generated image in pixels.
            width: Width of the generated image in pixels.
            num_inference_steps: Number of denoising steps to perform.

        Returns:
            Generation result containing images or error information.
        """
        try:
            logger.info(
                "SP generating content with size %dx%d, %d inference steps, sp_size=%d",
                height, width, num_inference_steps, self.sp_size,
            )

            generated_bytes = self.model.generate(
                prompt_embeds=prompt_embeds,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
            )

            # Check for worker errors after generation
            self._check_worker_errors()

            logger.info("SP generation completed successfully, got %d images as PNG bytes", len(generated_bytes))
            return BatchGenerationResult(status=Status.SUCCESS, generated=generated_bytes)

        except Exception as e:
            logger.exception("SP generation failed: %s", str(e))
            return BatchGenerationResult(status=Status.ERROR, error_message=f"Generation failed: {str(e)}")

    def start_profile(self, output_dir: str = "./profiler_output") -> list[str]:
        """Start PyTorch profiler on rank 0 and all SP workers.

        Args:
            output_dir: Directory where profiler traces will be saved.

        Returns:
            List of trace file paths (rank 0 + workers).
        """
        from cornserve.task_executors.geri.distributed.parallel import get_sp_group
        from cornserve.task_executors.geri.executor.worker import broadcast_profile_start_command

        os.makedirs(output_dir, exist_ok=True)
        trace_paths: list[str] = []

        # Start profiler on rank 0
        rank0_trace_path = os.path.join(output_dir, "sp-worker-rank-0-trace.json")
        if self._profiler_info is not None:
            logger.warning("Profiler is already running on rank 0")
        else:
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                with_stack=True,
            )
            profiler.start()
            self._profiler_info = (profiler, rank0_trace_path)
            logger.info("Profiler started on rank 0, output: %s", rank0_trace_path)
        trace_paths.append(rank0_trace_path)

        # Broadcast profile start to workers
        sp_group = get_sp_group()
        broadcast_profile_start_command(sp_group)

        # Workers save traces with their rank in the default output dir
        for worker_handle in self.sp_workers:
            trace_paths.append(os.path.join(output_dir, f"sp-worker-rank-{worker_handle.rank}-trace.json"))

        logger.info("Profiler started on all %d SP ranks, traces: %s", self.sp_size, trace_paths)
        return trace_paths

    def stop_profile(self) -> list[str]:
        """Stop PyTorch profiler on rank 0 and all SP workers.

        Returns:
            List of trace file paths saved.
        """
        from cornserve.task_executors.geri.distributed.parallel import get_sp_group
        from cornserve.task_executors.geri.executor.worker import broadcast_profile_stop_command

        trace_paths: list[str] = []

        # Stop profiler on rank 0
        if self._profiler_info is not None:
            profiler, rank0_trace_path = self._profiler_info
            profiler.stop()
            profiler.export_chrome_trace(rank0_trace_path)
            self._profiler_info = None
            trace_paths.append(rank0_trace_path)
            logger.info("Profiler stopped on rank 0, trace: %s", rank0_trace_path)
        else:
            logger.warning("Profiler was not running on rank 0")

        # Broadcast profile stop to workers
        sp_group = get_sp_group()
        broadcast_profile_stop_command(sp_group)

        for worker_handle in self.sp_workers:
            trace_paths.append(f"./profiler_output/sp-worker-rank-{worker_handle.rank}-trace.json")

        logger.info("Profiler stopped on all SP ranks, traces: %s", trace_paths)
        return trace_paths

    def shutdown(self) -> None:
        """Shutdown SP workers and clean up distributed resources."""
        logger.info("Shutting down SPBatchExecutor (sp_size=%d)", self.sp_size)

        # Stop profiler if running
        if self._profiler_info is not None:
            with suppress(Exception):
                profiler, _ = self._profiler_info
                profiler.stop()
                self._profiler_info = None

        # Send shutdown command to workers via NCCL broadcast
        try:
            from cornserve.task_executors.geri.distributed.parallel import get_sp_group
            from cornserve.task_executors.geri.executor.worker import broadcast_shutdown_command

            sp_group = get_sp_group()
            broadcast_shutdown_command(sp_group)
        except Exception:
            logger.warning("Failed to send SP shutdown command via NCCL; terminating workers directly.")

        # Wait for workers to terminate, then force-kill if needed
        for worker_handle in self.sp_workers:
            proc = worker_handle.process
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2)

            # Clean up ZMQ socket files
            with suppress(FileNotFoundError):
                os.remove(worker_handle.ready_zmq_path.replace("ipc://", ""))
            with suppress(FileNotFoundError):
                os.remove(worker_handle.error_zmq_path.replace("ipc://", ""))

        # Clean up error ZMQ sockets
        for sock in self._error_sockets:
            sock.close()
        self._error_sockets.clear()
        if self._error_zmq_ctx is not None:
            self._error_zmq_ctx.destroy(linger=0)
            self._error_zmq_ctx = None

        # Destroy distributed process group
        try:
            from cornserve.task_executors.geri.distributed.parallel import destroy_sp_distributed

            destroy_sp_distributed()
        except Exception:
            logger.debug("Failed to destroy SP distributed group (may already be destroyed).")

        # Clean up model
        super().shutdown()

        logger.info("SPBatchExecutor shut down.")


class StreamExecutor(ModelExecutor[StreamGeriModel]):
    """Executor for streamed generation requests."""

    def __init__(self, model: StreamGeriModel) -> None:
        """Initialize the batch executor."""
        self.model = model

    def generate(
        self,
        prompt_embeds: list[torch.Tensor],
        chunk_size: int | None,
        left_context_size: int | None,
    ) -> StreamGenerationResult:
        """Execute streamed generation with the model.

        Currently, the primary use case for this class is audio generation.

        Args:
            prompt_embeds: List of text embeddings from the LLM encoder, one per batch item.
            chunk_size: number of codes to be processed at a time
            left_context_size: number of codes immediately prior to each chunk to be processed as context

        Returns:
            Result holding a generator that will iteratively yield results as they become ready.
        """
        try:
            logger.info("Beginning streamed generation")

            streamed_generator: Generator[list[torch.Tensor | None], None, None] = self.model.generate(
                prompt_embeds, chunk_size, left_context_size
            )

            logger.info("Obtained generator object")
            return StreamGenerationResult(status=Status.SUCCESS, generator=streamed_generator)

        except Exception as e:
            logger.exception("Generation failed: %s", str(e))
            return StreamGenerationResult(status=Status.ERROR, error_message=f"Generation failed: {str(e)}")
