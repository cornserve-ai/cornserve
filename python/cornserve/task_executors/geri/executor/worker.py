"""Worker processes for multi-GPU sequence parallel inference in Geri.

Each worker owns one GPU and participates in the SP group. Workers communicate
via NCCL for SP collectives (all-to-all, all-gather). This mirrors Eric's
``executor/worker.py`` but adapted for Geri's generation workload.

Architecture:
- Rank 0 is special: it runs inside the engine process and drives the pipeline.
- Ranks 1..N-1 are spawned in separate processes and enter a command loop,
  waiting for generation parameters broadcast from rank 0 via NCCL.
- All ranks execute the same diffusers pipeline call in lockstep so that
  NCCL collectives inside the patched SP attention layers are properly matched.

Command Protocol:
- Commands are dispatched via NCCL broadcast of a ``torch.long`` tensor.
- The first element is the command type, remaining elements are command-specific.
- For simple commands (GENERATE, SHUTDOWN, PROFILE_START, PROFILE_STOP),
  all parameters fit in the command tensor.
- For future extensibility, ``_CMD_METHOD`` supports arbitrary method dispatch
  via a pickled payload broadcast as a byte tensor.

Error Propagation:
- Each worker has a ZMQ PUSH socket for reporting errors back to the executor.
- On exception during generation, the worker sends a pickled exception via ZMQ.
- The executor polls error sockets after each ``generate()`` call.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import signal
import traceback
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.process import BaseProcess

import psutil
import torch
import zmq

from cornserve.logging import get_logger
from cornserve.task_executors.geri.distributed.parallel import (
    SPGroup,
    destroy_sp_distributed,
    get_sp_group,
    init_sp_distributed,
)
from cornserve.task_executors.geri.models.sp_wrapper import execute_sp_generate, load_and_patch_pipeline_for_sp
from cornserve.task_executors.geri.utils.zmq import get_open_zmq_ipc_path, zmq_sync_socket_ctx

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Command protocol constants
# ---------------------------------------------------------------------------

_CMD_GENERATE = 0
_CMD_SHUTDOWN = 1
_CMD_PROFILE_START = 2
_CMD_PROFILE_STOP = 3
_CMD_METHOD = 4  # Generic method dispatch (future extensibility)

# Fixed command tensor width (pad unused fields with 0)
_CMD_TENSOR_SIZE = 8


@dataclass
class SPWorkerHandle:
    """Handle to a spawned SP worker process, held by the executor.

    Attributes:
        process: The worker process.
        rank: The SP rank of this worker.
        ready_zmq_path: ZMQ socket path used for the initial ready handshake.
        error_zmq_path: ZMQ socket path for receiving error reports from worker.
    """

    process: BaseProcess
    rank: int
    ready_zmq_path: str
    error_zmq_path: str


# ---------------------------------------------------------------------------
# Command broadcast helpers (called by rank 0)
# ---------------------------------------------------------------------------


def _make_cmd_tensor(*values: int) -> torch.Tensor:
    """Create a padded command tensor from the given values."""
    padded = list(values) + [0] * (_CMD_TENSOR_SIZE - len(values))
    return torch.tensor(padded[:_CMD_TENSOR_SIZE], dtype=torch.long, device="cuda")


def broadcast_generate_command(
    sp_group: SPGroup,
    height: int,
    width: int,
    num_inference_steps: int,
    batch_size: int,
    max_seq_len: int,
) -> None:
    """Broadcast a generate command from rank 0 to all SP workers.

    Rank 0 packs generation parameters into a tensor and broadcasts it.

    Args:
        sp_group: The SP group.
        height: Image height.
        width: Image width.
        num_inference_steps: Number of denoising steps.
        batch_size: Batch size.
        max_seq_len: Maximum sequence length of prompt embeddings.
    """
    cmd = _make_cmd_tensor(_CMD_GENERATE, height, width, num_inference_steps, batch_size, max_seq_len)
    sp_group.broadcast(cmd, src=0)


def broadcast_shutdown_command(sp_group: SPGroup) -> None:
    """Broadcast a shutdown command from rank 0 to all SP workers."""
    cmd = _make_cmd_tensor(_CMD_SHUTDOWN)
    sp_group.broadcast(cmd, src=0)


def broadcast_profile_start_command(sp_group: SPGroup, output_dir_encoded: int = 0) -> None:
    """Broadcast a profile start command from rank 0 to all SP workers.

    Args:
        sp_group: The SP group.
        output_dir_encoded: Reserved for future use (e.g., length of a follow-up
            broadcast containing the output directory path). Currently workers
            use a default output directory.
    """
    cmd = _make_cmd_tensor(_CMD_PROFILE_START, output_dir_encoded)
    sp_group.broadcast(cmd, src=0)


def broadcast_profile_stop_command(sp_group: SPGroup) -> None:
    """Broadcast a profile stop command from rank 0 to all SP workers."""
    cmd = _make_cmd_tensor(_CMD_PROFILE_STOP)
    sp_group.broadcast(cmd, src=0)


def receive_command(sp_group: SPGroup) -> torch.Tensor:
    """Receive a command tensor broadcast from rank 0.

    Called by non-rank-0 workers.

    Returns:
        Command tensor of size ``_CMD_TENSOR_SIZE``.
    """
    cmd = torch.zeros(_CMD_TENSOR_SIZE, dtype=torch.long, device="cuda")
    sp_group.broadcast(cmd, src=0)
    return cmd


class SPWorker:
    """Runs SP-parallel model inference on a single GPU (non-rank-0).

    The worker:
    1. Initializes NCCL distributed and the SP group.
    2. Loads the model and patches it for sequence parallelism.
    3. Enters a loop waiting for commands from rank 0.

    Error propagation:
    - On exception, the worker sends the error via a ZMQ PUSH socket
      so the executor can detect and re-raise it.
    """

    def __init__(
        self,
        model_id: str,
        sp_rank: int,
        sp_size: int,
        torch_dtype: torch.dtype,
        init_method: str,
        error_zmq_path: str | None = None,
    ) -> None:
        """Initialize the worker.

        Args:
            model_id: Hugging Face model ID.
            sp_rank: This worker's rank in the SP group.
            sp_size: Total number of SP workers.
            torch_dtype: Data type for model weights.
            init_method: Rendezvous URL for ``torch.distributed``.
            error_zmq_path: ZMQ IPC path for error reporting back to executor.
        """
        self.sp_rank = sp_rank
        self.sp_size = sp_size
        self.error_zmq_path = error_zmq_path

        # Set up error reporting socket (ZMQ PUSH → executor's PULL)
        self._error_zmq_ctx: zmq.Context | None = None
        self._error_sock: zmq.Socket | None = None
        if error_zmq_path:
            self._error_zmq_ctx = zmq.Context(io_threads=1)
            error_sock = self._error_zmq_ctx.socket(zmq.PUSH)
            error_sock.setsockopt(zmq.SNDHWM, 0)
            error_sock.setsockopt(zmq.LINGER, 1000)
            error_sock.connect(error_zmq_path)
            self._error_sock = error_sock

        # Initialize torch.distributed and SP group
        init_sp_distributed(world_size=sp_size, rank=sp_rank, init_method=init_method)

        sp_group = get_sp_group()

        # Load and patch the pipeline for SP
        self.pipeline, self.embedding_dim = load_and_patch_pipeline_for_sp(
            model_id=model_id,
            torch_dtype=torch_dtype,
            torch_device=torch.device("cuda"),
            sp_group=sp_group,
        )

        # Profiler state (mirrors Eric's worker pattern)
        self.profiler_info: tuple[torch.profiler.profile, str] | None = None

        logger.info("SP worker %d: pipeline loaded and patched for SP.", sp_rank)

    def _report_error(self, exc: Exception) -> None:
        """Send an exception to the executor via ZMQ."""
        if self._error_sock is not None:
            try:
                error_data = pickle.dumps(
                    {
                        "rank": self.sp_rank,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                self._error_sock.send(error_data, zmq.NOBLOCK)
            except Exception:
                logger.warning("SP worker %d: failed to send error via ZMQ.", self.sp_rank)

    def run(self) -> None:
        """Main worker loop: wait for commands from rank 0 and execute them."""
        sp_group = get_sp_group()
        logger.info("SP worker %d entering command loop.", self.sp_rank)

        while True:
            cmd = receive_command(sp_group)
            cmd_type = cmd[0].item()

            if cmd_type == _CMD_SHUTDOWN:
                logger.info("SP worker %d received shutdown command.", self.sp_rank)
                break

            elif cmd_type == _CMD_GENERATE:
                height = int(cmd[1].item())
                width = int(cmd[2].item())
                num_inference_steps = int(cmd[3].item())
                batch_size = int(cmd[4].item())
                max_seq_len = int(cmd[5].item())

                logger.info(
                    "SP worker %d executing generate: h=%d, w=%d, steps=%d, batch=%d, max_seq=%d",
                    self.sp_rank,
                    height,
                    width,
                    num_inference_steps,
                    batch_size,
                    max_seq_len,
                )

                try:
                    # Execute SP generate — this participates in collectives with rank 0.
                    # Non-rank-0 workers discard the output.
                    execute_sp_generate(
                        pipeline=self.pipeline,
                        sp_group=sp_group,
                        prompt_embeds=None,  # Will be received via broadcast from rank 0
                        prompt_embeds_mask=None,  # Will be received via broadcast from rank 0
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        batch_size=batch_size,
                        max_seq_len=max_seq_len,
                    )
                except Exception as e:
                    logger.exception("SP worker %d hit an exception during generate.", self.sp_rank)
                    self._report_error(e)
                    raise

            elif cmd_type == _CMD_PROFILE_START:
                self.start_profile()

            elif cmd_type == _CMD_PROFILE_STOP:
                self.stop_profile()

            elif cmd_type == _CMD_METHOD:
                # Generic method dispatch (future extensibility).
                # Receive pickled (method_name, args, kwargs) via a follow-up broadcast.
                logger.warning("SP worker %d received _CMD_METHOD — not yet implemented.", self.sp_rank)

            else:
                logger.warning("SP worker %d received unknown command type %d", self.sp_rank, cmd_type)

    def start_profile(self, output_dir: str = "./profiler_output") -> str:
        """Start PyTorch profiler on this worker.

        Args:
            output_dir: Directory where profiler traces will be saved.

        Returns:
            Path to the trace file that will be created when profiling stops.
        """
        if self.profiler_info is not None:
            logger.warning("Profiler is already running on SP worker %d", self.sp_rank)
            return self.profiler_info[1]

        os.makedirs(output_dir, exist_ok=True)
        profiler_output_path = os.path.join(output_dir, f"sp-worker-rank-{self.sp_rank}-trace.json")

        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        )
        profiler.start()
        logger.info("Profiler started on SP worker %d, output: %s", self.sp_rank, profiler_output_path)

        self.profiler_info = (profiler, profiler_output_path)
        return profiler_output_path

    def stop_profile(self) -> str | None:
        """Stop PyTorch profiler and export chrome trace.

        Returns:
            Path to the exported trace file, or None if profiler was not running.
        """
        if self.profiler_info is None:
            logger.warning("Profiler is not running on SP worker %d", self.sp_rank)
            return None

        profiler, profiler_output_path = self.profiler_info
        profiler.stop()
        profiler.export_chrome_trace(profiler_output_path)
        logger.info("Profiler stopped on SP worker %d, trace: %s", self.sp_rank, profiler_output_path)

        self.profiler_info = None
        return profiler_output_path

    def shutdown(self) -> None:
        """Shutdown the worker and clean up resources."""
        logger.info("Shutting down SP worker %d", self.sp_rank)

        # Stop profiler if running
        if self.profiler_info is not None:
            with suppress(Exception):
                self.stop_profile()

        # Close error socket
        if self._error_sock is not None:
            self._error_sock.close()
        if self._error_zmq_ctx is not None:
            self._error_zmq_ctx.destroy(linger=0)

        destroy_sp_distributed()

    @staticmethod
    def spawn_worker(
        model_id: str,
        sp_rank: int,
        sp_size: int,
        torch_dtype: torch.dtype,
        init_method: str,
    ) -> SPWorkerHandle:
        """Spawn a worker process (does NOT wait for readiness).

        The caller must separately wait for the worker to be ready after
        all workers have been spawned and the distributed group initialized.

        Args:
            model_id: Hugging Face model ID.
            sp_rank: SP rank for this worker.
            sp_size: Total SP world size.
            torch_dtype: Data type for model weights.
            init_method: Rendezvous URL for torch.distributed.

        Returns:
            Handle to the spawned worker (not yet confirmed ready).
        """
        ready_zmq_path = get_open_zmq_ipc_path(f"sp-worker-{sp_rank}-ready")
        error_zmq_path = get_open_zmq_ipc_path(f"sp-worker-{sp_rank}-error")

        context = mp.get_context("spawn")
        worker_proc = context.Process(
            target=SPWorker._main,
            kwargs=dict(
                model_id=model_id,
                sp_rank=sp_rank,
                sp_size=sp_size,
                torch_dtype=torch_dtype,
                init_method=init_method,
                ready_zmq_path=ready_zmq_path,
                error_zmq_path=error_zmq_path,
            ),
            daemon=True,
        )
        worker_proc.start()
        logger.info("SP worker %d spawned with PID %d", sp_rank, worker_proc.pid)

        return SPWorkerHandle(
            process=worker_proc,
            rank=sp_rank,
            ready_zmq_path=ready_zmq_path,
            error_zmq_path=error_zmq_path,
        )

    @staticmethod
    def wait_for_worker_ready(handle: SPWorkerHandle, timeout_ms: int = 300000) -> None:
        """Wait for a spawned worker to signal readiness.

        Args:
            handle: The SPWorkerHandle returned by spawn_worker.
            timeout_ms: Maximum time to wait in milliseconds.
        """
        with zmq_sync_socket_ctx(handle.ready_zmq_path, zmq.PULL) as ready_sock:
            while ready_sock.poll(timeout=min(timeout_ms, 10000)) == 0:
                timeout_ms -= 10000
                if timeout_ms <= 0:
                    raise RuntimeError(f"SP worker {handle.rank} timed out during initialization.")
                logger.debug("Waiting for SP worker %d to be ready", handle.rank)
                if not handle.process.is_alive():
                    raise RuntimeError(f"SP worker {handle.rank} process failed to start.")

            ready_sock.recv()

        logger.info("SP worker %d is ready.", handle.rank)

    @staticmethod
    def _main(
        model_id: str,
        sp_rank: int,
        sp_size: int,
        torch_dtype: torch.dtype,
        init_method: str,
        ready_zmq_path: str,
        error_zmq_path: str,
    ) -> None:
        """Entry point for the spawned worker process."""
        shutdown_requested = False

        def shutdown_handler(*_) -> None:
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        signal.signal(signal.SIGINT, shutdown_handler)
        signal.signal(signal.SIGTERM, shutdown_handler)

        worker: SPWorker | None = None
        parent_process = psutil.Process().parent()
        try:
            worker = SPWorker(
                model_id=model_id,
                sp_rank=sp_rank,
                sp_size=sp_size,
                torch_dtype=torch_dtype,
                init_method=init_method,
                error_zmq_path=error_zmq_path,
            )

            # Signal readiness
            with zmq_sync_socket_ctx(ready_zmq_path, zmq.PUSH) as ready_sock:
                ready_sock.send(b"ready")

            # Enter command loop
            worker.run()

        except SystemExit:
            logger.debug("SP worker %d interrupted by signal.", sp_rank)
        except Exception:
            logger.exception("SP worker %d hit an exception.", sp_rank)
            if parent_process:
                parent_process.send_signal(signal.SIGUSR1)
        finally:
            if worker:
                worker.shutdown()
