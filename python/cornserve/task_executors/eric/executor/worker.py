"""Worker processes use the GPUs to run tensor parallel inference."""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import signal
from dataclasses import dataclass
from multiprocessing.process import BaseProcess

import psutil
import torch
import zmq
from opentelemetry import context as context_api
from opentelemetry import propagate, trace

from cornserve.logging import get_logger
from cornserve.sidecar.api import Sidecar
from cornserve.sidecar.schema import SidecarConfig
from cornserve.task_executors.eric.distributed.parallel import (
    destroy_distributed,
    init_distributed,
)
from cornserve.task_executors.eric.distributed.shm_broadcast import (
    MessageQueue,
    MessageQueueHandle,
)
from cornserve.task_executors.eric.executor.loader import load_model
from cornserve.task_executors.eric.schema import WorkerBatch
from cornserve.task_executors.eric.utils.zmq import (
    get_open_zmq_ipc_path,
    zmq_sync_socket,
)
from cornserve.tracing import configure_otel

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)
propagator = propagate.get_global_textmap()


@dataclass
class WorkerHandle:
    """Handle to a worker process held by the executor.

    Attributes:
        process: The worker process.
        rank: The rank of the worker.
        ready_zmq_path: ZMQ socket path to receive the worker's ready signal.
        response_mq: The response message queue.
    """

    process: BaseProcess
    rank: int
    ready_zmq_path: str
    response_mq: MessageQueue


class Worker:
    """Runs model inference."""

    def __init__(
        self,
        model_id: str,
        adapter_model_ids: list[str],
        tp_rank: int,
        tp_size: int,
        input_mq: MessageQueue,
        response_mq: MessageQueue,
        sender_sidecar_ranks: list[int] | None,
    ) -> None:
        """Initialize the worker."""
        # Cached variables
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.input_mq = input_mq
        self.response_mq = response_mq
        self.sender_sidecar_ranks = sender_sidecar_ranks

        # Initialize torch.distributed and tensor parallelism
        init_distributed(world_size=tp_size, rank=tp_rank)

        # TP rank is now known. Instantiate the load the model.
        self.model = load_model(
            model_name_or_path=model_id,
            adapter_model_names_or_paths=adapter_model_ids,
        )

        # Initialize the sender sidecar client
        if sender_sidecar_ranks:
            self.sender_sidecar_client = Sidecar(
                config=SidecarConfig(
                    sidecar_rank=sender_sidecar_ranks[tp_rank],
                    group=sender_sidecar_ranks,
                    send_tensor_dtype=self.model.dtype,
                    send_tensor_shape=(-1, *self.model.chunk_shape),
                )
            )
        else:
            self.sender_sidecar_client = None

        # Initialize profiler as None (will be created when start_profile is called)
        self.profiler_info: tuple[torch.profiler.profile, str] | None = None

    @staticmethod
    def spawn_worker(
        model_id: str,
        adapter_model_ids: list[str],
        tp_rank: int,
        tp_size: int,
        input_mq_handle: MessageQueueHandle,
        sender_sidecar_ranks: list[int] | None,
    ) -> WorkerHandle:
        """Spawn the worker process.

        Called by the executor. We're not inside the worker process yet!

        This function spawns a single worker in a separate process and
        waits for it to be ready. When the worker's ready, it returns a
        message queue that the worker pushes responses to.
        """
        # ZMQ socket path for the worker process to send a ready signal
        ready_zmq_path = get_open_zmq_ipc_path(f"worker-{tp_rank}-ready")

        # Spawn the worker process
        context = mp.get_context("spawn")
        worker_proc = context.Process(
            target=Worker.main,
            kwargs=dict(
                model_id=model_id,
                adapter_model_ids=adapter_model_ids,
                tp_rank=tp_rank,
                tp_size=tp_size,
                input_mq_handle=input_mq_handle,
                ready_zmq_path=ready_zmq_path,
                sender_sidecar_ranks=sender_sidecar_ranks,
            ),
            daemon=True,
        )
        worker_proc.start()
        logger.info("Worker %d spawned with PID %d", tp_rank, worker_proc.pid)

        # Wait for the worker to be ready and return MessageQueueHandle
        with zmq_sync_socket(ready_zmq_path, zmq.PULL) as ready_sock:
            while ready_sock.poll(timeout=5000) == 0:
                logger.debug("Waiting for worker %d to be ready", tp_rank)

                if not worker_proc.is_alive():
                    raise RuntimeError(f"Worker {tp_rank} process failed to start.")

            handle_frame = ready_sock.recv(copy=False)
            mq_handle = pickle.loads(handle_frame.buffer)

        response_mq = MessageQueue.create_from_handle(mq_handle, 0)

        return WorkerHandle(
            process=worker_proc,
            rank=tp_rank,
            ready_zmq_path=ready_zmq_path,
            response_mq=response_mq,
        )

    @staticmethod
    def main(
        model_id: str,
        adapter_model_ids: list[str],
        tp_rank: int,
        tp_size: int,
        input_mq_handle: MessageQueueHandle,
        ready_zmq_path: str,
        sender_sidecar_ranks: list[int] | None,
    ) -> None:
        """Entrypoint for the worker process when it's spawned.

        This function registers signal handlers and performs exception handling
        for the worker process's main loop.
        """
        # Install signal handlers for graceful termination.
        # Users send SIGNIT, the executor sends SIGTERM.
        shutdown_requested = False

        configure_otel(f"worker[{tp_rank}]")

        def shutdown(*_) -> None:
            """Idempotently shutdown the worker process."""
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        worker: Worker | None = None
        parent_process = psutil.Process().parent()
        try:
            # Prepare message queues
            input_mq = MessageQueue.create_from_handle(input_mq_handle, tp_rank)
            response_mq = MessageQueue(1, 1)

            # Worker is ready, so signal the executor
            with zmq_sync_socket(ready_zmq_path, zmq.PUSH) as ready_sock:
                # Send the response message queue handle to the executor
                # as a signal of readiness
                handle_frame = pickle.dumps(
                    response_mq.export_handle(),
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                ready_sock.send(handle_frame)

            # Instantiate the worker, which will initialize tensor parallelism
            # and load the model
            worker = Worker(
                model_id=model_id,
                adapter_model_ids=adapter_model_ids,
                tp_rank=tp_rank,
                tp_size=tp_size,
                input_mq=input_mq,
                response_mq=response_mq,
                sender_sidecar_ranks=sender_sidecar_ranks,
            )

            # Wait until the message queues are ready. Order is critical.
            input_mq.wait_until_ready()
            response_mq.wait_until_ready()

            # Run the worker loop
            worker.run()

        except SystemExit:
            logger.debug("Worker interrupted by signal.")
        except Exception:
            logger.exception("Worker hit an exception.")
            if parent_process:
                parent_process.send_signal(signal.SIGUSR1)
        finally:
            if worker:
                worker.shutdown()

    def shutdown(self) -> None:
        """Shutdown the worker process."""
        logger.info("Shutting down worker %d", self.tp_rank)
        del self.input_mq
        del self.response_mq
        destroy_distributed()

    def run(self) -> None:
        """Main worker loop.

        Wait for batches of data from the executor, run the model, and send
        results back to the executor.
        """
        logger.info("Worker %d ready to work", self.tp_rank)
        while True:
            # Wait for the executor to invoke the worker
            method_name, args, kwargs = self.input_mq.dequeue()
            args = args or []
            kwargs = kwargs or {}

            # Execute the method
            try:
                assert isinstance(method_name, str)
                method = getattr(self, method_name)
                output = method(*args, **kwargs)
                self.response_mq.enqueue(output)
            except Exception as e:
                logger.info(
                    "Worker %d hit an exception while running %s(args=%s, kwargs=%s): %s",
                    self.tp_rank,
                    method_name,
                    args,
                    kwargs,
                    e,
                )
                output = e
                self.response_mq.enqueue(output)
                raise

    @torch.inference_mode()
    def execute_model(self, batch: WorkerBatch) -> None:
        """Run the model on a batch of data."""
        unique_spans = {}
        for request_id, carrier in zip(batch.request_ids, batch.otel_carriers, strict=True):
            if carrier and request_id not in unique_spans:
                context = propagator.extract(carrier)
                worker_span = tracer.start_span("Worker.execute_model", context=context)
                worker_span.set_attribute(
                    "worker.execute_model.batch_size",
                    len(batch),
                )
                worker_span.add_event("model_foward.start")
                unique_spans[request_id] = worker_span

        # Run model forward pass (with profiler step if profiling is active)
        if self.profiler_info is not None:
            with torch.profiler.record_function("model_forward"):
                output = self.model(
                    modality=batch.modality,
                    adapter_name=batch.adapter_name,
                    batch=batch.data,
                )
            self.profiler_info[0].step()
        else:
            output = self.model(
                modality=batch.modality,
                adapter_name=batch.adapter_name,
                batch=batch.data,
            )
        for worker_span in unique_spans.values():
            worker_span.add_event("model_forward.done")

        logger.info(
            "Worker %d processed batch with request IDs %s and returned a tensor of shape %s",
            self.tp_rank,
            batch.request_ids,
            [t.shape for t in output],
        )

        # Send to sidecar
        if self.sender_sidecar_client is not None:
            for i, request_id in enumerate(batch.request_ids):
                if (dst_sidecar_ranks := batch.receiver_ranks[i]) is None:
                    continue
                token = None
                if request_id in unique_spans:
                    context = trace.set_span_in_context(unique_spans[request_id])
                    token = context_api.attach(context)
                self.sender_sidecar_client.send(
                    data=output[i],
                    id=batch.data_ids[i],
                    chunk_id=batch.chunk_ids[i],
                    num_chunks=batch.num_chunks[i],
                    dst_sidecar_ranks=dst_sidecar_ranks,
                )
                if token:
                    context_api.detach(token)
                    unique_spans[request_id].end()

        # Dump tensors for debugging if requested
        if batch._dump_prefix is not None and self.tp_rank == 0:
            torch.save(
                [o.cpu() for o in output],
                batch._dump_prefix + f"-tp{self.tp_size}.pt",
            )

    def start_profile(self, output_dir: str = "./profiler_output") -> str:
        """Start PyTorch profiler.

        Args:
            output_dir: Directory where profiler traces will be saved.

        Returns:
            Path to the trace file that will be created when profiling stops.
        """
        if self.profiler_info is not None:
            logger.warning("Profiler is already running on worker %d", self.tp_rank)
            return self.profiler_info[1]

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Set up the output path for this worker
        profiler_output_path = os.path.join(output_dir, f"worker-rank-{self.tp_rank}-trace.json")

        # Create the profiler
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=True,
        )

        # Start profiling
        profiler.__enter__()
        logger.info("Profiler started on worker %d, output will be saved to %s", self.tp_rank, profiler_output_path)

        self.profiler_info = (profiler, profiler_output_path)

        return profiler_output_path

    def stop_profile(self) -> str | None:
        """Stop PyTorch profiler and export chrome trace.

        Returns:
            Path to the exported trace file, or None if profiler was not running.
        """
        if self.profiler_info is None:
            logger.warning("Profiler is not running on worker %d", self.tp_rank)
            return None

        profiler, profiler_output_path = self.profiler_info

        # Stop the profiler
        profiler.__exit__(None, None, None)

        # Export the chrome trace
        profiler.export_chrome_trace(profiler_output_path)
        logger.info("Profiler stopped on worker %d, trace saved to %s", self.tp_rank, profiler_output_path)

        # Clean up
        self.profiler_info = None

        return profiler_output_path
