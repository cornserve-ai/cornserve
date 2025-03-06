"""Worker processes use the GPUs to run tensor parallel inference."""

import signal
import pickle
import multiprocessing as mp
from dataclasses import dataclass
from multiprocessing.process import BaseProcess

import zmq
import torch
import psutil

from cornserve.task_executors.eric.distributed.shm_broadcast import (
    MessageQueueHandle,
    MessageQueue,
)
from cornserve.task_executors.eric.executor.loader import load_model
from cornserve.task_executors.eric.schema import Batch
from cornserve.task_executors.eric.utils.zmq import (
    get_open_zmq_ipc_path,
    zmq_sync_socket,
)
from cornserve.task_executors.eric.distributed.parallel import (
    init_distributed,
    destroy_distributed,
)

# from cornserve.services.sidecar.api import TensorSidecarSender
from cornserve.logging import get_logger

logger = get_logger(__name__)


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
        tp_rank: int,
        tp_size: int,
        input_mq: MessageQueue,
        response_mq: MessageQueue,
        sender_sidecar_rank: int,
    ) -> None:
        """Initialize the worker."""
        # Cached variables
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.input_mq = input_mq
        self.response_mq = response_mq
        self.sender_sidecar_rank = sender_sidecar_rank

        # Initialize torch.distributed and tensor parallelism
        init_distributed(world_size=tp_size, rank=tp_rank)

        # TP rank is now known. Instantiate the load the model.
        self.model = load_model(model_name_or_path=model_id)

        # Initialize the sender sidecar client
        # self.sender_sidecar_client = TensorSidecarSender(
        #     sidecar_rank=sender_sidecar_rank,
        #     chunk_shape=self.model.chunk_shape,
        #     dtype=self.model.dtype,
        #     shard_rank=tp_rank,
        #     num_shards=tp_size,
        # )

    @staticmethod
    def spawn_worker(
        model_id: str,
        tp_rank: int,
        tp_size: int,
        input_mq_handle: MessageQueueHandle,
        sender_sidecar_rank: int,
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
                tp_rank=tp_rank,
                tp_size=tp_size,
                input_mq_handle=input_mq_handle,
                ready_zmq_path=ready_zmq_path,
                sender_sidecar_rank=sender_sidecar_rank,
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
        tp_rank: int,
        tp_size: int,
        input_mq_handle: MessageQueueHandle,
        ready_zmq_path: str,
        sender_sidecar_rank: int,
    ) -> None:
        """Entrypoint for the worker process when it's spawned.

        This function registers signal handlers and performs exception handling
        for the worker process's main loop.
        """
        # Install signal handlers for graceful termination.
        # Users send SIGNIT, the executor sends SIGTERM.
        shutdown_requested = False

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
                tp_rank=tp_rank,
                tp_size=tp_size,
                input_mq=input_mq,
                response_mq=response_mq,
                sender_sidecar_rank=sender_sidecar_rank,
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
    def execute_model(self, batch: Batch) -> None:
        """Run the model on a batch of data."""
        output = self.model(modality=batch.modality, batch=batch.data)
        logger.info(
            "Worker %d processed batch with request IDs %s and returned a tensor of shape %s",
            self.tp_rank,
            batch.request_ids,
            [t.shape for t in output],
        )

        # Send to sidecar
        for i in range(len(batch.data_ids)):
            if batch.receiver_ranks[i] is None:
                continue
            # self.sender_sidecar_client.send(
            #     chunk=output[i],
            #     id=batch.request_ids[i] + batch.data_ids[i],
            #     chunk_id=batch.chunk_ids[i],
            #     num_chunks=batch.num_chunks[i],
            #     dst_sidecar_ranks=batch.receiver_ranks[i],
            # )

        # Dump tensors for debugging if requested
        if batch._dump_prefix is not None and self.tp_rank == 0:
            torch.save(
                [o.cpu() for o in output],
                batch._dump_prefix + f"-tp{self.tp_size}.pt",
            )
