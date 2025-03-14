"""The model executor manages multiple workers that execute inference."""

import os
import time
import signal
from typing import Any
import psutil
from contextlib import suppress

from cornserve.task_executors.eric.distributed.shm_broadcast import MessageQueue
from cornserve.task_executors.eric.executor.worker import WorkerHandle, Worker
from cornserve.task_executors.eric.schema import Batch, BatchResult, Status
from cornserve.logging import get_logger

logger = get_logger(__name__)


class ModelExecutor:
    """A class to execute a model with multiple workers.

    This class is instaintiated by the engine, and provides a method to
    trigger the execution of the model with multiple workers.

    Initialization:
    1. A shared memory ring buffer (to broadcast inputs to workers) and a
        response ZMQ socket (to receive signals from workers) are created.
    2. Workers are spawned with a handle to the shared memory ring buffer
        and the address to the response ZMQ socket.
    3. Workers initialize the model and loads pretrained weights.
    4. Workers send a READY signal to the executor.

    Executing a batch:
    1. The executor's `execute_model` method is called with a batch of data.
    2. Data is broadcasted to all workers using the shared memory ring buffer.
    3. Workers receive the data and run inference on the model.
    4. Workers send the results to the Tensor Sidecar with a separate thread.
    5. When results are sent, workers send a DONE signal to the executor.
    """

    def __init__(
        self,
        model_id: str,
        tp_size: int,
        sender_sidecar_ranks: list[int],
    ) -> None:
        """Initialize the executor and spawn workers."""

        # Install shutdown signal handler
        def shutdown(*_) -> None:
            logger.critical("Received signal from worker. Shutting down.")
            if (parent := psutil.Process().parent()) and parent.name() == "eric_engine":
                parent.send_signal(signal.SIGUSR1)
            self.shutdown()

        signal.signal(signal.SIGUSR1, shutdown)

        # Message queue for communication between executor and workers
        self.input_mq = MessageQueue(
            tp_size,
            tp_size,
            max_chunk_bytes=1024 * 1024 * 1024,  # 1GB
            max_chunks=2,
        )
        input_mq_handle = self.input_mq.export_handle()

        # Spawn workers
        self.workers: list[WorkerHandle] = []
        for tp_rank in range(tp_size):
            start_time = time.monotonic()
            logger.info("Spawning worker %d", tp_rank)
            worker = Worker.spawn_worker(
                model_id=model_id,
                tp_rank=tp_rank,
                tp_size=tp_size,
                input_mq_handle=input_mq_handle,
                sender_sidecar_rank=sender_sidecar_ranks[tp_rank],
            )
            logger.info(
                "Took %.2f seconds to spawn worker %d",
                time.monotonic() - start_time,
                tp_rank,
            )
            self.workers.append(worker)

        # Wait until the message queues are ready. Order is critical.
        self.input_mq.wait_until_ready()
        for worker in self.workers:
            worker.response_mq.wait_until_ready()

    def shutdown(self) -> None:
        """Ensure workers are terminated and shut down the executor."""
        if hasattr(self, "shutdown_called"):
            return

        self.shutdown_called = True
        logger.info("Shutting down executor.")

        # Close the input message queue
        if hasattr(self, "input_mq"):
            del self.input_mq

        # Ensure workers are terminated
        for worker in self.workers:
            del worker.response_mq

        def wait_for_termination(procs, timeout):
            if not time:
                # If we are in late stage shutdown, the interpreter may replace
                # `time` with `None`.
                return all(not proc.is_alive() for proc in procs)
            start_time = time.time()
            while time.time() - start_time < timeout:
                if all(not proc.is_alive() for proc in procs):
                    return True
                time.sleep(0.1)
            return False

        # Send SIGTERM if still running
        active_procs = [w.process for w in self.workers if w.process.is_alive()]
        for p in active_procs:
            p.terminate()
        if not wait_for_termination(active_procs, 4):
            # Send SIGKILL if still running
            active_procs = [p for p in active_procs if p.is_alive()]
            for p in active_procs:
                p.kill()

        # Clean up ZMQ socket files
        for worker in self.workers:
            with suppress(FileNotFoundError):
                os.remove(worker.ready_zmq_path.replace("ipc://", ""))

    def run_workers(
        self,
        method: str,
        args: tuple | None = None,
        kwargs: dict | None = None,
    ) -> Any:
        """Dispatch a method to all workers and retrieve results."""
        self.input_mq.enqueue((method, args, kwargs))

        responses = []

        for w in self.workers:
            result = w.response_mq.dequeue()
            responses.append(result)

        if any(isinstance(r, Exception) for r in responses):
            logger.error("One or more workers failed with an exception: %s", responses)
            raise RuntimeError("One or more workers failed to process the request.")

        return responses

    def execute_model(self, batch: Batch) -> BatchResult:
        """Invoke the workers to run inference on the model."""
        logger.info("Executing model with %d items", len(batch.data_ids))
        self.run_workers("execute_model", kwargs={"batch": batch})
        return BatchResult(
            request_ids=batch.request_ids,
            data_ids=batch.data_ids,
            chunk_ids=batch.chunk_ids,
            num_chunks=batch.num_chunks,
            receiver_ranks=batch.receiver_ranks,
            status=Status.SUCCESS,
        )
