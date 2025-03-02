"""Eric engine core."""

import gc
import queue
import signal
import threading
import multiprocessing as mp
from typing import Any
from multiprocessing.process import BaseProcess
from multiprocessing.connection import Connection

import psutil
import zmq

from cornserve.task_executors.eric.config import EricConfig
from cornserve.task_executors.eric.utils.zmq import zmq_sync_socket
from cornserve.task_executors.eric.utils.serde import MsgpackEncoder, MsgpackDecoder
from cornserve.task_executors.eric.executor.executor import ModelExecutor
from cornserve.task_executors.eric.engine.scheduler import Scheduler
from cornserve.task_executors.eric.schema import (
    EngineOpcode,
    EngineRequest,
    EngineResponse,
)
from cornserve.logging import get_logger

logger = get_logger(__name__)


class Engine:
    """Eric core engine.

    The engine receives modality embedding requests from the router and
    invokes the model executor to launch embedding computation. When tenosrs
    are sent to the sidecar, the engine sends a message to the router to
    signal completion.
    """

    def __init__(
        self,
        config: EricConfig,
        request_sock_path: str,
        response_sock_path: str,
    ) -> None:
        """Initialize the engine."""
        self.config = config

        self.executor = ModelExecutor(
            model_id=config.model.id,
            modality=config.modality.ty,
            tp_size=config.model.tp_size,
        )

        self.scheduler = Scheduler()

        # Background thread that continuously receives from the request
        # ZMQ socket and pushes it into the request queue
        self.request_queue: queue.Queue[tuple[EngineOpcode, Any]] = queue.Queue()
        threading.Thread(
            target=self._request_receive_loop,
            kwargs=dict(sock_path=request_sock_path),
            daemon=True,
        ).start()

        # Background thread that continuously pulls from the response
        # queue and sends it to the router via the response ZMQ socket
        self.response_queue: queue.Queue[EngineResponse] = queue.Queue()
        threading.Thread(
            target=self._response_send_loop,
            kwargs=dict(sock_path=response_sock_path),
            daemon=True,
        ).start()

    @staticmethod
    def spawn_engine(
        config: EricConfig,
        request_sock_path: str,
        response_sock_path: str,
    ) -> BaseProcess:
        """Spawn the engine process.

        Called by the engine client. We're not inside the engine process yet!

        This function spawns the engine in a separate process and
        waits for it to be ready by blocking on a pipe.
        """
        context = mp.get_context("spawn")
        reader, writer = context.Pipe(duplex=False)
        ready_message = b"ready"
        engine_proc = context.Process(
            # The Executor's shutdown handler depends on this name to
            # identify the engine process and send SIGUSR1 to it.
            name="eric_engine",
            target=Engine.main,
            kwargs=dict(
                config=config,
                request_sock_path=request_sock_path,
                response_sock_path=response_sock_path,
                ready_pipe=writer,
                ready_message=ready_message,
            ),
        )
        engine_proc.start()
        if reader.recv() != ready_message:
            raise RuntimeError("Engine process failed to start")

        reader.close()
        writer.close()

        return engine_proc

    @staticmethod
    def main(
        config: EricConfig,
        request_sock_path: str,
        response_sock_path: str,
        ready_pipe: Connection,
        ready_message: bytes,
    ) -> None:
        """Entrypoint for the engine process when it's spawned.

        This function registers signal handlers and performs exception handling
        around the engine's main loop.
        """
        # Install signal handlers for graceful shutdown.
        # Users send SIGINT, the engine client sends SIGTERM.
        shutdown_requested = False

        def shutdown(*_) -> None:
            """Idempotently shutdown the engine process."""
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        # Start the engine process.
        engine: Engine | None = None
        parent_process = psutil.Process().parent()
        try:
            engine = Engine(
                config=config,
                request_sock_path=request_sock_path,
                response_sock_path=response_sock_path,
            )
            # Send the ready message back to the engine client.
            ready_pipe.send(ready_message)
            ready_pipe.close()

            # Run the engine loop.
            engine.run()

        except SystemExit:
            logger.debug("Engine interrupted by signal.")
        except Exception:
            logger.exception("Engine hit an exception.")
            if parent_process:
                parent_process.send_signal(signal.SIGUSR1)
        finally:
            if engine:
                engine.shutdown()

    def shutdown(self) -> None:
        """Shutdown the engine process."""
        logger.info("Shutting down engine.")
        self.executor.shutdown()

    def run(self):
        """Main engine loop."""
        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # Poll the input queue until there is work to do.
            if not self.scheduler.has_waiting_requests():
                while True:
                    try:
                        req = self.request_queue.get(timeout=3.0)
                        self._handle_client_request(*req)
                        break
                    except queue.Empty:
                        logger.debug("EngineCore busy loop waiting.")
                    except BaseException:
                        raise

            # Handle any new client requests that arrived during the wait.
            while not self.request_queue.empty():
                req = self.request_queue.get_nowait()
                self._handle_client_request(*req)

            # Step the engine core.
            responses = self.step()

            # Put EngineCoreOutputs in the response queue.
            self.response_queue.put_nowait(responses)

    def step(self) -> EngineResponse:
        """Step the engine core.

        This function is called in a loop to process requests and send
        responses. It handles scheduling, executing, and processing results.
        """
        batch = self.scheduler.schedule()
        batch_result = self.executor.execute_model(batch)

        return EngineResponse(
            request_ids=batch_result.request_ids,
            status=batch_result.status,
            error_message=batch_result.error_message,
        )

    def _handle_client_request(self, opcode: EngineOpcode, request: Any) -> None:
        """Dispatch request from client."""
        match opcode:
            case EngineOpcode.ENQUEUE:
                logger.info("Adding request: %s", request.request_id)
                self.scheduler.enqueue(request)
            case EngineOpcode.PROFILE:
                self.should_profile = True
                raise NotImplementedError(f"Opcode {opcode} not implemented.")

    def _request_receive_loop(self, sock_path: str) -> None:
        """Continuously receive requests from a ZMQ socket and enqueue them."""
        new_request_decoder = MsgpackDecoder(ty=EngineRequest)
        generic_decoder = MsgpackDecoder()

        with zmq_sync_socket(sock_path, zmq.PULL) as sock:
            while True:
                opcode_frame, inst_frame = sock.recv_multipart(copy=False)
                opcode = EngineOpcode(bytes(opcode_frame.buffer))

                if opcode == EngineOpcode.ENQUEUE:
                    request = new_request_decoder.decode(inst_frame.buffer)
                else:
                    request = generic_decoder.decode(inst_frame.buffer)

                self.request_queue.put((opcode, request))

    def _response_send_loop(self, sock_path: str) -> None:
        """Continuously dequeue responses and send them to the router."""
        encoder = MsgpackEncoder()
        buffer = bytearray()  # Reuse buffer

        with zmq_sync_socket(sock_path, zmq.PUSH) as sock:
            while True:
                resp = self.response_queue.get()
                encoder.encode_into(resp, buffer)
                sock.send(buffer, copy=False)
