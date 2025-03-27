"""The engine client lives in the router process and interacts with the engine process."""

import asyncio
import os
from asyncio.futures import Future
from contextlib import suppress

import zmq
import zmq.asyncio
from opentelemetry import propagate, trace

from cornserve.logging import get_logger
from cornserve.task_executors.eric.config import EricConfig
from cornserve.task_executors.eric.engine.core import Engine
from cornserve.task_executors.eric.schema import (
    EmbeddingResponse,
    EngineEnqueueMessage,
    EngineOpcode,
    EngineResponse,
    ProcessedEmbeddingData,
)
from cornserve.task_executors.eric.utils.process import kill_process_tree
from cornserve.task_executors.eric.utils.serde import MsgpackDecoder, MsgpackEncoder
from cornserve.task_executors.eric.utils.zmq import (
    get_open_zmq_ipc_path,
    make_zmq_socket,
)

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)
propagator = propagate.get_global_textmap()


class EngineClient:
    """Client that communicates with the engine process."""

    def __init__(self, config: EricConfig):
        """Initialize the engine client.

        1. Creates ZMQ sockets for communication with the engine process.
        2. Sets up a response listener async task to handle incoming messages.
        3. Starts the engine process.
        """
        # Create ZMQ sockets for communication with the engine
        self.ctx = zmq.asyncio.Context(io_threads=2)
        self.request_sock_path = get_open_zmq_ipc_path("engine-request")
        self.request_sock = make_zmq_socket(self.ctx, self.request_sock_path, zmq.PUSH)
        self.response_sock_path = get_open_zmq_ipc_path("engine-response")
        self.response_sock = make_zmq_socket(self.ctx, self.response_sock_path, zmq.PULL)

        # Start an async task that listens for responses from the engine and
        # sets the result of the future corresponding to the request
        self.responses: dict[str, Future[EmbeddingResponse]] = {}
        asyncio.create_task(self._response_listener())

        # Cached variables
        self.config = config
        self.loop = asyncio.get_event_loop()
        self.encoder = MsgpackEncoder()

        # Spawn the engine process and wait for it to be ready
        self.engine_proc = Engine.spawn_engine(
            config=config,
            request_sock_path=self.request_sock_path,
            response_sock_path=self.response_sock_path,
        )

    def health_check(self) -> bool:
        """Check if the engine process is alive."""
        return self.engine_proc.is_alive()

    def shutdown(self) -> None:
        """Shutdown the engine process and close sockets."""
        logger.info("Shutting down engine client.")

        # Terminate the engine process
        if self.engine_proc.is_alive():
            # The engine process shuts down on SIGTERM
            self.engine_proc.terminate()
            self.engine_proc.join(timeout=3)
            if self.engine_proc.is_alive():
                kill_process_tree(self.engine_proc.pid)

        # Closes all sockets and terminates the context
        self.ctx.destroy()

        # Delete socket files
        with suppress(FileNotFoundError):
            os.remove(self.request_sock_path.replace("ipc://", ""))
        with suppress(FileNotFoundError):
            os.remove(self.response_sock_path.replace("ipc://", ""))

    async def _response_listener(self) -> None:
        """Listen for engine responses and set the result of the corresponding future."""
        decoder = MsgpackDecoder(ty=EngineResponse)
        while True:
            message = await self.response_sock.recv()
            resp: EngineResponse = decoder.decode(message)
            result = EmbeddingResponse(
                status=resp.status,
                error_message=resp.error_message,
            )
            for req_id in resp.request_ids:
                try:
                    self.responses.pop(req_id).set_result(result)
                except KeyError:
                    logger.warning(
                        "Response listener received a response for an unknown request ID: %s",
                        req_id,
                    )

    async def embed(
        self,
        request_id: str,
        receiver_sidecar_ranks: list[int] | None,
        processed: list[ProcessedEmbeddingData],
    ) -> EmbeddingResponse:
        """Send the embedding request to the engine and wait for the response."""
        # This future will be resolved by the response listener task
        # when the engine process sends a response back
        fut: Future[EmbeddingResponse] = self.loop.create_future()
        self.responses[request_id] = fut

        carrier = {}
        propagator.inject(carrier)

        # Build and send the request
        req = EngineEnqueueMessage(
            request_id=request_id,
            data=processed,
            receiver_sidecar_ranks=receiver_sidecar_ranks,
            otel_carrier=carrier,
        )

        msg_bytes = self.encoder.encode(req)
        await self.request_sock.send_multipart(
            (EngineOpcode.ENQUEUE.value, msg_bytes),
            copy=False,
        )

        return await fut
