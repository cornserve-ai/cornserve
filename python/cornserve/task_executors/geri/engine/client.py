"""The engine client lives in the router process and interacts with the engine process."""

from __future__ import annotations

import asyncio
from asyncio.futures import Future
from contextlib import suppress

import torch
import zmq
import zmq.asyncio
from opentelemetry import propagate, trace

from cornserve.logging import get_logger
from cornserve.sidecar.api import Sidecar
from cornserve.sidecar.schema import SidecarConfig
from cornserve.task_executors.geri.api import GenerationRequest, GenerationResponse, Status
from cornserve.task_executors.geri.config import GeriConfig
from cornserve.task_executors.geri.engine.core import Engine
from cornserve.task_executors.geri.executor.loader import load_model
from cornserve.task_executors.geri.schema import (
    EngineOpcode,
    EngineRequest,
    EngineResponse,
)
from cornserve.task_executors.geri.utils.serde import MsgpackDecoder, MsgpackEncoder
from cornserve.task_executors.geri.utils.zmq import (
    get_open_zmq_ipc_path,
    make_zmq_socket,
)

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)
propagator = propagate.get_global_textmap()


class EngineClient:
    """Client that communicates with the engine process."""

    def __init__(self, config: GeriConfig) -> None:
        """Initialize the engine client.

        1. Creates ZMQ sockets for communication with the engine process.
        2. Sets up a response listener async task to handle incoming messages.
        3. Starts the engine process.
        4. Initializes the sidecar client that waits for data to arrive before enqueuing requests.
        """
        # Figure out the embedding dimension from a temporary model instance
        with torch.device("meta"):
            temp_model = load_model(
                model_id=config.model.id,
                torch_device=torch.device("meta"),
            )

        # Create ZMQ sockets for communication with the engine
        self.ctx = zmq.asyncio.Context(io_threads=2)
        self.request_sock_path = get_open_zmq_ipc_path("geri-engine-request")
        self.request_sock = make_zmq_socket(self.ctx, self.request_sock_path, zmq.PUSH)
        self.response_sock_path = get_open_zmq_ipc_path("geri-engine-response")
        self.response_sock = make_zmq_socket(self.ctx, self.response_sock_path, zmq.PULL)

        # Set up serialization
        self.encoder = MsgpackEncoder()
        self.decoder = MsgpackDecoder(EngineResponse)

        # Track pending requests
        self.pending_requests: dict[str, Future[EngineResponse]] = {}

        # Initialize the sidecar client
        self.sidecar = Sidecar(
            SidecarConfig(
                sidecar_rank=sorted(config.sidecar.ranks)[0],
                group=sorted(config.sidecar.ranks),
                recv_tensor_dtype=temp_model.dtype,
                recv_tensor_shape=(-1, temp_model.embedding_dim),
            )
        )

        # Start the engine process
        self.engine_process = Engine.spawn_engine(
            config=config,
            request_sock_path=self.request_sock_path,
            response_sock_path=self.response_sock_path,
        )

        logger.info("EngineClient initialized with engine process PID: %d", self.engine_process.pid)

        # Start response listener task (after engine is ready)
        self.response_task = asyncio.create_task(self._response_listener())

    async def shutdown(self) -> None:
        """Shutdown the engine client and process."""
        logger.info("Shutting down EngineClient")

        # Send shutdown message to engine
        await self.request_sock.send_multipart((EngineOpcode.SHUTDOWN.value,), copy=False)

        # Wait for engine process to shutdown
        self.engine_process.join(timeout=10)
        if self.engine_process.is_alive():
            logger.warning("Engine process did not shutdown gracefully, terminating")
            self.engine_process.terminate()
            self.engine_process.join()

        # Cancel response listener
        self.response_task.cancel()
        with suppress(asyncio.CancelledError):
            await self.response_task

        # Close ZMQ context
        self.ctx.destroy()

    @tracer.start_as_current_span("engine_client.generate")
    async def generate(self, request_id: str, request: GenerationRequest) -> GenerationResponse:
        """Generate content using the engine process."""
        # Propagate trace context
        span_context = {}
        propagator.inject(span_context)

        # Wait for the embeddings to arrive in the sidecar
        with tracer.start_as_current_span("engine_client.generate.sidecar_recv_wait"):
            chunk_id = 0
            while True:
                result = await self.sidecar.recv(id=request.embedding_data_id, chunk_id=chunk_id)
                if result is None:
                    break
                chunk_id += 1

        # Create message
        message = EngineRequest(
            request_id=request_id,
            embedding_data_id=request.embedding_data_id,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            skip_tokens=request.skip_tokens,
            span_context=span_context,
        )

        # Create future for response
        future: Future[EngineResponse] = asyncio.Future()
        self.pending_requests[request_id] = future

        # Send message to engine
        await self.request_sock.send_multipart(
            (EngineOpcode.GENERATE.value, self.encoder.encode(message)),
            copy=False,
        )

        # Wait for response
        try:
            engine_response = await future
        except Exception:
            # Clean up pending request on error
            self.pending_requests.pop(request_id, None)
            raise

        # Convert engine response to API response
        if engine_response.status == Status.SUCCESS:
            return GenerationResponse(
                status=Status.SUCCESS,
                generated=engine_response.generated,
            )
        else:
            return GenerationResponse(
                status=Status.ERROR,
                error_message=engine_response.error_message,
            )

    async def _response_listener(self) -> None:
        """Listen for responses from the engine process."""
        logger.info("Starting response listener")

        try:
            while True:
                # Receive response from engine
                raw_response = await self.response_sock.recv()
                response: EngineResponse = self.decoder.decode(raw_response)

                # Find pending request and complete it
                future = self.pending_requests.pop(response.request_id, None)
                if future and not future.done():
                    future.set_result(response)
                else:
                    logger.warning("Received response for unknown request: %s", response.request_id)

        except asyncio.CancelledError:
            logger.info("Response listener cancelled")
        except Exception:
            logger.exception("Response listener failed")
