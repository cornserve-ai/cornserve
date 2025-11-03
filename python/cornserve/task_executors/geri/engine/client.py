"""The engine client lives in the router process and interacts with the engine process."""

from __future__ import annotations

import asyncio
import warnings
from asyncio.futures import Future
from collections.abc import AsyncGenerator
from contextlib import suppress

import torch

# Workaround for PyTorch 2.8.0 circular import issue
import torch._dynamo  # noqa: F401
import zmq
import zmq.asyncio
from opentelemetry import propagate, trace

from cornserve.logging import get_logger
from cornserve.sidecar.api import Sidecar
from cornserve.sidecar.schema import SidecarConfig
from cornserve.task_executors.geri.api import (
    BatchGenerationResponse,
    BatchGeriRequest,
    StreamGeriRequest,
    StreamGeriResponseChunk,
)
from cornserve.task_executors.geri.config import GeriConfig
from cornserve.task_executors.geri.engine.core import (
    BatchGeriEngine,
    StreamGeriEngine,
)
from cornserve.task_executors.geri.executor.loader import load_model

# from cornserve.task_executors.geri.models.registry import MODEL_REGISTRY
from cornserve.task_executors.geri.schema import (
    BatchEngineRequest,
    BatchEngineResponse,
    EngineOpcode,
    GeriMode,
    Status,
    StreamEngineRequest,
    StreamEngineResponse,
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
        meta_device = torch.device("meta")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*copying from a non-meta parameter.*",
                category=UserWarning,
            )
            with meta_device:
                temp_model = load_model(model_id=config.model.id, torch_device=meta_device)

        # Create ZMQ sockets for communication with the engine
        self.ctx = zmq.asyncio.Context(io_threads=2)
        self.request_sock_path = get_open_zmq_ipc_path("geri-engine-request")
        self.request_sock = make_zmq_socket(self.ctx, self.request_sock_path, zmq.PUSH)
        self.response_sock_path = get_open_zmq_ipc_path("geri-engine-response")
        self.response_sock = make_zmq_socket(self.ctx, self.response_sock_path, zmq.PULL)

        # Determine Geri mode to run engine core in
        # TODO: from laoder
        self.geri_mode = GeriMode.STREAMING

        # Track pending requests
        self.pending_requests: dict[str, Future[BatchEngineResponse]] = {}

        # For streaming responses, we need producer-consumer queues
        self.pending_streams: dict[str, asyncio.Queue[StreamEngineResponse]] = {}

        # Initialize the sidecar client
        self.sidecar = Sidecar(
            SidecarConfig(
                sidecar_rank=sorted(config.sidecar.ranks)[0],
                group=sorted(config.sidecar.ranks),
                recv_tensor_dtype=temp_model.dtype,
                recv_tensor_shape=(-1, temp_model.embedding_dim),
            )
        )

        # Set up serialization
        self.encoder = MsgpackEncoder()

        # Configure engine process based on what Geri mode we're in (streaming vs batched)
        match self.geri_mode:
            case GeriMode.BATCH:
                self.decoder = MsgpackDecoder(BatchEngineResponse)
                engine_type = BatchGeriEngine
                response_listener = self._batch_response_listener
            case GeriMode.STREAMING:
                self.decoder = MsgpackDecoder(StreamEngineResponse)
                engine_type = StreamGeriEngine
                response_listener = self._stream_response_listener
            case _:
                raise ValueError(f"Unsupported Geri mode: {self.geri_mode}")

        self.engine_process = engine_type.spawn_engine(
            config=config,
            request_sock_path=self.request_sock_path,
            response_sock_path=self.response_sock_path,
        )

        logger.info("EngineClient initialized with engine process PID: %d", self.engine_process.pid)

        # Start response listener task (after engine is ready)
        self.response_task = asyncio.create_task(response_listener())

    async def shutdown(self) -> None:
        """Shutdown the engine client and process."""
        logger.info("Shutting down EngineClient")

        # Send shutdown message to engine
        await self.request_sock.send_multipart((EngineOpcode.SHUTDOWN.value, b""), copy=False)

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

    @tracer.start_as_current_span("engine_client.generate_batch")
    async def generate_batch(self, request_id: str, request: BatchGeriRequest) -> BatchGenerationResponse:
        """Generate image content using the engine process."""
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
        message: BatchEngineRequest = request.to_batch_engine_request(request_id, span_context)

        # Fill in parameters not part of the API request itself
        message.span_context = span_context

        # Create future for response
        future: Future[BatchEngineResponse] = asyncio.Future()
        self.pending_requests[request_id] = future

        # Send message to engine
        await self.request_sock.send_multipart(
            (EngineOpcode.GENERATE.value, self.encoder.encode(message)),
            copy=False,
        )
        logger.info("Sent generate request to engine: %s", message)

        # Wait for response
        try:
            engine_response = await future
        except Exception:
            # Clean up pending request on error
            self.pending_requests.pop(request_id, None)
            raise

        # Convert engine response to API response
        if engine_response.status == Status.SUCCESS:
            return BatchGenerationResponse(
                status=Status.SUCCESS,
                generated=engine_response.generated,
            )
        else:
            return BatchGenerationResponse(
                status=Status.ERROR,
                error_message=engine_response.error_message,
            )

    @tracer.start_as_current_span("engine_client.generate_streaming")
    async def generate_streaming(
        self,
        request_id: str,
        request: StreamGeriRequest,
        stream_inputs: bool = False,  # for now, only outputs are streamed
    ) -> AsyncGenerator[str, None]:
        """Generate streamed-output audio using the engine process."""
        # Propagate trace context
        span_context = {}
        propagator.inject(span_context)

        # TODO: handle streaming inputs

        # Wait for *all* embeddings to arrive in the sidecar
        with tracer.start_as_current_span("engine_client.generate.sidecar_recv_wait"):
            chunk_id = 0
            while True:
                result = await self.sidecar.recv(id=request.embedding_data_id, chunk_id=chunk_id)
                if result is None:
                    break
                chunk_id += 1

        # Create a producer-consumer queue for this request.
        # Will contain EngineResponse objects which hold bytes of wav data.
        response_queue: asyncio.Queue[StreamEngineResponse] = asyncio.Queue()
        self.pending_streams[request_id] = response_queue

        # Create message
        message: StreamEngineRequest = request.to_stream_engine_request(request_id, span_context)

        # Fill in parameters not part of the API request itself
        message.span_context = span_context

        # Send message to engine
        await self.request_sock.send_multipart(
            (EngineOpcode.GENERATE.value, self.encoder.encode(message)),
            copy=False,
        )
        logger.info("Sent generate request to engine: %s", message)

        while True:
            engine_response = await response_queue.get()
            if engine_response.status == Status.SUCCESS:
                if engine_response.generate_bytes is not None:
                    chunk = StreamGeriResponseChunk(engine_response.generate_bytes)
                    # Pack it as a SSE event
                    yield f"data: {chunk.model_dump_json()}\n\n"
                else:
                    logger.info("Generated bytes found to be None for request %s", request_id)
            elif engine_response.status == Status.FINISHED:
                logger.info("Successfully finished request %s", request_id)
                break
            else:
                logger.info("Error detected for request %s", request_id)
                break

        # Cleanup
        self.pending_streams.pop(request_id)

    async def _batch_response_listener(self) -> None:
        """Listen for batch responses from the engine process."""
        logger.info("Starting batch response listener")

        try:
            while True:
                # Receive response from engine
                raw_response = await self.response_sock.recv()
                response: BatchEngineResponse = self.decoder.decode(raw_response)

                # Find pending request and complete it
                future = self.pending_requests.pop(response.request_id, None)
                if future and not future.done():
                    future.set_result(response)
                else:
                    logger.warning("Received response for unknown request: %s", response.request_id)

        except asyncio.CancelledError:
            logger.info("Batch response listener cancelled")
        except Exception:
            logger.exception("Batch response listener failed")

    async def _stream_response_listener(self) -> None:
        """Listen for stream responses from the engine process."""
        logger.info("Starting stream response listener")

        try:
            while True:
                # Receive response from engine
                raw_response = await self.response_sock.recv()
                response: StreamEngineResponse = self.decoder.decode(raw_response)

                # Find pending response data queue
                queue = self.pending_streams.get(response.request_id)
                if queue is not None:
                    await queue.put(response)
                else:
                    logger.warning("Received streaming response for unknown request: %s", response.request_id)

        except asyncio.CancelledError:
            logger.info("Stream response listener cancelled")
        except Exception:
            logger.exception("Stream response listener failed")
