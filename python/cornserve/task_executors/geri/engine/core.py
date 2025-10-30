"""Geri engine core."""

from __future__ import annotations

import multiprocessing as mp
import queue
import signal
import threading
from multiprocessing.context import SpawnProcess
from typing import Any

import torch
import zmq
from opentelemetry import propagate, trace

from cornserve.logging import get_logger
from cornserve.sidecar.api import Sidecar
from cornserve.sidecar.schema import SidecarConfig
from cornserve.task_executors.geri.api import Status
from cornserve.task_executors.geri.config import GeriConfig
from cornserve.task_executors.geri.engine.scheduler import Scheduler
from cornserve.task_executors.geri.executor.executor import ModelExecutor
from cornserve.task_executors.geri.executor.loader import load_model
from cornserve.task_executors.geri.schema import (
    EngineOpcode,
    EngineRequest,
    EngineRequestType,
    EngineResponse,
)
from cornserve.task_executors.geri.utils.serde import MsgpackDecoder, MsgpackEncoder
from cornserve.task_executors.geri.utils.zmq import zmq_sync_socket_ctx
from cornserve.tracing import configure_otel

logger = get_logger(__name__)
tracer = trace.get_tracer(__name__)
propagator = propagate.get_global_textmap()


class Engine:
    """Geri core engine.

    The engine receives generation requests from the router and
    invokes the model executor to launch image generation. When content
    is generated, the engine sends a response back to the router.
    """

    def __init__(
        self,
        config: GeriConfig,
        request_sock_path: str,
        response_sock_path: str,
    ) -> None:
        """Initialize the engine.

        Args:
            config: Geri configuration.
            request_sock_path: Path for receiving requests from router.
            response_sock_path: Path for sending responses to router.
        """
        self.config = config

        model = load_model(model_id=config.model.id, torch_device=torch.device("cuda"))

        self.sidecar = Sidecar(
            SidecarConfig(
                sidecar_rank=sorted(config.sidecar.ranks)[0],
                group=sorted(config.sidecar.ranks),
                recv_tensor_dtype=model.dtype,
                recv_tensor_shape=(-1, model.embedding_dim),
            )
        )

        # Set up serialization
        self.decoder = MsgpackDecoder(EngineRequest)
        self.encoder = MsgpackEncoder()

        # Initialize model executor
        self.executor = ModelExecutor(model=model)

        # Initialize scheduler
        self.scheduler = Scheduler(max_batch_size=config.server.max_batch_size)

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

        logger.info("Engine core initialized")

    def run(self) -> None:
        """Main engine loop."""
        logger.info("Starting engine loop")

        # Loop until process is sent a SIGINT or SIGTERM
        while True:
            # Poll the input queue until there is work to do
            if not self.scheduler.has_waiting_requests():
                while True:
                    try:
                        req = self.request_queue.get(timeout=3.0)
                        self._handle_client_request(*req)
                        break
                    except queue.Empty:
                        logger.debug("Engine busy loop waiting")
                    except BaseException:
                        raise

            # Handle any new client requests that arrived during the wait
            while not self.request_queue.empty():
                req = self.request_queue.get_nowait()
                self._handle_client_request(*req)

            # Step the engine core
            responses = self.step()

            # Put responses in the response queue
            if responses:
                for response in responses:
                    self.response_queue.put_nowait(response)

    def step(self) -> list[EngineResponse]:
        """Step the engine core.

        This function is called in a loop to process requests and send
        responses. It handles scheduling, executing, and processing results.
        """
        batch = self.scheduler.schedule()
        if batch is None:
            return []

        try:
            # Collect all embeddings for the batch
            prompt_embeds = []
            for embedding_data_id, skip_tokens in zip(batch.embedding_data_ids, batch.skip_tokens, strict=True):
                # Collect chunks for this embedding data ID.
                # Since data was already awaited by the engine client, these
                # `recv_sync` calls should return immediately.
                embedding_chunks = []
                chunk_id = 0
                while True:
                    chunk = self.sidecar.recv_sync(embedding_data_id, chunk_id=chunk_id)
                    if chunk is None:
                        break
                    embedding_chunks.append(chunk)
                    chunk_id += 1

                # Concatenate chunks for this request and slice initial tokens as specified
                if embedding_chunks:
                    embedding = torch.cat(embedding_chunks, dim=0)[skip_tokens:].contiguous()
                    prompt_embeds.append(embedding)
                    logger.debug(
                        "Retrieved embedding for data ID %s with %s and %d chunks (skipped %d initial tokens).",
                        embedding_data_id,
                        list(embedding.shape),
                        chunk_id,
                        skip_tokens,
                    )
                else:
                    logger.error("No embedding chunks received for data ID: %s", embedding_data_id)
                    raise RuntimeError(f"No embeddings received for data ID: {embedding_data_id}")

            # Create a batch-level span for the entire generation operation
            with tracer.start_as_current_span("geri.engine.generate_batch") as batch_span:
                batch_span.set_attribute("geri.batch_size", len(batch))
                batch_span.set_attribute("geri.height", batch.height)
                batch_span.set_attribute("geri.width", batch.width)
                batch_span.set_attribute("geri.num_inference_steps", batch.num_inference_steps)

                # Create individual spans for each request in the batch as child spans
                request_spans: list[trace.Span] = []
                for i, (request_id, original_span) in enumerate(zip(batch.request_ids, batch.spans, strict=True)):
                    if original_span is not None:
                        # Create a child span under the original request's context
                        context = trace.set_span_in_context(original_span)
                        request_span = tracer.start_span("geri.engine.generate_request", context=context)
                        request_span.set_attribute("geri.request_id", request_id)
                        request_span.set_attribute("geri.batch_position", i)
                        request_span.set_attribute("geri.batch_size", len(batch))
                        request_spans.append(request_span)

                result = None
                if batch.request_type == EngineRequestType.NON_STREAMING:
                    # Execute the batch
                    result = self.executor.generate(
                        prompt_embeds=[e.cuda() for e in prompt_embeds],
                        height=batch.height,
                        width=batch.width,
                        num_inference_steps=batch.num_inference_steps,
                    )

                    # End individual request spans with result information
                    for request_span in request_spans:
                        request_span.set_attribute("geri.batch_status", result.status.value)
                        if result.error_message:
                            request_span.set_attribute("geri.batch_error_message", result.error_message)
                        request_span.end()

                # The streaming case
                streaming_result = None
                if batch.request_type == EngineRequestType.STREAMING:
                    # For streaming responses, we can't immediately end individual request spans
                    # since the obtained result simply holds a generator object.
                    streaming_result = self.executor.generate_streaming(
                        prompt_embeds=[e.cuda() for e in prompt_embeds],
                    )

            # Non-streaming case
            if result is not None:
                # Split the batched results back to individual responses
                responses: list[EngineResponse] = []
                for request_id, generated, span in zip(batch.request_ids, result.generated, batch.spans, strict=True):
                    response = EngineResponse(
                        request_id=request_id,
                        status=result.status,
                        generated=generated,
                        error_message=result.error_message,
                    )
                    responses.append(response)

                    # End the original request span (the top-level span for this request)
                    if span is not None:
                        span.set_attribute("geri.status", result.status.value)
                        if result.error_message:
                            span.set_attribute("geri.error_message", result.error_message)
                        span.end()

                logger.info("Processed batch of %d requests", len(batch))
                return responses

            elif streaming_result is not None:
                # All requests should all have the same request_id if streaming
                request_id = batch.requests[0].request_id

                if streaming_result.status == Status.ERROR or streaming_result.streamed_generator is None:
                    # Use outer error handler to inform all requests in batch of failure
                    for request_span in request_spans:
                        request_span.set_attribute("geri.batch_status", streaming_result.status.value)
                        if streaming_result.error_message:
                            request_span.set_attribute("geri.batch_error_message", streaming_result.error_message)
                        request_span.end()
                    raise ValueError(
                        "Generator formation failed with status "
                        f"{streaming_result.status}: "
                        f"{streaming_result.error_message}"
                    )

                # Note that the i-th yield of streamed_generator no longer corresponds to the
                # i-th request in the batch, since the streamed_generator yields whenever a
                # fixed chunk of results become ready, not when it has finished processing
                # data for a single entire request.
                if streaming_result.status == Status.SUCCESS:
                    for wav_chunk in streaming_result.streamed_generator:
                        response = EngineResponse(
                            request_id=request_id,
                            status=Status.SUCCESS,
                            generated=wav_chunk.cpu().numpy().tobytes(),
                            request_type=EngineRequestType.STREAMING,
                        )
                        self.response_queue.put_nowait(response)

                    # Signal that the stream has ended
                    response = EngineResponse(
                        request_id=request_id,
                        status=Status.FINISHED,
                        request_type=EngineRequestType.STREAMING,
                    )
                    self.response_queue.put_nowait(response)

                # Now end the individual spans we didn't end earlier
                for request_span in request_spans:
                    request_span.set_attribute("geri.batch_status", streaming_result.status.value)
                    if streaming_result.error_message:
                        request_span.set_attribute("geri.batch_error_message", streaming_result.error_message)
                    request_span.end()

                for span in batch.spans:
                    # End the original request span (the top-level span for this request)
                    if span is not None:
                        span.set_attribute("geri.status", streaming_result.status.value)
                        if streaming_result.error_message:
                            span.set_attribute("geri.error_message", streaming_result.error_message)
                        span.end()

            return []

        except Exception as e:
            logger.exception("Batch processing failed")

            # Send error responses to all requests in the batch
            error_responses = []
            for request_id, span in zip(batch.request_ids, batch.spans, strict=True):
                response = EngineResponse(
                    request_id=request_id,
                    status=Status.ERROR,
                    error_message=f"Batch processing failed: {str(e)}",
                )
                error_responses.append(response)

                # End the original request span with error information
                if span is not None:
                    span.set_attribute("geri.status", "ERROR")
                    span.set_attribute("geri.error_message", f"Batch processing failed: {str(e)}")
                    span.record_exception(e)
                    span.end()

            return error_responses

    def _handle_client_request(self, opcode: EngineOpcode, request: Any) -> None:
        """Dispatch request from client."""
        match opcode:
            case EngineOpcode.GENERATE:
                logger.info("Adding request: %s", request.request_id)
                if not isinstance(request, EngineRequest):
                    logger.error("Invalid request type for GENERATE: %s", type(request))
                    return

                # Set up tracing span if context is provided
                span = None
                if request.span_context is not None:
                    context = propagator.extract(request.span_context)
                    span = tracer.start_span("geri.engine.process_request", context=context)
                    span.set_attribute("geri.engine.process_request.request_id", request.request_id)
                    span.set_attribute("geri.engine.process_request.height", request.height)
                    span.set_attribute("geri.engine.process_request.width", request.width)

                self.scheduler.enqueue(request, span)
            case EngineOpcode.SHUTDOWN:
                logger.info("Received shutdown message")
                raise SystemExit()
            case _:
                logger.error("Unknown opcode: %s", opcode)

    def _request_receive_loop(self, sock_path: str) -> None:
        """Continuously receive requests from a ZMQ socket and enqueue them."""
        logger.info("Starting request receive thread. Listening on %s", sock_path)
        with zmq_sync_socket_ctx(sock_path, zmq.PULL) as sock:
            while True:
                opcode_frame, inst_frame = sock.recv_multipart(copy=False)
                opcode = EngineOpcode(bytes(opcode_frame.buffer))

                request = self.decoder.decode(inst_frame.buffer) if opcode == EngineOpcode.GENERATE else None

                self.request_queue.put((opcode, request))

    def _response_send_loop(self, sock_path: str) -> None:
        """Continuously dequeue responses and send them to the router."""
        buffer = bytearray()  # Reuse buffer

        with zmq_sync_socket_ctx(sock_path, zmq.PUSH) as sock:
            while True:
                resp = self.response_queue.get()
                self.encoder.encode_into(resp, buffer)
                sock.send(buffer, copy=False)

    def shutdown(self) -> None:
        """Shutdown the engine and clean up resources."""
        logger.info("Shutting down engine")

        if hasattr(self, "executor"):
            self.executor.shutdown()

    @classmethod
    def spawn_engine(
        cls,
        config: GeriConfig,
        request_sock_path: str,
        response_sock_path: str,
    ) -> SpawnProcess:
        """Spawn the engine process.

        Called by the engine client. We're not inside the engine process yet!

        This function spawns the engine in a separate process and
        waits for it to be ready by blocking on a pipe.
        """
        context = mp.get_context("spawn")
        reader, writer = context.Pipe(duplex=False)
        ready_message = b"ready"
        engine_proc = context.Process(
            name="geri_engine",
            target=cls.main,
            kwargs=dict(
                config=config,
                request_sock_path=request_sock_path,
                response_sock_path=response_sock_path,
                ready_pipe=writer,
                ready_message=ready_message,
            ),
        )
        engine_proc.start()

        # Wait for engine to be ready
        logger.info("Waiting for engine to be ready...")
        received_message = reader.recv()
        if received_message != ready_message:
            raise RuntimeError(f"Engine failed to start, got message: {received_message}")

        reader.close()
        logger.info("Engine is ready")
        return engine_proc

    @staticmethod
    def main(
        config: GeriConfig,
        request_sock_path: str,
        response_sock_path: str,
        ready_pipe,
        ready_message: bytes,
    ) -> None:
        """Main entry point for the engine process."""
        # Configure OpenTelemetry for this process
        configure_otel("geri-engine")

        shutdown_requested = False

        def signal_handler(signum: int, frame) -> None:
            nonlocal shutdown_requested
            logger.info("Received signal %d, shutting down engine", signum)
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Start the engine
        engine: Engine | None = None
        try:
            # Create and initialize engine
            engine = Engine(
                config=config,
                request_sock_path=request_sock_path,
                response_sock_path=response_sock_path,
            )

            # Signal that we're ready
            ready_pipe.send(ready_message)
            ready_pipe.close()

            # Run the engine loop
            engine.run()

        except SystemExit:
            logger.debug("Engine interrupted by signal.")
        except Exception:
            logger.exception("Engine hit an exception.")
        finally:
            if engine:
                engine.shutdown()
