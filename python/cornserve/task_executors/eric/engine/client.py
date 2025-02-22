import os
import signal
import asyncio
import multiprocessing as mp
from contextlib import suppress
from asyncio.futures import Future

import zmq
import zmq.asyncio
import msgspec
import torch
import numpy as np
from transformers import BatchFeature

from cornserve.task_executors.eric.config import EricConfig
from cornserve.task_executors.eric.utils import get_open_zmq_ipc_path, make_zmq_socket, kill_process_tree
from cornserve.task_executors.eric.engine.core import Engine
from cornserve.task_executors.eric.schema import (
    EngineRequest, EngineResponse, Modality, EmbeddingStatus
)
from cornserve.logging import get_logger

logger = get_logger(__name__)


class EngineClient:
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
        self.responses: dict[str, Future[EngineResponse]] = {}
        asyncio.create_task(self._response_listener())

        # Cached variables
        self.config = config
        self.loop = asyncio.get_event_loop()
        self.encoder = msgspec.msgpack.Encoder()

        # Spawn the engine process and wait for it to be ready
        context = mp.get_context("spawn")
        reader, writer = context.Pipe(duplex=False)
        ready_message = b"ready"
        self.engine_proc = context.Process(
            target=Engine.run_engine,
            kwargs=dict(
                config=config,
                model_id=self.config.model.id,
                modality=config.modality.ty,
                request_sock_path=self.request_sock_path,
                response_sock_path=self.response_sock_path,
                ready_pipe=writer,
                ready_message=ready_message,
            )
        )
        self.engine_proc.start()
        if reader.recv() != ready_message:
            raise RuntimeError("Engine process failed to start")

    def health_check(self) -> bool:
        """Check if the engine process is alive."""
        return self.engine_proc.is_alive()

    def shutdown(self) -> None:
        """Shutdown the engine process and close sockets."""
        # Terminate the engine process
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
        decoder = msgspec.msgpack.Decoder(type=EngineResponse)
        while True:
            message = await self.response_sock.recv()
            resp = decoder.decode(message)
            req_id = resp.request_id
            try:
                fut = self.responses.pop(req_id)
                fut.set_result(resp)
            except KeyError:
                logger.warning("Response listener received a response for an unknown request ID: %s", req_id)
                pass

    async def embed(self, request_id: str, processed: list[BatchFeature]) -> EngineResponse:
        """Send the embedding request to the engine and wait for the response."""
        # This future will be resolved by the response listener task
        # when the engine process sends a response back
        fut: Future[EngineResponse] = self.loop.create_future()
        self.responses[request_id] = fut

        # Build and send the request
        # TODO: BatchFeature to request. First, try actually running it through the HF model.
        arr = tensors.numpy()
        shape = tuple(arr.shape)
        dtype_str = str(arr.dtype)
        raw_data = arr.tobytes()
        req = EngineRequest(
            request_id=request_id,
            shape=shape,
            dtype=dtype_str,
            processed_tensors=raw_data
        )
        msg_bytes = self.encoder.encode(req)
        await self.request_sock.send(msg_bytes)

        return await fut
