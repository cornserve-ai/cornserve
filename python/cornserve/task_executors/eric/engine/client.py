import asyncio
import multiprocessing as mp

import zmq
import zmq.asyncio
import msgspec
import torch
import numpy as np
from transformers import BatchFeature

from cornserve.task_executors.eric.config import EricConfig
from cornserve.task_executors.eric.zmq_utils import make_zmq_socket
from cornserve.task_executors.eric.engine.core import run_engine
from cornserve.task_executors.eric.models import (
    EngineRequest, EngineResponse, Modality, EmbeddingStatus
)

_encoder = msgspec.msgpack.Encoder()
_decoder = msgspec.msgpack.Decoder(type=EngineResponse)

class EngineClient:
    def __init__(self, config: EricConfig):
        """Initialize the engine client.

        1. Creates ZMQ sockets for communication with the engine process.
        2. Sets up a response listener async task to handle incoming messages.
        3. Starts the engine process.
        """
        self.model_id = config.model_id
        self.ctx = zmq.asyncio.Context(io_threads=2)
        self.push_sock = make_zmq_socket(self.ctx, "tcp://127.0.0.1:5555", zmq.PUSH)
        self.pull_sock = make_zmq_socket(self.ctx, "tcp://127.0.0.1:5556", zmq.PULL)
        self.responses = {}
        self.loop = asyncio.get_event_loop()

        asyncio.create_task(self._response_listener())

        self.engine_proc = mp.get_context("spawn").Process(
            target=run_engine,
            args=(self.model_id,)
        )
        self.engine_proc.start()

    def health_check(self) -> bool:
        """Check if the engine process is alive."""
        return self.engine_proc.is_alive()

    def shutdown(self) -> None:
        """Shutdown the engine process and close sockets."""
        self.engine_proc.terminate()
        self.engine_proc.join(timeout=3)
        if self.engine_proc.exitcode is None:
            self.engine_proc.kill()
            self.engine_proc.join()

        # Closes all sockets and terminates the context
        self.ctx.destroy()

    async def _response_listener(self):
        while True:
            msg_bytes = await self.loop.run_in_executor(None, self.pull_sock.recv)
            resp = _decoder.decode(msg_bytes)
            req_id = resp.request_id
            if req_id in self.responses:
                fut = self.responses.pop(req_id)
                fut.set_result(resp)

    async def embed(self, request_id: str, processed: list[BatchFeature]) -> EngineResponse:
        fut = self.loop.create_future()
        self.responses[request_id] = fut

        # Build and send the request
        # TODO: BatchFeature to request
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
        msg_bytes = _encoder.encode(req)
        await self.loop.run_in_executor(None, self.push_sock.send, msg_bytes)

        # Wait for the engine's response
        return await fut
