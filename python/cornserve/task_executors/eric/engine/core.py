import queue
import threading
from multiprocessing.connection import Connection

import zmq
import msgspec.msgpack as msp
import numpy as np
import torch

from cornserve.task_executors.eric.config import EricConfig
from cornserve.task_executors.eric.utils import make_zmq_socket
from cornserve.task_executors.eric.executor.executor import ModelExecutor
from cornserve.task_executors.eric.engine.scheduler import Scheduler
from cornserve.task_executors.eric.schema import (
    EngineRequest, EngineResponse, EmbeddingStatus, Modality
)

_encoder = msp.Encoder()
_decoder = msp.Decoder()


class Engine:
    """Eric core engine.

    The engine receives modality embedding requests from the router and
    invokes the model executor to launch embedding computation. When tenosrs
    are sent to the sidecar, the engine sends a message to the router to
    signal completion.
    """

    def __init__(self, model_id: str, modality: Modality) -> None:
        """Initialize the engine."""
        self.model_id = model_id
        self.ctx = zmq.Context(io_threads=2)

        # Pull socket for receiving requests from the router
        self.pull_sock = make_zmq_socket(
            ctx=self.ctx,
            path="tcp://127.0.0.1:5555",
            sock_type=zmq.PULL,
        )
        # Push socket for sending responses back to the router
        self.push_sock = make_zmq_socket(
            ctx=self.ctx,
            path="tcp://127.0.0.1:5556",
            sock_type=zmq.PUSH,
        )

        self.input_queue = queue.Queue()
        self.executor = ModelExecutor(model_id, modality)
        self.scheduler = Scheduler()

        # Background thread for receiving requests
        self._receive_thread = threading.Thread(
            target=self._receive_loop,
            daemon=True
        )

    @staticmethod
    def run_engine(
        config: EricConfig,
        model_id: str,
        modality: Modality,
        request_sock_path: str,
        response_sock_path: str,
        ready_pipe: Connection,
        ready_message: bytes,
    ) -> None:
        """Start the engine process."""


    def _receive_loop(self):
        """Continuously receive requests from the pull socket and enqueue them."""
        while True:
            msg_bytes = self.pull_sock.recv()
            req = _decoder.decode(msg_bytes, type=EngineRequest)
            self.input_queue.put(req)

    def run(self):
        """Main engine loop: schedule batches, run the model, send results."""
        self._receive_thread.start()
        while True:
            items = self.scheduler.batch(self.input_queue)
            if not items:
                continue
            try:
                # Convert each EngineRequest's raw data to a torch Tensor on GPU
                batched_tensors = []
                for req in items:
                    arr = np.frombuffer(req.data, dtype=req.dtype).reshape(req.shape)
                    t = torch.from_numpy(arr).to(self.worker.device)
                    batched_tensors.append((req, t))

                # Execute the model
                results = self.worker.execute_model(batched_tensors)

                # Send out responses
                for (req, _), emb in zip(batched_tensors, results):
                    emb_cpu = emb.cpu().contiguous()
                    out_arr = emb_cpu.numpy()
                    out_shape = list(out_arr.shape)
                    out_dtype = str(out_arr.dtype)
                    out_data = out_arr.tobytes()

                    resp = EngineResponse(
                        request_id=req.request_id,
                        status=EmbeddingStatus.SUCCESS,
                    )
                    self.push_sock.send(_encoder.encode(resp))

            except Exception as e:
                # On error, return an error response for each item
                for req in items:
                    err_resp = EngineResponse(
                        request_id=req.request_id,
                        status=EmbeddingStatus.ERROR,
                        error_message=str(e),
                    )
                    self.push_sock.send(_encoder.encode(err_resp))
