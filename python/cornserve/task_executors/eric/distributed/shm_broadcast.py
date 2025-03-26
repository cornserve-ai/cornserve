"""Inter-process broadcast using shared memory and ZMQ."""

from __future__ import annotations

import os
import pickle
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from unittest.mock import patch

import torch
import zmq

from cornserve.logging import get_logger
from cornserve.task_executors.eric.utils.network import get_open_port

RINGBUFFER_FULL_WARNING_INTERVAL = 10

logger = get_logger(__name__)

# We prefer to use os.sched_yield as it results in tighter polling loops,
# measured to be around 3e-7 seconds. However on earlier versions of Python
# os.sched_yield() does not release the GIL, so we fall back to time.sleep(0)
USE_SCHED_YIELD = (sys.version_info[:3] >= (3, 11, 1)) or (sys.version_info[:2] == (3, 10) and sys.version_info[2] >= 8)


def sched_yield():
    """Yield the processor to other threads."""
    if USE_SCHED_YIELD:
        os.sched_yield()
    else:
        time.sleep(0)


class ShmRingBuffer:
    """A shared memory ring buffer implementation for broadcast communication.

    This ring buffer is co-designed with MessageQueue, which actually reads
    and modifies the state bits in the ring buffer.

    Essentially, it is a queue where only one will `enqueue` and multiple
    will `dequeue`. The max size of each item, together with the max number
    of items that can be stored in the buffer are known in advance.
    In this case, we don't need to synchronize the access to the buffer.

    ## Buffer memory layout

              data                                 metadata
                |                                      |
                | (current_idx)                        | (current_idx)
                v                                      v
    +-------------------------------+----------------------------------------+
    | chunk0 | chunk1 | ... | chunk | metadata0 | metadata1 | ... | metadata |
    +-------------------------------+----------------------------------------+
    | max_chunks x max_chunk_bytes  | max_chunks x (1 + n_reader) bytes      |

    metadata memory layout: each byte is a flag, the first byte is the written
    flag, and the rest are reader flags. The flags are set to 0 by default.
    +--------------+--------------+--------------+-----+--------------+
    | written_flag | reader0_flag | reader1_flag | ... | readerN_flag |
    +--------------+--------------+--------------+-----+--------------+

    ## Metadata state

    (case 1) 0???...???: Block not written. No read, can write.
    (case 2) 1000...000: Block just written. Can read, no write.
    (case 3) 1???...???: Block written and read by some. Read once, no write.
    (case 4) 1111...111: Block written and read by all. No read, can write.

    ## State transition for readers

    1. `MessageQueue.acquire_read` called by reader.
    2. `get_metadata` polled until a readable block (case 2 or 3) is found.
    3. `get_data` called to get the data block, which is `yield`ed to the caller.
    4. When we're back, and the reader marks the block as read (from 0 to 1).

    ## State transition for writers

    1. `MessageQueue.acquire_write` called by writer.
    2. `get_metadata` polled until a writable block (case 1 or 4) is found.
    3. The metadata block's written flag is set to 0, transitioning to case 1.
    4. `get_data` called to get the data block, which is `yield`ed to the caller.
    5. When we're back, first set all reader flags to 0, and then set the
        written flag to 1. Order is important here. If done in opposite order,
        readers may think it's a written block that they already read.

    During creation, `name` is None and the buffer is created. We can pass the
    created object to other processes by pickling it. The other processes will
    get the name of the shared memory and open it so that they can access the
    same shared memory buffer.
    """

    def __init__(
        self,
        n_reader: int,
        max_chunk_bytes: int,
        max_chunks: int,
        name: str | None = None,
    ) -> None:
        """Initialize the shared memory ring buffer."""
        self.n_reader = n_reader
        self.metadata_size = 1 + n_reader
        self.max_chunk_bytes = max_chunk_bytes
        self.max_chunks = max_chunks
        self.total_bytes_of_buffer = (self.max_chunk_bytes + self.metadata_size) * self.max_chunks
        self.data_offset = 0
        self.metadata_offset = self.max_chunk_bytes * self.max_chunks

        if name is None:
            # we are creating a buffer
            self.is_creator = True
            self.shared_memory = shared_memory.SharedMemory(create=True, size=self.total_bytes_of_buffer)
            # initialize the metadata section to 0
            with memoryview(self.shared_memory.buf[self.metadata_offset :]) as metadata_buffer:
                torch.frombuffer(metadata_buffer, dtype=torch.uint8).fill_(0)
        else:
            # we are opening an existing buffer
            self.is_creator = False
            # fix to https://stackoverflow.com/q/62748654/9191338
            # Python incorrectly tracks shared memory even if it is not
            # created by the process. The following patch is a workaround.
            with patch(
                "multiprocessing.resource_tracker.register",
                lambda *args, **kwargs: None,
            ):
                try:
                    self.shared_memory = shared_memory.SharedMemory(name=name)
                    assert self.shared_memory.size == self.total_bytes_of_buffer
                except FileNotFoundError:
                    # we might deserialize the object in a different node
                    # in this case, this object is not used,
                    # and we should suppress the error
                    pass

    def handle(self):
        """Get a multiprocessing-aware handle to the buffer."""
        return (
            self.n_reader,
            self.max_chunk_bytes,
            self.max_chunks,
            self.shared_memory.name,
        )

    def __reduce__(self):
        """Return a tuple that can be used to reconstruct the object."""
        return (
            self.__class__,
            self.handle(),
        )

    def __del__(self):
        """Clean up the shared memory buffer."""
        if hasattr(self, "shared_memory"):
            self.shared_memory.close()
            if self.is_creator:
                self.shared_memory.unlink()

    @contextmanager
    def get_data(self, current_idx: int):
        """Get a memoryview of the data chunk at the given index."""
        start = self.data_offset + current_idx * self.max_chunk_bytes
        end = start + self.max_chunk_bytes
        with memoryview(self.shared_memory.buf[start:end]) as buf:
            yield buf

    @contextmanager
    def get_metadata(self, current_idx: int):
        """Get a memoryview of the metadata chunk at the given index."""
        start = self.metadata_offset + current_idx * self.metadata_size
        end = start + self.metadata_size
        with memoryview(self.shared_memory.buf[start:end]) as buf:
            yield buf


@dataclass
class MessageQueueHandle:
    """A handle to the message queue."""

    connect_ip: str
    local_reader_ranks: list[int] = field(default_factory=list)
    buffer_handle: tuple[int, int, int, str] | None = None
    local_subscribe_port: int | None = None


class MessageQueue:
    """A message queue for inter-process communication using shared memory and ZMQ."""

    def __init__(
        self,
        n_reader,  # number of all readers
        n_local_reader,  # number of local readers through shared memory
        local_reader_ranks: list[int] | None = None,
        max_chunk_bytes: int = 1024 * 1024 * 10,
        max_chunks: int = 10,
        connect_ip: str | None = None,
    ) -> None:
        """Initialize the message queue."""
        if local_reader_ranks is None:
            local_reader_ranks = list(range(n_local_reader))
        else:
            assert len(local_reader_ranks) == n_local_reader
        self.n_local_reader = n_local_reader
        n_remote_reader = n_reader - n_local_reader
        self.n_remote_reader = n_remote_reader

        if connect_ip is None:
            assert n_remote_reader == 0
            connect_ip = "127.0.0.1"
            # connect_ip = get_ip() if n_remote_reader > 0 else "127.0.0.1"

        context = zmq.Context()

        assert n_local_reader > 0
        # for local readers, we will:
        # 1. create a shared memory ring buffer to communicate small data
        # 2. create a publish-subscribe socket to communicate large data
        self.buffer = ShmRingBuffer(n_local_reader, max_chunk_bytes, max_chunks)

        # XPUB is very similar to PUB,
        # except that it can receive subscription messages
        # to confirm the number of subscribers
        self.local_socket = context.socket(zmq.XPUB)
        # set the verbose option so that we can receive every subscription
        # message. otherwise, we will only receive the first subscription
        # see http://api.zeromq.org/3-3:zmq-setsockopt for more details
        self.local_socket.setsockopt(zmq.XPUB_VERBOSE, True)
        local_subscribe_port = get_open_port()
        socket_addr = f"tcp://127.0.0.1:{local_subscribe_port}"
        logger.debug("Binding to %s", socket_addr)
        self.local_socket.bind(socket_addr)

        self.current_idx = 0

        self._is_writer = True
        self._is_local_reader = False
        self.local_reader_rank = -1
        # rank does not matter for remote readers
        self._is_remote_reader = False

        self.handle = MessageQueueHandle(
            connect_ip=connect_ip,
            local_reader_ranks=local_reader_ranks,
            buffer_handle=self.buffer.handle() if self.buffer is not None else None,
            local_subscribe_port=local_subscribe_port,
        )

        logger.info("Message queue created from communication handle: %s", self.handle)

    def export_handle(self) -> MessageQueueHandle:
        """Export the handle to the message queue.

        This is used to create the message queue in other processes.
        """
        return self.handle

    @staticmethod
    def create_from_handle(handle: MessageQueueHandle, rank: int) -> MessageQueue:
        """Create a message queue from the handle."""
        # Bypass the __init__ method to avoid creating a new buffer
        self = MessageQueue.__new__(MessageQueue)
        self.handle = handle
        self._is_writer = False

        context = zmq.Context()

        assert rank in handle.local_reader_ranks

        assert handle.buffer_handle is not None
        self.buffer = ShmRingBuffer(*handle.buffer_handle)
        self.current_idx = 0
        self.local_reader_rank = handle.local_reader_ranks.index(rank)
        self._is_local_reader = True
        self._is_remote_reader = False

        self.local_socket = context.socket(zmq.SUB)
        self.local_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        socket_addr = f"tcp://127.0.0.1:{handle.local_subscribe_port}"
        logger.debug("Connecting to %s", socket_addr)
        self.local_socket.connect(socket_addr)

        return self

    def wait_until_ready(self):
        """Wait until all readers are connected to the writer.

        This is a collective operation. All processes (including the readers
        and the writer) should call this function.
        """
        if self._is_writer:
            # wait for all readers to connect
            for _ in range(self.n_local_reader):
                # wait for subscription messages from all local readers
                self.local_socket.recv()
            if self.n_local_reader > 0:
                # send a message to all local readers
                # to make sure the publish channel is working
                self.local_socket.send(b"READY")
        elif self._is_local_reader:
            # wait for the writer to send a message
            recv = self.local_socket.recv()
            assert recv == b"READY"

    @contextmanager
    def acquire_write(self, timeout: float | None = None):
        """Acquire the next block in the ring buffer to write to."""
        assert self._is_writer, "Only writers can acquire write"
        start_time = time.monotonic()
        n_warning = 1
        while True:
            with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
                read_count = sum(metadata_buffer[1:])
                written_flag = metadata_buffer[0]
                if written_flag and read_count != self.buffer.n_reader:
                    # this block is written and not read by all readers
                    # for writers, `self.current_idx` is the next block to write
                    # if this block is not ready to write,
                    # we need to wait until it is read by all readers

                    # Release the processor to other threads
                    sched_yield()

                    # if we wait for a long time, log a message
                    if time.monotonic() - start_time > RINGBUFFER_FULL_WARNING_INTERVAL * n_warning:
                        logger.debug(
                            "No available block found in %s second. ",
                            RINGBUFFER_FULL_WARNING_INTERVAL,
                        )
                        n_warning += 1

                    # if we time out, raise an exception
                    if timeout is not None and time.monotonic() - start_time > timeout:
                        raise TimeoutError

                    continue
                # found a block that is either
                # (1) not written
                # (2) read by all readers

                # mark the block as not written
                metadata_buffer[0] = 0
                # let caller write to the buffer
                with self.buffer.get_data(self.current_idx) as buf:
                    yield buf

                # caller has written to the buffer
                # NOTE: order is important here
                # first set the read flags to 0
                # then set the written flag to 1
                # otherwise, the readers may think they already read the block
                for i in range(1, self.buffer.n_reader + 1):
                    # set read flag to 0, meaning it is not read yet
                    metadata_buffer[i] = 0
                # mark the block as written
                metadata_buffer[0] = 1
                self.current_idx = (self.current_idx + 1) % self.buffer.max_chunks
                break

    @contextmanager
    def acquire_read(self, timeout: float | None = None):
        """Acquire the next block in the ring buffer to read from."""
        assert self._is_local_reader, "Only readers can acquire read"
        start_time = time.monotonic()
        n_warning = 1
        while True:
            with self.buffer.get_metadata(self.current_idx) as metadata_buffer:
                read_flag = metadata_buffer[self.local_reader_rank + 1]
                written_flag = metadata_buffer[0]
                if not written_flag or read_flag:
                    # this block is either
                    # (1) not written
                    # (2) already read by this reader

                    # for readers, `self.current_idx` is the next block to read
                    # if this block is not ready,
                    # we need to wait until it is written

                    # Release the processor to other threads
                    sched_yield()

                    # if we wait for a long time, log a message
                    if time.monotonic() - start_time > RINGBUFFER_FULL_WARNING_INTERVAL * n_warning:
                        logger.debug(
                            "No available block found in %s second. ",
                            RINGBUFFER_FULL_WARNING_INTERVAL,
                        )
                        n_warning += 1

                    # if we time out, raise an exception
                    if timeout is not None and time.monotonic() - start_time > timeout:
                        raise TimeoutError

                    continue
                # found a block that is not read by this reader
                # let caller read from the buffer
                with self.buffer.get_data(self.current_idx) as buf:
                    yield buf

                # caller has read from the buffer
                # set the read flag
                metadata_buffer[self.local_reader_rank + 1] = 1
                self.current_idx = (self.current_idx + 1) % self.buffer.max_chunks
                break

    def enqueue(self, obj, timeout: float | None = None):
        """Write to message queue.

        Args:
            obj: Object to be broadcasted
            timeout: Timeout in seconds. If None, wait indefinitely.
        """
        assert self._is_writer, "Only writers can enqueue"
        serialized_obj = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        if self.n_local_reader > 0:
            if len(serialized_obj) >= self.buffer.max_chunk_bytes:
                with self.acquire_write(timeout) as buf:
                    buf[0] = 1  # overflow
                self.local_socket.send(serialized_obj)
            else:
                with self.acquire_write(timeout) as buf:
                    buf[0] = 0  # not overflow
                    buf[1 : len(serialized_obj) + 1] = serialized_obj

    def dequeue(self, timeout: float | None = None):
        """Read from message queue.

        Args:
            timeout: Timeout in seconds. If None, wait indefinitely.
        """
        assert self._is_local_reader, "Only readers can dequeue"
        if self._is_local_reader:
            with self.acquire_read(timeout) as buf:
                overflow = buf[0] == 1
                if not overflow:
                    # no need to know the size of serialized object
                    # pickle format contains the size information internally
                    # see https://docs.python.org/3/library/pickle.html
                    obj = pickle.loads(buf[1:])
            if overflow:
                recv = self.local_socket.recv()
                obj = pickle.loads(recv)
        return obj  # type: ignore

    def broadcast_object(self, obj=None):
        """Helper to broadcast an object to all readers.

        Writer calls with object, readers call without object.
        """
        if self._is_writer:
            self.enqueue(obj)
            return obj
        else:
            return self.dequeue()
