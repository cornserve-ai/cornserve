"""Test the sidecar server and client within the same node."""

import os
import uuid
import time
import random
import asyncio
from typing import Generator

import torch
import pytest
import multiprocessing

from cornserve.services.sidecar.api import (
    TensorSidecarSender,
    TensorSidecarAsyncReceiver,
)

torch.manual_seed(0)
random.seed(0)
MAX_SERVERS = int(os.environ.get("MAX_SERVERS", 4))


def mock_grpc_channel_from_rank(rank: int) -> str:
    """Mock version that maps a local channel to a rank."""
    assert rank >= 0, "Rank should be non-negative"
    return f"localhost:{10000 + rank}"


def mock_grpc_channel() -> None:
    """Mock the grpc_channel_from_rank function."""
    mocker = pytest.MonkeyPatch()
    mocker.setattr(
        "cornserve.services.sidecar.utils.grpc_channel_from_rank",
        mock_grpc_channel_from_rank,
    )
    mocker.setattr(
        "cornserve.services.sidecar.api.grpc_channel_from_rank",
        mock_grpc_channel_from_rank,
    )


def run_server(rank: int, world_size: int, shm_size: int) -> None:
    """Sidecar server entrypoint that will run in a subprocess."""
    mock_grpc_channel()

    from cornserve.services.sidecar.server import main

    # Set environment variables
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["SHM_SIZE"] = str(shm_size)

    asyncio.run(main())


def start_sidecar_servers(
    n: int = 4, shm_size: int = 2 << 28
) -> list[multiprocessing.Process]:
    """Start n sidecar servers in n processes."""
    processes = []
    for rank in range(n):
        process = multiprocessing.Process(
            target=run_server,
            args=(rank, n, shm_size),
        )
        process.start()
        processes.append(process)
    return processes


def terminate_processes(processes: list[multiprocessing.Process]) -> None:
    """Terminate all processes."""
    for process in processes:
        process.terminate()
        process.join(timeout=2)

        # Force kill if still running
        if process.is_alive():
            process.kill()
            process.join()


@pytest.fixture(scope="session", autouse=True)
def mock_grpc_channel_fixture() -> None:
    """Fixture to automatically mock the grpc_channel_from_rank function."""
    mock_grpc_channel()


@pytest.fixture(scope="session")
def sidecar_servers(
    request: pytest.FixtureRequest,
) -> Generator[list[multiprocessing.Process], None, None]:
    """Fixture to start sidecar servers for all tests."""
    shm_size = getattr(request, "param", 2 << 28)
    servers = start_sidecar_servers(MAX_SERVERS, shm_size)
    # Wait for servers to start up
    time.sleep(5)

    # Check all servers are alive
    for rank, server in enumerate(servers):
        assert server.is_alive(), f"Server with rank {rank} is not running"
    yield servers
    # Cleanup after all tests are done
    terminate_processes(servers)


@pytest.mark.asyncio
@pytest.mark.parametrize("sidecar_servers", [2 << 26], indirect=True)
async def test_sidecar_liveness(sidecar_servers: list[multiprocessing.Process]):
    """Test n sidecar servers can launch and each can be registered."""
    from cornserve.services.sidecar.api import (
        TensorSidecarSender,
        TensorSidecarAsyncReceiver,
    )

    for rank in range(MAX_SERVERS):
        _ = TensorSidecarSender(
            sidecar_rank=rank, slot_shape=(5,), dtype=torch.bfloat16
        )
        _ = TensorSidecarAsyncReceiver(
            sidecar_rank=rank, gpu_rank=rank, shape=(-1, 5), dtype=torch.bfloat16
        )


# fmt: off
@pytest.mark.asyncio
@pytest.mark.parametrize("sidecar_servers", [2 << 26], indirect=True)
@pytest.mark.parametrize(
    "sender_n, receiver_n, num_chunks, shape, dtype, token_min, token_max",
    [
        (1, 1, 1, (100,), torch.float64, 5, 10),  # single sender and receiver
        (2, 1, 1, (100,), torch.float64, 5, 10),  # multi sender with imbalanced shards
        (2, 2, 1, (100,), torch.float64, 1, 1),  # multi sender-receiver with corner case
        (1, 1, 1, (1176,), torch.bfloat16, 1, 50000),  # qwen2-vl
        (1, 1, 1, (1176,), torch.bfloat16, 40000, 50000),  # qwen2-vl, tp=2 with back pressure
        (2, 1, 1, (1176,), torch.bfloat16, 40000, 50000),  # qwen2-vl, tp=2 with back pressure
        (1, 1, 4, (576, 4096), torch.bfloat16, 1, 1),  # fixed-resolution ViT
        (2, 1, 4, (4096, 2), torch.bfloat16, 1601, 1601),  # fixed-resolution 2 with back pressure
    ],
)
# fmt: on
async def test_send_recv(
    sidecar_servers: list[multiprocessing.Process],
    sender_n: int,
    receiver_n: int,
    num_chunks: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    token_min: int,
    token_max: int,
    count: int = 3,
):
    """Test sending and receiving tensors between n senders and n receivers.

    Args:
      sender_n: Number of senders, this is the same as tp_size
      receiver_n: Number of receivers
      num_chunks: Number of chunks in each data transfer is split into
      shape: The shape of a token or some multiple of a token that is used as share memory slot shape (size)
      dtype: The data type of the tensor
      token_min: the minimum number of tokens (of `shape`) that will use, should be 1 to simulate fixed resolution ViT
      token_max: the maximum number of tokens (of `shape`) that will use, should be 1 to simulate fixed resolution ViT
    """
    assert sidecar_servers is not None, "Servers fixture should be available"
    await asyncio.sleep(1)
    senders: list[TensorSidecarSender] = []
    for i in range(sender_n):
        senders.append(
            TensorSidecarSender(
                sidecar_rank=i,
                slot_shape=shape,
                dtype=dtype,
                shard_rank=i,
                num_shards=sender_n,
            )
        )

    receivers = []
    for i in range(sender_n, sender_n + receiver_n):
        receivers.append(
            TensorSidecarAsyncReceiver(
                sidecar_rank=i, gpu_rank=i, shape=(-1, *shape), dtype=dtype
            )
        )

    ids = []
    data = []
    for _ in range(count):
        chunks = []
        id = f"{_}" * 32 + uuid.uuid4().hex
        ids.append(id)
        n = random.randint(token_min, token_max)
        for i in range(num_chunks):
            chunk = torch.randn(n, *shape, dtype=dtype)
            chunks.append(chunk)
            for j, sender in enumerate(senders):
                print(f"-->Sender {j} sending chunk {i}")
                sender.send(
                    chunk=chunk.to(f"cuda:{j}"),
                    id=id,
                    dst_sidecar_ranks=list(range(sender_n, sender_n + receiver_n)),
                    chunk_id=i,
                    num_chunks=num_chunks,
                )

        sent = torch.cat(chunks, dim=0)
        data.append(sent)

    print(f"Queued up sending {count} data, now queueing up receiving")

    async def verify(
        k: int,
        i: int,
        id: str,
        receiver: TensorSidecarAsyncReceiver,
        data: list[torch.Tensor],
        sender_n: int,
    ):
        received = await receiver.recv(id=id)
        sent = data[k].to(f"cuda:{i+sender_n}")
        received = received.to(f"cuda:{i+sender_n}")
        assert torch.allclose(sent, received)
        await receiver.mark_done(id=id)

    futures = []
    for k, id in enumerate(ids):
        for i, receiver in enumerate(receivers):
            future = verify(k, i, id, receiver, data, sender_n)
            futures.append(future)

    # we gather them all to avoid live lock from memory fragmentation
    await asyncio.gather(*futures)

    for receiver in receivers:
        await receiver.shutdown()

    await asyncio.sleep(1)
