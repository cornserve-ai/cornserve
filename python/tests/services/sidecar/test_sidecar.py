import asyncio
import multiprocessing
import os
import random
from typing import Any, Generator, Literal
import uuid

import pytest
import torch

from .utils import (
    device_from_rank,
    mock_device,
    mock_grpc_channel,
    mock_ucx_url,
    start_sidecar_servers,
    terminate_processes,
    wait_for_servers_to_start,
)

torch.manual_seed(0)
random.seed(0)
uuid.uuid4 = lambda: uuid.UUID(int=random.getrandbits(128))

MAX_SERVERS = int(os.environ.get("MAX_SERVERS", 4))
CLUSTER_SIZE = int(os.environ.get("CLUSTER_SIZE", 2))


@pytest.fixture(scope="module", autouse=True)
def mock_grpc_channel_fixture() -> None:
    """Fixture to automatically mock the grpc_channel_from_rank function."""
    mock_grpc_channel()


@pytest.fixture(scope="module", autouse=True)
def mock_ucx_url_fixture() -> None:
    """Fixture to automatically mock the ucx_url_from_rank function."""
    mock_ucx_url()


@pytest.fixture(scope="module", autouse=True)
def mock_device_fixture() -> None:
    """Fixture to automatically mock the device_from_rank function."""
    mock_device()


@pytest.fixture(scope="module")
def sidecar_servers(request: pytest.FixtureRequest) -> Generator[tuple[list[multiprocessing.Process], Literal["intranode", "internode"]], None, None]:
    """Parameterized fixture for both intranode and internode server setups.
    
    Returns:
        A tuple of (servers, setup_type) where setup_type is either "intranode" or "internode"
    """
    setup_type = request.param
    shm_size = 2 << 26  # Fixed value from original tests
    
    if setup_type == "internode":
        servers = start_sidecar_servers(MAX_SERVERS, CLUSTER_SIZE, shm_size)
    else:  # intranode
        servers = start_sidecar_servers(MAX_SERVERS, MAX_SERVERS, shm_size)
    for rank, server in enumerate(servers):
        assert server.is_alive(), f"Server with rank {rank} is not running"
        wait_for_servers_to_start(rank)
    
    yield (servers, setup_type)
    
    # Cleanup after all tests are done
    terminate_processes(servers)


obj_test_params = [
    ([0], [2], ["string", b"bytes", 5, 10.6, False]),
    ([0,1], [2,3], ["another string", b"different bytes", 100, 5/3, True]),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("sidecar_servers", ["intranode", "internode"], indirect=True)
@pytest.mark.parametrize("sender_ranks, receiver_ranks, objects", obj_test_params)
async def test_objs(
    sidecar_servers: tuple[list[multiprocessing.Process], Literal["intranode", "internode"]],
    sender_ranks: list[int],
    receiver_ranks: list[int],
    objects: list[Any],
):
    """Test sending and receiving objects between senders and receivers.
    Works with both intranode and internode setups.
    """
    servers, setup_type = sidecar_servers
    assert servers is not None, "Servers fixture should be available"
    assert len(objects), "Test objects should be provided"

    from cornserve.sidecar.api import Sidecar
    from cornserve.sidecar.schema import SidecarConfig

    print(f"------------------------------------------------------------")
    print(f"Testing {setup_type} object communication:")
    print(f"Sender ranks: {sender_ranks} Receiver ranks: {receiver_ranks}")

    sidecar_senders = []
    sidecar_receivers = []

    for r in sender_ranks:
        config = SidecarConfig(
            sidecar_rank=r,
            group=sender_ranks,
            send_tensor_shape=(-1, 5),
            send_tensor_dtype=torch.bfloat16,
        )
        sidecar_senders.append(Sidecar(config))

    for r in receiver_ranks:
        config = SidecarConfig(
            sidecar_rank=r,
            group=receiver_ranks,
            recv_tensor_shape=(-1, 5),
            recv_tensor_dtype=torch.bfloat16,
        )
        sidecar_receivers.append(Sidecar(config))

    def send_all(senders: list[Sidecar], receiver_ranks: list[int]) -> list[str]:
        ids = []
        for obj in objects:
            id = uuid.uuid4().hex
            ids.append(id)
            for sender in senders:
                print(f"--> Sender {sender.sidecar_rank} sending {type(obj)} object {obj} with id {id} to {receiver_ranks}")
                sender.send(id=id, data=obj, dst_sidecar_ranks=[receiver_ranks])
        return ids

    async def verify(k: int, ids: list[str], receivers: list[Sidecar]):
        for receiver in receivers:
            received = await receiver.recv(id=ids[k])
            print(f"--> Receiver {receiver.sidecar_rank} received {type(received)} object {received}")
            sent = objects[k]
            assert sent == received
        await receivers[0].mark_done(id=ids[k])

    async def verify_all(ids: list[str], receivers: list[Sidecar]):
        futures = []
        for k in range(len(objects)):
            future = verify(k, ids, receivers)
            futures.append(future)
        await asyncio.gather(*futures)

    # sender => receiver
    ids = send_all(sidecar_senders, receiver_ranks)
    await verify_all(ids, sidecar_receivers)

    print("------------------------------------------------------------")

    # receiver => sender
    ids = send_all(sidecar_receivers, sender_ranks)
    await verify_all(ids, sidecar_senders)

    for sender in sidecar_senders:
        await sender.shutdown()
    for receiver in sidecar_receivers:
        await receiver.shutdown()

tensor_test_params = [
    # Single copy
    ([0], [2], 1, (5,), torch.bfloat16, 5, 5, False, 5),
    ([0, 1], [2], 2, (20,), torch.bfloat16, 5, 10, False, 6),
    ([0, 1], [2, 3], 4, (100,), torch.float64, 100, 200, False, 7),
    # Concurrent copy
    ([0], [2], 4, (5,), torch.bfloat16, 100, 200, True, 8),
    ([0, 1], [2], 2, (20,), torch.bfloat16, 50, 100, True, 9),
    ([0, 1], [2, 3], 1, (100,), torch.float64, 5, 50, True, 10),
]
@pytest.mark.asyncio
@pytest.mark.parametrize("sidecar_servers", ["intranode", "internode"], indirect=True)
@pytest.mark.parametrize(
    "sender_ranks, receiver_ranks, num_chunks, shape, dtype, token_min, token_max, concurrent_copy, count",
    tensor_test_params
)
async def test_tensors(
    sidecar_servers: tuple[list[multiprocessing.Process], Literal["intranode", "internode"]],
    sender_ranks: list[int],
    receiver_ranks: list[int],
    num_chunks: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    token_min: int,
    token_max: int,
    concurrent_copy: bool,
    count: int,
):
    """Test sending and receiving tensors between senders and receivers.
    Works with both intranode and internode setups.
    """
    servers, setup_type = sidecar_servers
    assert servers is not None, "Servers fixture should be available"

    from cornserve.sidecar.api import Sidecar
    from cornserve.sidecar.schema import SidecarConfig

    print("------------------------------------------------------------")
    print(
        f"Testing {setup_type} tensor communication:\n",
        f"Sender ranks: {sender_ranks} Receiver ranks: {receiver_ranks}\n",
        f"num_chunks: {num_chunks} shape: {shape} dtype: {dtype} token_min: {token_min} token_max: {token_max}",
    )

    sidecar_senders = []
    sidecar_receivers = []

    for r in sender_ranks:
        config = SidecarConfig(
            sidecar_rank=r,
            group=sender_ranks,
            send_tensor_shape=(-1, *shape),
            send_tensor_dtype=dtype,
            concurrent_copy=concurrent_copy,
        )
        sidecar_senders.append(Sidecar(config))

    for r in receiver_ranks:
        config = SidecarConfig(
            sidecar_rank=r,
            group=receiver_ranks,
            recv_tensor_shape=(-1, *shape),
            recv_tensor_dtype=dtype,
            concurrent_copy=concurrent_copy,
        )
        sidecar_receivers.append(Sidecar(config))


    def send_all(
        senders: list[Sidecar],
        receiver_ranks: list[int],
    ) -> tuple[list[str], list[list[torch.Tensor]]]:
        ids = []
        data = []
        for _ in range(count):
            chunks = []
            id = uuid.uuid4().hex
            ids.append(id)
            n = random.randint(token_min, token_max)
            for i in range(num_chunks):
                print(f"--> Sender {sender_ranks} sending chunk {i} of data_id {id} with {n} tokens")
                chunk = torch.randn(n, *shape, dtype=dtype)
                chunks.append(chunk)
                for r, sender in enumerate(senders):
                    sender.send(
                        id=id,
                        data=chunk.to(device_from_rank(r)),
                        dst_sidecar_ranks=[receiver_ranks],
                        chunk_id=i,
                        num_chunks=num_chunks,
                    )
                print("TEST: Sent data id", id, "chunk", i, "of tensor", chunk)
            data.append(chunks)
        return ids, data


    async def verify(
        id: str,
        data: list[torch.Tensor],
        sender_rank: int,
        receivers: list[Sidecar],
    ):
        for i in range(num_chunks):
            for receiver in receivers:
                received = await receiver.recv(id=id, chunk_id=i)
                sent = data[i].to(device_from_rank(sender_rank))
                received = received.to(device_from_rank(sender_rank))
                if i == 0:
                    print("TEST: Received data id", id, " of tensor", received)
                assert torch.allclose(sent, received), f"Data mismatch for id {id}: {sent} vs {received}"
            await receivers[0].mark_done(id=id, chunk_id=i)

    async def verify_all(
        ids: list[str],
        data: list[list[torch.Tensor]],
        sender_ranks: list[int],
        receivers: list[Sidecar],
    ):
        futures = []
        for k, id in enumerate(ids):
            future = verify(id, data[k], sender_ranks[0], receivers)
            futures.append(future)

        # we gather them all to avoid live lock from memory fragmentation
        await asyncio.gather(*futures)

    # sender => receiver
    ids, data = send_all(sidecar_senders, receiver_ranks)
    await verify_all(ids, data, sender_ranks, sidecar_receivers)

    print("------------------------------------------------------------")

    # receiver => sender
    ids, data = send_all(sidecar_receivers, sender_ranks)
    await verify_all(ids, data, receiver_ranks, sidecar_senders)

    for sender in sidecar_senders:
        await sender.shutdown()
    for receiver in sidecar_receivers:
        await receiver.shutdown()
