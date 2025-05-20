import asyncio
import multiprocessing
import os
import signal
import time

import grpc
import pytest
import torch

from cornserve.services.pb import sidecar_pb2, sidecar_pb2_grpc
from cornserve.sidecar.utils import grpc_url_from_rank


def run_server(rank: int, world_size: int, local_peer_ranks: list[int], shm_size: int) -> None:
    """Sidecar server entrypoint that will run in a subprocess."""
    mock_device()

    # Set environment variables
    os.environ["SIDECAR_RANK"] = str(rank)
    os.environ["SIDECAR_WORLD_SIZE"] = str(world_size)
    os.environ["SIDECAR_LOCAL_PEER_RANKS"] = ",".join(map(str, local_peer_ranks))
    os.environ["SIDECAR_SHM_SIZE"] = str(shm_size)
    os.environ["SIDECAR_IS_LOCAL"] = "true"

    from cornserve.services.sidecar.server import cleanup_coroutines, main

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.run_until_complete(asyncio.gather(*cleanup_coroutines))
        loop.close()


def device_from_rank(rank: int) -> torch.device:
    """Get the device for a given rank."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    return torch.device("cpu")


def mock_device() -> None:
    mocker = pytest.MonkeyPatch()
    mocker.setattr(
        "cornserve.sidecar.utils.device_from_rank",
        device_from_rank,
    )


def start_sidecar_servers(
    n: int = 4,
    cluster_size: int = 2,
    shm_size: int = 2 << 28,
) -> list[multiprocessing.Process]:
    """Start n sidecar servers in n processes."""
    processes = []
    ctx = multiprocessing.get_context("spawn")
    for rank in range(n):
        cluster_start = (rank // cluster_size) * cluster_size
        cluster_ranks = list(range(cluster_start, cluster_start + cluster_size))
        print("Starting sidecar server of rank", rank, "with cluster ranks", cluster_ranks)
        process = ctx.Process(
            target=run_server,
            args=(rank, n, cluster_ranks, shm_size),
        )
        process.start()
        processes.append(process)
    return processes


def server_is_online(stub: sidecar_pb2_grpc.SidecarStub) -> bool:
    """Check if the server is running."""
    try:
        req = sidecar_pb2.CheckHealthRequest()
        res = stub.CheckHealth(req)
        return res.status == sidecar_pb2.HealthStatus.HEALTH_ALL_GOOD
    except grpc.RpcError:
        return False


def wait_for_servers_to_start(rank: int) -> None:
    while True:
        with grpc.insecure_channel(grpc_url_from_rank(rank)) as channel:
            stub = sidecar_pb2_grpc.SidecarStub(channel)
            if server_is_online(stub):
                break
            else:
                time.sleep(10.2)


def terminate_processes(processes: list[multiprocessing.Process]) -> None:
    """Terminate all processes."""

    for process in processes:
        if process.pid:
            os.kill(process.pid, signal.SIGINT)

    for process in processes:
        process.join(timeout=5)
        if not process.is_alive():
            continue
        process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join()
