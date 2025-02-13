import asyncio
import torch
from cornserve.services.sidecar.api import TensorSidecarReceiver
from cornserve.logging import get_logger
from cornserve.services.utils import tensor_hash
import os

logger = get_logger(__name__)


async def main() -> None:
    RANK = int(os.environ.get("RANK", 3))
    NUM_SHARDS = int(os.environ.get("NUM_SHARDS", 1))
    dtype = torch.bfloat16
    shape = (4, 1601, 4096)

    sidecar = TensorSidecarReceiver(
        sidecar_rank=RANK,
        shape=shape,
        dtype=dtype,
    )
    device = torch.device(f"cuda:{RANK}")
    logger.info(f"Starting encoder server using device {device} on rank {RANK}")

    tensor = torch.rand(shape, device=device, dtype=dtype)

    lock = asyncio.Lock()

    async def recv_and_process_req(id: int) -> None:
        received_tensor = await sidecar.async_recv(id)
        async with lock:
            tensor.copy_(received_tensor)
            if NUM_SHARDS == 1:
                logger.info(
                    f"Received tensor with req id {id} with hash {tensor_hash(received_tensor)} of shape {received_tensor.shape}"
                )
            else:
                shards = torch.chunk(received_tensor, NUM_SHARDS, dim=0)
                for i, shard in enumerate(shards):
                    # this is to check the hash of each shard
                    logger.info(
                        f"Received shard {i} in req {id} with hash {tensor_hash(shard)} of shape {shard.shape} with sum {shard.sum()}"
                    )
        await sidecar.async_mark_done(id)
        # process the tensor

    id = 0
    for _ in range(5):
        all_tasks = []
        logger.info(f"start to receive 3 requests starting from {id}")
        for _ in range(3):
            all_tasks.append(asyncio.create_task(recv_and_process_req(id)))
            id += 1
        logger.info(f"waiting for 3 requests starting from {id}")
        await asyncio.gather(*all_tasks)
        await asyncio.sleep(10)
    sidecar.unregister()


if __name__ == "__main__":
    asyncio.run(main())
