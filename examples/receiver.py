""" An example receiver that instatiates a TensorSidecarAsyncReceiver. """
import asyncio
import hashlib
import os
import random

from cornserve.logging import get_logger
from cornserve.services.sidecar.api import TensorSidecarAsyncReceiver
from cornserve.services.utils import tensor_hash
import torch

logger = get_logger(__name__)
torch.manual_seed(0)
random.seed(0)

def fake_uuid(i: int) -> str:
    """Generate a fake UUID for testing purposes."""
    return str(i) * 32 + hashlib.sha256(f'{i}'.encode('utf-8')).hexdigest()[:32]

async def main() -> None:
    RANK = int(os.environ.get("RANK", 3))
    CHUNKING = bool(os.environ.get("CHUNKING", False))
    n = int(os.environ.get("N", 1))

    if CHUNKING:
        slot_shape = (-1, 576, 4096,)
    else:
        slot_shape = (-1, 1176,)
    dtype = torch.bfloat16

    sidecar_receiver = TensorSidecarAsyncReceiver(
        sidecar_rank=RANK,
        gpu_rank=RANK,
        shape=slot_shape,
        dtype=dtype,
        )
    device = torch.device(f"cuda:{RANK}")
    logger.info("Starting encoder server using device %s on rank %d", device, RANK)

    for i in range(n):
        id = fake_uuid(i)
        data = await sidecar_receiver.recv(id)
        logger.info(f"Received request %s with hash %s of shape %s", id, tensor_hash(data), data.shape)

    await sidecar_receiver.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
