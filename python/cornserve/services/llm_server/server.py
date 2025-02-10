import asyncio
import torch
from cornserve.services.sidecar.api import TensorSidecar, SidecarMode
from cornserve.logging import get_logger
from cornserve.services.utils import tensor_hash
import os

logger = get_logger(__name__)


async def main() -> None:
    RANK = int(os.environ.get("RANK", 3))
    dtype = torch.bfloat16
    shape = (4, 1601, 4096)

    sidecar = TensorSidecar(
        mode=SidecarMode.RECV,
        shape=shape,
        dtype=dtype,
        rank=RANK,
    )
    device = torch.device(f"cuda:{RANK}")
    logger.info(f"Starting encoder server using device {device} on rank {RANK}")

    id = 0

    while True:
        tensor = await sidecar.recv([id])
        # log the product of the tensor
        logger.info(
            f"Received tensor for req id {id} with hash {tensor_hash(tensor)} of shape {tensor.shape}"
        )
        id += 1


if __name__ == "__main__":
    asyncio.run(main())
