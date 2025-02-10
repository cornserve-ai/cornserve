import asyncio
import torch
from cornserve.services.sidecar.api import TensorSidecar, SidecarMode
from cornserve.logging import get_logger
from cornserve.services.utils import tensor_hash
import os

logger = get_logger(__name__)


async def main() -> None:
    RANK = int(os.environ.get("RANK", 0))
    DST_RANK = 3
    dtype = torch.bfloat16
    shape = (4, 1601, 4096)

    sidecar = TensorSidecar(
        mode=SidecarMode.SEND,
        shape=shape,
        dtype=dtype,
        rank=RANK,
    )
    device = torch.device(f"cuda:{RANK}")
    logger.info(f"Starting encoder server using device {device} on rank {RANK}")

    id = 10 * RANK

    for _ in range(10):
        dummy_tensor = torch.rand(shape, device=device, dtype=dtype)
        logger.info(
            f"Sending tensor with req id {id} with hash {tensor_hash(dummy_tensor)} of shape {dummy_tensor.shape}"
        )
        await sidecar.send(dummy_tensor, [id], [DST_RANK], [0])
        await asyncio.sleep(5)
        id += 1


if __name__ == "__main__":
    asyncio.run(main())
