import torch
from cornserve.services.sidecar.api import TensorSidecarSender
from cornserve.logging import get_logger
from cornserve.services.utils import tensor_hash
import os
import random
import time

logger = get_logger(__name__)


def main() -> None:
    RANK = int(os.environ.get("RANK", 0))
    SHARD_RANK = int(os.environ.get("SHARD_RANK", 0))
    NUM_SHARDS = int(os.environ.get("NUM_SHARDS", 1))

    DST_RANK = 3

    dtype = torch.bfloat16
    tensor_shape = (4 // NUM_SHARDS, 1601, 4096)
    num_chunks = 4 // NUM_SHARDS
    chunk_shape = (1601, 4096)

    sidecar = TensorSidecarSender(
        sidecar_rank=RANK,
        chunk_shape=chunk_shape,
        dtype=dtype,
        shard_rank=SHARD_RANK,
        num_shards=NUM_SHARDS,
    )
    logger.info(f"Connected to sidecar-{RANK}")

    device = torch.device(f"cuda:{RANK}")
    logger.info(f"Starting encoder server using device {device} on sidecar_rank {RANK}")

    id = 0
    # single tensor
    for _ in range(15):
        if NUM_SHARDS == 1:
            dummy_tensor = torch.rand(tensor_shape, device=device, dtype=dtype)
        else:
            num_elements = tensor_shape[0] * tensor_shape[1] * tensor_shape[2]
            if SHARD_RANK == 0:
                dummy_tensor = torch.arange(
                    1, num_elements + 1, device=device, dtype=dtype
                ).reshape(tensor_shape)
            else:
                dummy_tensor = torch.arange(
                    1 + num_elements, 2 * num_elements + 1, device=device, dtype=dtype
                ).reshape(tensor_shape)

        logger.info(
            f"Sending tensor with req id {id} with hash {tensor_hash(dummy_tensor)} of shape {dummy_tensor.shape} with sum {dummy_tensor.sum()}"
        )
        for i in range(num_chunks):
            chunk = dummy_tensor[i]
            logger.info(f"Sending chunk {i}")
            sidecar.send(
                chunk=chunk,
                req_id=id,
                chunk_id=i,
                num_chunks=num_chunks,
                dst_sidecar_rank=DST_RANK,
            )

        time.sleep(random.randint(1, 10))
        id += 1
    sidecar.unregister()


if __name__ == "__main__":
    main()
