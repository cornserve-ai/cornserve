""" An example sender that instatiates a TensorSidecarSender. """
import os
import random
import time
import hashlib

from cornserve.logging import get_logger
from cornserve.services.sidecar.api import TensorSidecarSender
from cornserve.services.utils import tensor_hash
import torch

logger = get_logger(__name__)
torch.manual_seed(0)
random.seed(0)

def fake_uuid(i: int) -> str:
    """Generate a fake UUID for testing purposes."""
    return str(i) * 32 + hashlib.sha256(f'{i}'.encode('utf-8')).hexdigest()[:32]

def main() -> None:
    TP_SIZE = int(os.environ.get("TP_SIZE", 1))
    NUM_CHUNKS = int(os.environ.get("NUM_CHUNKS", 1))
    DST_RANK = int(os.environ.get("DST_RANK", 3))
    n = int(os.environ.get("N", 1))

    if NUM_CHUNKS == 1:
        slot_shape = (1176,)
        num_tokens_range = (100,5000)
    else:
        slot_shape = (4096,)
        num_tokens_range = (576,576)
    dtype = torch.bfloat16

    sidecar_senders = []
    devices = []
    for i in range(TP_SIZE):
        sidecar_senders.append(TensorSidecarSender(
            sidecar_rank=i,
            slot_shape=slot_shape,
            dtype=dtype,
            shard_rank=i,
            num_shards=TP_SIZE,
        ))
        devices.append(torch.device(f"cuda:{i}"))
        logger.info("TP worker %d: Connected to sidecar-%d", i, i)

    for i in range(n):
        id = fake_uuid(i)
        for c in range(NUM_CHUNKS):
            data_shape = (random.randint(*num_tokens_range),*slot_shape)
            data = torch.rand(data_shape, dtype=dtype)
            for j, sidecar_sender in enumerate(sidecar_senders):
                device = devices[j]
                data = data.to(device)
                logger.info("TP worker %d: Sending chunk %d of request %s with hash %s",j, c, id, tensor_hash(data))
                sidecar_sender.send(
                    chunk=data,
                    id=id,
                    dst_sidecar_ranks=[DST_RANK],
                    chunk_id=c,
                    num_chunks=NUM_CHUNKS,
                )
    time.sleep(30)


if __name__ == "__main__":
    main()
