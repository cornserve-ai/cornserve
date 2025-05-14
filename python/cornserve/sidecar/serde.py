"""Utilities for serializing and deserializing objects."""

import pickle
from typing import Any

import torch
from msgspec import msgpack
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

CUSTOM_TYPE_GPU_TENSOR = 1
CUSTOM_TYPE_SHARED_TENSOR_HANDLE = 2
CUSTOM_TYPE_PICKLE = 3
CUSTOM_TYPE_FORWARD_TENSOR_HANDLE = 4


# CANNOT BE @dataclass
class SharedTensorHandle:
    """Shared memory handle for a tensor used for intra-node DataForward."""

    def __init__(self, offset: int, numel: int) -> None:
        """Initialize the SharedTensorHandle.

        Args:
            offset: Offset of the shared buffer from the full tensor.
            numel: Number of elements in the tensor.
        """
        self.offset = offset
        self.numel = numel

    def __repr__(self) -> str:
        """Return a string representation of the SharedTensorHandle."""
        return f"SharedTensorHandle(offset={self.offset}, numel={self.numel})"


# CANNOT BE @dataclass
class ForwardTensorHandle:
    """Tensor handle for used for inter-node DataForward."""

    def __init__(self, total_numel: int, shard_rank: int, num_shards: int) -> None:
        """Initialize the ForwardTensorHandle.

        Args:
            total_numel: Total number of elements in the tensor.
            shard_rank: Shard rank of current forward.
            num_shards: Total number of shards.
        """
        self.total_numel = total_numel
        self.shard_rank = shard_rank
        self.num_shards = num_shards

    def __repr__(self) -> str:
        """Return a string representation of the ForwardTensorHandle."""
        return (
            f"ForwardTensorHandle(total_numel={self.total_numel}, "
            f"shard_rank={self.shard_rank}, num_shards={self.num_shards})"
        )


class MsgpackEncoder:
    """Msgpack encoder that implements custom serialization."""

    def __init__(self) -> None:
        """Initialize the encoder."""
        self.encoder = msgpack.Encoder(enc_hook=enc_hook)

    def encode(self, obj: Any) -> bytes:
        """Encode an object to bytes."""
        return self.encoder.encode(obj)

    def encode_into(self, obj: Any, buffer: bytearray) -> None:
        """Encode an object into a buffer."""
        self.encoder.encode_into(obj, buffer)


class MsgpackDecoder:
    """Msgpack decoder that implements custom deserialization."""

    def __init__(self) -> None:
        """Initialize the decoder."""
        self.decoder = msgpack.Decoder(ext_hook=ext_hook)

    def decode(self, data: bytes) -> Any:
        """Decode bytes to an object."""
        return self.decoder.decode(data)


def enc_hook(obj: Any) -> Any:
    """Use pickle to serialize Numpy arrays."""
    if isinstance(obj, torch.Tensor) and obj.is_cuda:
        # Torch GPU tensors are serialized as IPC handles
        _, rebuild_args = reduce_tensor(obj)
        return msgpack.Ext(
            CUSTOM_TYPE_GPU_TENSOR,
            pickle.dumps(rebuild_args, protocol=pickle.HIGHEST_PROTOCOL),
        )

    if isinstance(obj, SharedTensorHandle):
        return msgpack.Ext(
            CUSTOM_TYPE_SHARED_TENSOR_HANDLE,
            pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL),
        )

    if isinstance(obj, ForwardTensorHandle):
        return msgpack.Ext(
            CUSTOM_TYPE_FORWARD_TENSOR_HANDLE,
            pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL),
        )

    return msgpack.Ext(CUSTOM_TYPE_PICKLE, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def ext_hook(code: int, data: memoryview) -> Any:
    """Use pickle to deserialize Numpy arrays."""
    if code == CUSTOM_TYPE_GPU_TENSOR:
        # Torch tensors are deserialized as Numpy arrays.
        rebuild_args = pickle.loads(data)
        return rebuild_cuda_tensor(*rebuild_args)
    if code == CUSTOM_TYPE_SHARED_TENSOR_HANDLE:
        return pickle.loads(data)
    if code == CUSTOM_TYPE_FORWARD_TENSOR_HANDLE:
        return pickle.loads(data)
    if code == CUSTOM_TYPE_PICKLE:
        return pickle.loads(data)

    raise ValueError(f"Unknown custom serialization code: {code}")
