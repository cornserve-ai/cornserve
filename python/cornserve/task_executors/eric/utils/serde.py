import pickle
from typing import Any, Type

import torch
import numpy as np
from msgspec import msgpack

CUSTOM_TYPE_NUMPY = 1
CUSTOM_TYPE_TORCH = 2
CUSTOM_TYPE_PICKLE = 3


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

    def __init__(self, ty: Type | None = None) -> None:
        """Initialize the decoder."""
        self.decoder = msgpack.Decoder(type=ty, ext_hook=ext_hook)

    def decode(self, data: bytes) -> Any:
        """Decode bytes to an object."""
        return self.decoder.decode(data)


def enc_hook(obj: Any) -> Any:
    """Use pickle to serialize Numpy arrays.

    https://gist.github.com/tlrmchlsmth/8067f1b24a82b6e2f90450e7764fa103
    """
    if isinstance(obj, np.ndarray):
        return msgpack.Ext(
            CUSTOM_TYPE_NUMPY, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        )
    if isinstance(obj, torch.Tensor):
        # Torch tensors are serialized as Numpy arrays.
        return msgpack.Ext(
            CUSTOM_TYPE_TORCH,
            pickle.dumps(obj.numpy(), protocol=pickle.HIGHEST_PROTOCOL),
        )

    return msgpack.Ext(
        CUSTOM_TYPE_PICKLE, pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    )


def ext_hook(code: int, data: memoryview) -> Any:
    """Use pickle to deserialize Numpy arrays."""
    if code == CUSTOM_TYPE_NUMPY:
        return pickle.loads(data)
    if code == CUSTOM_TYPE_TORCH:
        # Torch tensors are deserialized as Numpy arrays.
        return torch.from_numpy(pickle.loads(data))
    if code == CUSTOM_TYPE_PICKLE:
        return pickle.loads(data)

    raise ValueError(f"Unknown custom serialization code: {code}")
