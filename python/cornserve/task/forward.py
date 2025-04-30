"""Classes for representing data forwarding between tasks in the data plane."""

from __future__ import annotations

import enum
import uuid
from typing import Generic, Self, TypeVar

from pydantic import BaseModel, Field, model_validator


class Tensor:
    """Represents a tensor object for data forwarding."""


class ForwardableType(enum.StrEnum):
    """Types of data that can be forwarded between tasks."""

    BYTES = "bytes"
    STR = "str"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    TENSOR = "Tensor"


DataT = TypeVar("DataT")


class DataForward(BaseModel, Generic[DataT]):
    """Represents data that is forwarded between tasks in the data plane."""

    # This ID identifies `DataForward` objects and ties them together in task input/outputs.
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)

    # The data type automatically parsed out of the generic type argument.
    data_type: ForwardableType = Field(init=False, default=ForwardableType.TENSOR)

    # Producer (source) sidecar ranks.
    src_sidecar_ranks: list[int] | None = Field(init=False, default=None)

    # Consumer (destination) sidecar ranks. This is a list of lists because the data
    # can be forwarded to multiple tasks (i.e., broadcasted) to more than one task executor.
    dst_sidecar_ranks: list[list[int]] | None = Field(init=False, default=None)

    @model_validator(mode="after")
    def _data_type(self) -> Self:
        """Validate the generic type argument.

        1. The generic type argument must be present.
        2. It should be one of the forwardable types (`ForwardableType`).
        """
        metadata = self.__class__.__pydantic_generic_metadata__
        if metadata["origin"] is None:
            raise ValueError("Generic type argument is missing.")

        args = metadata["args"]
        assert len(args) == 1, "If origin was not None, there should be exactly one argument."
        self.data_type = ForwardableType(args[0].__name__)

        return self
