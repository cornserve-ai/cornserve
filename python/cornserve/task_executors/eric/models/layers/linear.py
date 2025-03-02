from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from cornserve.logging import get_logger
from cornserve.task_executors.eric.distributed.utils import (
    divide,
    split_tensor_along_last_dim,
)
from cornserve.task_executors.eric.distributed.parallel import get_tensor_parallel_group

logger = get_logger(__name__)


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: dict[str, Any] | None,
):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(weight, key), f"Overwriting existing tensor attribute: {key}"
        setattr(weight, key, value)


class LinearBase(nn.Module):
    """Base linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        prefix: str = "",
    ):
        super().__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.tp_group = get_tensor_parallel_group()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, nn.Parameter | None]:
        raise NotImplementedError


class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        output_sizes: list[int] | None = None,
    ) -> None:
        """Initialize the layer.

        Args:
            input_size: first dimension of matrix A.
            output_size: second dimension of matrix A.
            bias: If true, add bias.
            gather_output: If true, call all-gather on output and make Y available
                to all GPUs, otherwise, every GPU will have its output
                which is Y_i = XA_i
            skip_bias_add: This was added to enable performance optimizations where
                bias can be fused with other element-wise operations. we skip adding
                bias but instead return it.
            params_dtype: Data type for the parameters.
            output_sizes: list of output sizes packed into one output, like for QKV
                the list would be size 3.
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype or torch.get_default_dtype()
        self.gather_output = gather_output

        self.tp_group = get_tensor_parallel_group()
        self.tp_rank = self.tp_group.rank
        self.tp_size = self.tp_group.world_size
        self.output_size_per_partition = divide(self.output_size, self.tp_size)
        self.output_partition_sizes = [self.output_size_per_partition]

        if output_sizes is None:
            output_sizes = [output_size]

        self.weight = nn.Parameter(
            torch.empty(
                sum(self.output_partition_sizes), self.input_size, dtype=params_dtype
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.weight, {"input_dim": 1, "output_dim": 0})

        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.output_size_per_partition, dtype=params_dtype)
            )
            set_weight_attrs(self.bias, {"output_dim": 0})
        else:
            self.register_parameter("bias", None)

        self.register_load_state_dict_pre_hook(self.__class__._load_hook)

    def _load_hook(self, state_dict: dict[str, Any], prefix: str, *args) -> None:
        """State dict hook to narrow the weight tensor to the sharded size."""
        for name, param in self.named_parameters(recurse=False):
            # Original weight in state dict
            weight_key = prefix + name
            weight = state_dict[weight_key]

            # TODO: Remove
            output_dim = getattr(param, "output_dim", None)
            if output_dim is None:
                continue

            # Shard the weight based on TP rank
            shard_size = divide(weight.shape[output_dim], self.tp_size)
            start_idx = self.tp_rank * shard_size
            sharded_weight = weight.narrow(output_dim, start_idx, shard_size)

            logger.debug(
                "%s: Loading weight %s. Original shape %s narrowed to %s by slicing %d:%d along dim=%d",
                self.__class__.__name__,
                weight_key,
                weight.shape,
                sharded_weight.shape,
                start_idx,
                start_idx + shard_size,
                output_dim,
            )

            assert (
                param.shape == sharded_weight.shape
            ), f"Weight shape mismatch: {param.shape=} != {sharded_weight.shape=}"

            # Set the sharded weight in the state dict
            # When the hook exits, this weight will be loaded into the parameter
            state_dict[weight_key] = sharded_weight

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, nn.Parameter | None]:
        """Run forward."""
        bias = self.bias if not self.skip_bias_add else None

        output = F.linear(x, self.weight, bias)
        if self.gather_output and self.tp_size > 1:
            output = self.tp_group.all_gather(output)

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
              -------
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
              -------
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = True,
    ) -> None:
        """Initialize the layer.

        Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        skip_bias_add: This was added to enable performance optimization where
                       bias can be fused with other element-wise operations.
                       We skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        reduce_results: If true, reduce the results across the GPUs.
                        Otherwise, each GPU will have its output.
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype or torch.get_default_dtype()
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results

        self.tp_group = get_tensor_parallel_group()
        self.tp_rank = self.tp_group.rank
        self.tp_size = self.tp_group.world_size
        self.input_size_per_partition = divide(input_size, self.tp_size)

        self.weight = nn.Parameter(
            torch.empty(
                self.output_size, self.input_size_per_partition, dtype=params_dtype
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.weight, {"input_dim": 1, "output_dim": 0})

        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size, dtype=params_dtype))
            set_weight_attrs(self.bias, {"output_dim": 0})
        else:
            self.register_parameter("bias", None)

        self.register_load_state_dict_pre_hook(self.__class__._load_hook)

    def _load_hook(self, state_dict: dict[str, Any], prefix: str, *args) -> None:
        """State dict hook to narrow the weight tensor to the sharded size."""
        for name, param in self.named_parameters(recurse=False):
            # Original weight in state dict
            weight_key = prefix + name
            weight = state_dict[weight_key]

            input_dim = getattr(param, "input_dim", None)
            if input_dim is None:
                continue

            # Shard the weight based on TP rank
            shard_size = divide(weight.shape[input_dim], self.tp_size)
            start_idx = self.tp_rank * shard_size
            sharded_weight = weight.narrow(input_dim, start_idx, shard_size)

            logger.debug(
                "%s: Loading weight %s. Original shape %s narrowed to %s by slicing %d:%d along dim=%d",
                self.__class__.__name__,
                weight_key,
                weight.shape,
                sharded_weight.shape,
                start_idx,
                start_idx + shard_size,
                input_dim,
            )

            assert (
                param.shape == sharded_weight.shape
            ), f"Weight shape mismatch: {param.shape=} != {sharded_weight.shape=}"

            # Set the sharded weight in the state dict
            # When the hook exits, this weight will be loaded into the parameter
            state_dict[weight_key] = sharded_weight

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, nn.Parameter | None]:
        """Run forward."""
        if self.input_is_parallel:
            input_parallel = x
        else:
            splitted_input = split_tensor_along_last_dim(x, num_partitions=self.tp_size)
            input_parallel = splitted_input[self.tp_rank].contiguous()

        # Matrix multiply.
        # Only fuse bias add into GEMM for rank 0 (this ensures that
        # bias will not get added more than once in TP>1 case)
        bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
        output = F.linear(input_parallel, self.weight, bias_)
        if self.reduce_results and self.tp_size > 1:
            output = self.tp_group.all_reduce(output)

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
