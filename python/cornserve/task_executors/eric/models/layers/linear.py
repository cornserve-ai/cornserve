from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from cornserve.logging import get_logger
from cornserve.task_executors.eric.distributed.utils import divide, split_tensor_along_last_dim
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

    def forward(self,
                x: torch.Tensor) -> tuple[torch.Tensor, nn.Parameter | None]:
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
            torch.empty(sum(self.output_partition_sizes), self.input_size, dtype=params_dtype),
            requires_grad=False,
        )
        set_weight_attrs(self.weight, {"input_dim": 1, "output_dim": 0})

        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition, dtype=params_dtype))
            set_weight_attrs(self.bias, {"output_dim": 0})
        else:
            self.register_parameter("bias", None)

        self.register_load_state_dict_pre_hook(self.__class__._load_hook)

    def _load_hook(self: nn.Module, state_dict: dict[str, Any], prefix: str, *args) -> None:
        """State dict hook to narrow the weight tensor to the sharded size."""
        tp_rank = self.tp_rank
        for name, param in self.named_parameters(recurse=False):
            # Original weight in state dict
            weight_key = prefix + name
            weight = state_dict[weight_key]

            # TODO: Remove
            output_dim = getattr(param, "output_dim", None)
            if output_dim is None:
                continue

            # Shard the weight based on TP rank
            shard_size = weight.shape[output_dim]
            start_idx = tp_rank * shard_size
            sharded_weight = weight.narrow(output_dim, start_idx, shard_size)

            logger.info(
                "%s: Loading weight %s. Original shape %s narrowed to %s by slicing %d:%d along dim=%d",
                self.__class__.__name__,
                weight_key,
                weight.shape,
                sharded_weight.shape,
                start_idx,
                start_idx + shard_size,
                output_dim,
            )

            assert param.shape == sharded_weight.shape, (
                f"Weight shape mismatch: {param.shape=} != {sharded_weight.shape=}"
            )

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
            torch.empty(self.output_size, self.input_size_per_partition, dtype=params_dtype),
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
        tp_rank = self.tp_rank
        for name, param in self.named_parameters(recurse=False):
            # Original weight in state dict
            weight_key = prefix + name
            weight = state_dict[weight_key]

            input_dim = getattr(param, "input_dim", None)
            if input_dim is None:
                continue

            # Shard the weight based on TP rank
            shard_size = weight.shape[input_dim]
            start_idx = tp_rank * shard_size
            sharded_weight = weight.narrow(input_dim, start_idx, shard_size)

            logger.info(
                "%s: Loading weight %s. Original shape %s narrowed to %s by slicing %d:%d along dim=%d",
                self.__class__.__name__,
                weight_key,
                weight.shape,
                sharded_weight.shape,
                start_idx,
                start_idx + shard_size,
                input_dim,
            )

            assert param.shape == sharded_weight.shape, (
                f"Weight shape mismatch: {param.shape=} != {sharded_weight.shape=}"
            )

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


# class _ColumnParallelLinear(LinearBase):
#     """Linear layer with column parallelism.
#
#     The linear layer is defined as Y = XA + b. A is parallelized along
#     its second dimension as A = [A_1, ..., A_p].
#
#     Args:
#         input_size: first dimension of matrix A.
#         output_size: second dimension of matrix A.
#         bias: If true, add bias.
#         gather_output: If true, call all-gather on output and make Y available
#                        to all GPUs, otherwise, every GPU will have its output
#                        which is Y_i = XA_i
#         skip_bias_add: This was added to enable performance optimizations where
#                        bias can be fused with other element-wise operations. we
#                        skip adding bias but instead return it.
#         params_dtype: Data type for the parameters.
#         output_sizes: list of output sizes packed into one output, like for QKV
#                        the list would be size 3.
#         prefix: The name of the layer in the state dict, including all parents
#                         (e.g. model.layers.0.qkv_proj) 
#     """
#
#     def __init__(self,
#                  input_size: int,
#                  output_size: int,
#                  bias: bool = True,
#                  gather_output: bool = False,
#                  skip_bias_add: bool = False,
#                  params_dtype: Optional[torch.dtype] = None,
#                  quant_config: Optional[QuantizationConfig] = None,
#                  output_sizes: Optional[list[int]] = None,
#                  prefix: str = ""):
#         super().__init__(input_size, output_size, skip_bias_add, params_dtype,
#                          quant_config, prefix)
#
#         self.gather_output = gather_output
#
#         # Divide the weight matrix along the last dimension.
#         tp_size = get_tensor_model_parallel_world_size()
#         assert self.quant_method is not None
#         self.output_size_per_partition = divide(self.output_size, tp_size)
#         self.output_partition_sizes = [self.output_size_per_partition]
#         # If QKV or MergedColumn, use output size of each partition.
#         if hasattr(self, "output_sizes"):
#             self.output_partition_sizes = [
#                 divide(output_size, tp_size)
#                 for output_size in self.output_sizes
#             ]
#
#         if output_sizes is None:
#             output_sizes = [output_size]
#
#         self.quant_method.create_weights(
#             layer=self,
#             input_size_per_partition=self.input_size,
#             output_partition_sizes=self.output_partition_sizes,
#             input_size=self.input_size,
#             output_size=self.output_size,
#             params_dtype=self.params_dtype,
#             weight_loader=(
#                 self.weight_loader_v2 if self.quant_method.__class__.__name__
#                 in WEIGHT_LOADER_V2_SUPPORTED else self.weight_loader))
#         if bias:
#             self.bias = nn.Parameter(
#                 torch.empty(self.output_size_per_partition,
#                             dtype=params_dtype))
#             set_weight_attrs(self.bias, {
#                 "output_dim": 0,
#                 "weight_loader": self.weight_loader,
#             })
#         else:
#             self.register_parameter("bias", None)
#
#     def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
#         tp_rank = get_tensor_model_parallel_rank()
#         output_dim = getattr(param, "output_dim", None)
#
#         # Special case for GGUF
#         is_gguf_weight = getattr(param, "is_gguf_weight", False)
#         is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
#         if is_gguf_weight_type:
#             param.weight_type = loaded_weight.item()
#
#         # Materialize GGUF UninitializedParameter
#         if is_gguf_weight and isinstance(param, UninitializedParameter):
#             param.materialize(loaded_weight.shape, dtype=loaded_weight.dtype)
#
#         use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
#         is_sharded_weight = getattr(param, "is_sharded_weight", False)
#         # bitsandbytes loads the weights of the specific portion
#         # no need to narrow
#         is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit
#
#         param_data = param.data
#         if output_dim is not None and not is_sharded_weight:
#             shard_size = param_data.shape[output_dim]
#             start_idx = tp_rank * shard_size
#             loaded_weight = loaded_weight.narrow(output_dim, start_idx,
#                                                  shard_size)
#
#         # Special case for loading scales off disk, which often do not
#         # have a shape (such as in the case of AutoFP8).
#         if len(loaded_weight.shape) == 0:
#             loaded_weight = loaded_weight.reshape(1)
#
#         assert param_data.shape == loaded_weight.shape
#         param_data.copy_(loaded_weight)
#
#     def weight_loader_v2(self, param: nn.Parameter, loaded_weight: torch.Tensor):
#         # Special case for loading scales off disk, which often do not
#         # have a shape (such as in the case of AutoFP8).
#         if len(loaded_weight.shape) == 0:
#             assert loaded_weight.numel() == 1
#             loaded_weight = loaded_weight.reshape(1)
#         param.load_column_parallel_weight(loaded_weight=loaded_weight)
#
#     def forward(self, input_) -> tuple[torch.Tensor, Optional[nn.Parameter]]:
#         bias = self.bias if not self.skip_bias_add else None
#
#         # Matrix multiply.
#         assert self.quant_method is not None
#         output_parallel = self.quant_method.apply(self, input_, bias)
#         if self.gather_output:
#             # All-gather across the partitions.
#             output = tensor_model_parallel_all_gather(output_parallel)
#         else:
#             output = output_parallel
#         output_bias = self.bias if self.skip_bias_add else None
#         return output, output_bias
#
#
# class _RowParallelLinear(LinearBase):
#     """Linear layer with row parallelism.
#
#     The linear layer is defined as Y = XA + b. A is parallelized along
#     its first dimension and X along its second dimension as:
#                -   -
#               | A_1 |
#               | .   |
#           A = | .   |        X = [X_1, ..., X_p]
#               | .   |
#               | A_p |
#                -   -
#     Arguments:
#         input_size: first dimension of matrix A.
#         output_size: second dimension of matrix A.
#         bias: If true, add bias. Note that bias is not parallelized.
#         input_is_parallel: If true, we assume that the input is already
#                            split across the GPUs and we do not split
#                            again.
#         skip_bias_add: This was added to enable performance optimization where
#                        bias can be fused with other element-wise operations.
#                        We skip adding bias but instead return it.
#         params_dtype: Data type for the parameters.
#         quant_config: Quantization configure.
#     """
#
#     def __init__(self,
#                  input_size: int,
#                  output_size: int,
#                  bias: bool = True,
#                  input_is_parallel: bool = True,
#                  skip_bias_add: bool = False,
#                  params_dtype: Optional[torch.dtype] = None,
#                  reduce_results: bool = True,
#                  quant_config: Optional[QuantizationConfig] = None,
#                  prefix: str = ""):
#         super().__init__(input_size, output_size, skip_bias_add, params_dtype,
#                          quant_config, prefix)
#
#         self.input_is_parallel = input_is_parallel
#         self.reduce_results = reduce_results
#
#         # Divide the weight matrix along the last dimension.
#         self.tp_rank = get_tensor_model_parallel_rank()
#         self.tp_size = get_tensor_model_parallel_world_size()
#         self.input_size_per_partition = divide(input_size, self.tp_size)
#         assert self.quant_method is not None
#
#         self.quant_method.create_weights(
#             layer=self,
#             input_size_per_partition=self.input_size_per_partition,
#             output_partition_sizes=[self.output_size],
#             input_size=self.input_size,
#             output_size=self.output_size,
#             params_dtype=self.params_dtype,
#             weight_loader=(
#                 self.weight_loader_v2 if self.quant_method.__class__.__name__
#                 in WEIGHT_LOADER_V2_SUPPORTED else self.weight_loader))
#         if not reduce_results and (bias and not skip_bias_add):
#             raise ValueError("When not reduce the results, adding bias to the "
#                              "results can lead to incorrect results")
#
#         if bias:
#             self.bias = nn.Parameter(
#                 torch.empty(self.output_size, dtype=params_dtype))
#             set_weight_attrs(self.bias, {
#                 "output_dim": 0,
#                 "weight_loader": self.weight_loader,
#             })
#         else:
#             self.register_parameter("bias", None)
#
#     def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
#         tp_rank = get_tensor_model_parallel_rank()
#         tp_size = get_tensor_model_parallel_world_size()
#         input_dim = getattr(param, "input_dim", None)
#         use_bitsandbytes_4bit = getattr(param, "use_bitsandbytes_4bit", False)
#         is_sharded_weight = getattr(param, "is_sharded_weight", False)
#         # bitsandbytes loads the weights of the specific portion
#         # no need to narrow
#         is_sharded_weight = is_sharded_weight or use_bitsandbytes_4bit
#
#         # Special case for GGUF
#         is_gguf_weight = getattr(param, "is_gguf_weight", False)
#         is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)
#         if is_gguf_weight_type:
#             param.weight_type = loaded_weight.item()
#
#         # Materialize GGUF UninitializedParameter
#         if is_gguf_weight and isinstance(param, UninitializedParameter):
#             weight_shape = list(loaded_weight.shape)
#             if input_dim:
#                 weight_shape[input_dim] = weight_shape[input_dim] // tp_size
#             param.materialize(tuple(weight_shape), dtype=loaded_weight.dtype)
#
#         param_data = param.data
#         if input_dim is not None and not is_sharded_weight:
#             shard_size = param_data.shape[input_dim]
#             start_idx = tp_rank * shard_size
#             loaded_weight = loaded_weight.narrow(input_dim, start_idx,
#                                                  shard_size)
#
#         # Special case for loading scales off disk, which often do not
#         # have a shape (such as in the case of AutoFP8).
#         if len(loaded_weight.shape) == 0:
#             loaded_weight = loaded_weight.reshape(1)
#
#         assert param_data.shape == loaded_weight.shape
#         param_data.copy_(loaded_weight)
#
#     def weight_loader_v2(self, param: BasevLLMParameter,
#                          loaded_weight: torch.Tensor):
#
#         # Special case for loading scales off disk, which often do not
#         # have a shape (such as in the case of AutoFP8).
#         if len(loaded_weight.shape) == 0:
#             assert loaded_weight.numel() == 1
#             loaded_weight = loaded_weight.reshape(1)
#
#         param.load_row_parallel_weight(loaded_weight=loaded_weight)
#
#     def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, nn.Parameter | None]:
#         if self.input_is_parallel:
#             input_parallel = x
#         else:
#             tp_rank = get_tensor_model_parallel_rank()
#             splitted_input = split_tensor_along_last_dim(
#                 x, num_partitions=self.tp_size)
#             input_parallel = splitted_input[tp_rank].contiguous()
#
#         # Matrix multiply.
#         assert self.quant_method is not None
#         # Only fuse bias add into GEMM for rank 0 (this ensures that
#         # bias will not get added more than once in TP>1 case)
#         bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
#         output_parallel = self.quant_method.apply(self,
#                                                   input_parallel,
#                                                   bias=bias_)
#         if self.reduce_results and self.tp_size > 1:
#             output = tensor_model_parallel_all_reduce(output_parallel)
#         else:
#             output = output_parallel
#
#         output_bias = self.bias if self.skip_bias_add else None
#
#         return output, output_bias
#
#
