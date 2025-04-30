"""Linear layers with tensor parallelism."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import register_module_module_registration_hook

from cornserve.logging import get_logger
from cornserve.task_executors.eric.utils.distributed import (
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

        # XXX: output_sizes can be removed entirely and output_size is enough?
        if output_sizes is None:
            self.output_partition_sizes = [self.output_size_per_partition]
        else:
            self.output_partition_sizes = [divide(output_size, self.tp_size) for output_size in output_sizes]
        # assert sum(self.output_partition_sizes) == self.output_size

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

    def _load_hook(self, state_dict: dict[str, Any], prefix: str, *args) -> None:
        """State dict hook to narrow the weight tensor to the sharded size."""
        for name, param in self.named_parameters(recurse=False):
            # Original weight in state dict
            weight_key = prefix + name
            weight = state_dict[weight_key]

            # XXX: Unnecessary? Remove and replace with zero?
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


class QKVParallelLinear(ColumnParallelLinear):
    """Fused QKV parallel linear layer.

    This layer is used when the Q, K, V linear weights are stored separately
    in the model's state dict into one weight tensor. Weight matrices are
    concatenated along the output dimension, and parallelized along the head
    dimension.

    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned. For instance,

    Full model definition:
    - Q heads: 16
    - KV heads: 4 (0, 1, 2, 3)

    Parallelized with TP size 8, each TP rank will have:
    - Q head: 2 (partitioned)
    - KV head: 1 (replicated twice; 0, 0, 1, 1, 2, 2, 3, 3)

    Args:
        hidden_size: Input hidden state size of the transformer.
        head_size: Size of each attention head.
        total_num_heads: Total number of attention query heads.
        total_num_kv_heads: Total number of attention key/value heads. If
            None, defaults to total_num_heads.
        bias: If true, add bias.
        skip_bias_add: This was added to enable performance optimizations where
            bias can be fused with other element-wise operations. we skip adding
            bias but instead return it.
        params_dtype: Data type for the parameters.
        gather_from_names: Names of the QKV nn.Linear layers to gather from.
            Expected to be in the order of Q, K, and V.
    """

    registered_model_hook: bool = False

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        gather_from_names: tuple[str, str, str] = ("q_proj", "k_proj", "v_proj"),
    ) -> None:
        """Initialize the layer."""
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads

        self.tp_group = get_tensor_parallel_group()
        self.tp_rank = self.tp_group.rank
        self.tp_size = self.tp_group.world_size

        self.num_heads = divide(total_num_heads, self.tp_size)
        if self.tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(self.tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, self.tp_size)
            self.num_kv_head_replicas = 1

        input_size = hidden_size
        output_size = (self.num_heads + 2 * self.num_kv_heads) * self.tp_size * head_size
        # XXX: Seems unnecessary; sum is equivalent to output_size.
        self.output_sizes = [
            self.num_heads * head_size * self.tp_size,  # q_proj
            self.num_kv_heads * head_size * self.tp_size,  # k_proj
            self.num_kv_heads * head_size * self.tp_size,  # v_proj
        ]

        self.qkv_to_module_name: dict[Literal["q", "k", "v"], str] = {
            "q": gather_from_names[0],
            "k": gather_from_names[1],
            "v": gather_from_names[2],
        }

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            output_sizes=self.output_sizes,
        )

        # Our state dict has separate entries for Q, K, and V nn.Linear layers,
        # e.g., `q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `q_proj.bias`,
        # `k_proj.bias`, `v_proj.bias`. On the other hand, our model definition
        # uses a single nn.Module for fused QKV projection:
        #
        #     class VisionAttention(nn.Module):
        #         def __init__(self, ...):
        #             self.qkv_proj = QKVParallelLinear(...)
        #
        # If we simply call `load_state_dict` on our model, the separate Q, K,
        # and V parameters will not be matched with the fused QKV parameters in
        # `qkv_proj`. This mismatch fundamentally has to be dealt with in the
        # parent module -- `VisionAttention` in this case -- on `load_state_dict`
        # where the parent module fuses the separate Q, K, and V parameters into
        # something that can be loaded into `qkv_proj`.
        #
        # Whenever a `QKVParallelLinear` object is instantiated, we know that it
        # is going to be used somewhere in our model. Thus, we register a global
        # hook that's called whenever `register_module` is called on `nn.Module`.
        # This hook will check whether there's a `QKVParallelLinear` module
        # being registered into a parent module, and if so, register a hook
        # that runs prior to `load_state_dict` on the parent module. This hook
        # will be responsible for fusing the separate Q, K, and V parameters and
        # updating the state dict so that it can be loaded into `qkv_proj`.
        #
        # Note that the instantiation of `QKVParallelLinear` precedes the call to
        # `register_module` on the parent module, so the hook will be registered
        # *just in time* before the first instantiation of `QKVParallelLinear`.
        def module_register_hook(parent: nn.Module, name: str, module: nn.Module):
            """Register load state dict hook on module containing QKVParallelLinear."""
            if isinstance(module, QKVParallelLinear):
                parent.register_load_state_dict_pre_hook(
                    module.get_parent_load_state_dict_pre_hook(name),
                )

        # Register the hook only once.
        if not QKVParallelLinear.registered_model_hook:
            register_module_module_registration_hook(module_register_hook)
            QKVParallelLinear.registered_model_hook = True

    def get_parent_load_state_dict_pre_hook(self, registered_name: str):
        """Get load state dict pre hook for parent module.

        Args:
            registered_name: Name of the QKVParallelLinear module in the parent.
        """

        def shard_concat(
            weights: dict[Literal["q", "k", "v"], torch.Tensor],
        ) -> torch.Tensor:
            """Shard Q, K, and V weights in the head dimension and concatenate."""
            if len(weights) != 3:
                raise ValueError(
                    f"Expected parameters for q, k, and v but got {'nothing' if not weights else weights.keys()}"
                )

            # If there are less K and V heads than TP degree, replicate them.
            for role in ("k", "v"):
                if self.num_kv_head_replicas > 1:
                    logger.debug(
                        "Replicating %s head %d times for TP size %d",
                        role,
                        self.num_kv_head_replicas,
                        self.tp_size,
                    )
                    weights[role] = weights[role].repeat_interleave(self.num_kv_head_replicas, dim=0)

            fused_weights = []
            for q, k, v in zip(
                weights["q"].chunk(self.tp_size, dim=0),
                weights["k"].chunk(self.tp_size, dim=0),
                weights["v"].chunk(self.tp_size, dim=0),
                strict=True,
            ):
                assert q.shape[0] == k.shape[0] == v.shape[0] == self.num_heads * self.head_size
                fused_weights.append(q)
                fused_weights.append(k)
                fused_weights.append(v)

            return torch.cat(fused_weights, dim=0)

        def hook(parent: nn.Module, state_dict: dict[str, Any], prefix: str, *args) -> None:
            """State dict hook to fuse Q, K, and V weights."""
            # Pop out individual Q, K, and V weights from state dict.
            weights: dict[Literal["q", "k", "v"], torch.Tensor] = {}
            biases: dict[Literal["q", "k", "v"], torch.Tensor] = {}
            for name in list(state_dict.keys()):
                for role, gather_name in self.qkv_to_module_name.items():
                    if name.startswith(prefix + gather_name):
                        if name.endswith("weight"):
                            weights[role] = state_dict.pop(name)
                        elif name.endswith("bias"):
                            biases[role] = state_dict.pop(name)
                        else:
                            raise ValueError(f"Expected {name} to end with either 'weight' or 'bias'")

            logger.debug(
                "Found %s weights and %s biases in state dict for %s",
                list(weights.keys()),
                "no" if not biases else list(biases.keys()),
                registered_name,
            )

            # Fuse Q, K, and V weights and biases and update state dict.
            fused_weights = shard_concat(weights)
            state_dict[prefix + registered_name + ".weight"] = fused_weights
            logger.debug(
                "Weight: Q (%s), K (%s), V (%s) -> QKV (%s)",
                weights["q"].shape,
                weights["k"].shape,
                weights["v"].shape,
                fused_weights.shape,
            )

            if self.bias is not None:
                fused_bias = shard_concat(biases)
                state_dict[prefix + registered_name + ".bias"] = fused_bias
                logger.debug(
                    "Bias: Q (%s), K (%s), V (%s) -> QKV (%s)",
                    biases["q"].shape,
                    biases["k"].shape,
                    biases["v"].shape,
                    fused_bias.shape,
                )

        return hook


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
