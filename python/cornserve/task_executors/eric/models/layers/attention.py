"""Generic attention implementation for ViTs."""

from __future__ import annotations

import torch
import torch.nn as nn
from xformers.ops import memory_efficient_attention_forward  # type: ignore


class Attention(nn.Module):
    """Full attention implementation for ViTs."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.num_queries_per_kv, rem = divmod(self.num_heads, self.num_kv_heads)
        assert rem == 0, f"{num_heads=} must be divisible by {num_kv_heads=}"

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Run forward.

        Args:
            query, key, value: [batch_size, seq_len, hidden_size]
        """
        bsz, q_len, _ = query.size()
        kv_len = key.size(1)

        query = query.view(bsz, q_len, self.num_heads, self.head_size)
        key = key.view(bsz, kv_len, self.num_kv_heads, self.head_size)
        value = value.view(bsz, kv_len, self.num_kv_heads, self.head_size)

        if (num_repeat := self.num_queries_per_kv) > 1:
            # Handle MQA and GQA
            key = torch.repeat_interleave(key, num_repeat, dim=2)
            value = torch.repeat_interleave(value, num_repeat, dim=2)

        out = memory_efficient_attention_forward(query, key, value, scale=self.scale)

        return out.reshape(bsz, q_len, -1)
