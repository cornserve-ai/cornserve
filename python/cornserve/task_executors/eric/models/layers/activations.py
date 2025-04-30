"""Activation functions."""

from __future__ import annotations

import torch
import torch.nn as nn


class QuickGELU(nn.Module):
    # https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py#L90
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


def get_act_fn(name: str) -> nn.Module:
    """Get an activation function by name."""
    match name.lower():
        case "gelu":
            return nn.GELU()
        case "quick_gelu":
            return QuickGELU()
        case "gelu_pytorch_tanh":
            return nn.GELU(approximate="tanh")
        case "relu":
            return nn.ReLU()
        case "silu":
            return nn.SiLU()
        case _:
            raise NotImplementedError(f"Activation function {name!r} is not implemented.")
