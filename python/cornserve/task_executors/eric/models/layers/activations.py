import torch
import torch.nn as nn


class QuickGELU(nn.Module):
    # https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py#L90
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)
