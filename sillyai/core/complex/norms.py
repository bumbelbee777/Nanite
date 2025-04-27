import math
import torch
import torch.nn as nn

class ComplexLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm_real = nn.LayerNorm(dim)
        self.norm_imag = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [..., 2]
        real = self.norm_real(x[..., 0])
        imag = self.norm_imag(x[..., 1])
        return torch.stack([real, imag], dim=-1)