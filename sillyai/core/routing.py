import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .complex.linear import ComplexLinear

class TaskComplexityEstimator(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.d_model
        self.probe = nn.Linear(embed_dim * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Same as current complexity estimator logic
        magnitudes = torch.norm(x, dim=-1)
        mean_magnitudes = torch.mean(magnitudes, dim=1)
        var_magnitudes = torch.var(magnitudes, dim=1)
        features = torch.cat([mean_magnitudes, var_magnitudes], dim=-1)
        complexity = self.probe(features)
        return torch.sigmoid(2.0 * complexity).squeeze(-1)  # [B]

    def complexity(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

class FeatureRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(config.d_model, 1)
        self.input_proj = ComplexLinear(config.d_model, config.d_model, factorized=config.factorized_linear, kronecker_rank=config.kronecker_rank)
        self.output_proj = ComplexLinear(config.d_model, config.d_model, factorized=config.factorized_linear, kronecker_rank=config.kronecker_rank)

    def forward(self, x: torch.Tensor, pos_enc: nn.Module, complexity: torch.Tensor) -> torch.Tensor:
        # Apply positional encoding conditionally based on feature importance
        encoded = pos_enc(x)

        # Project input and extract real parts for attention
        projected = self.input_proj(x)
        x_real = projected[..., 0]

        # Compute attention scores
        attn_output, _ = self.attention(x_real, x_real, x_real)

        # Scale positional encoding by attention
        scale = torch.sigmoid(attn_output).unsqueeze(-1)

        # Adjust scale based on complexity
        adjusted_scale = scale * complexity.unsqueeze(-1).unsqueeze(-1)

        # Apply the adjusted scale
        encoded = encoded * adjusted_scale

        # Project output
        return self.output_proj(encoded)
