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
        self.probe = nn.Linear(embed_dim * 2, 1)  # Doubled feature size to capture both variance and magnitude
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, L, D, 2] (complex-valued sequence)
        # Compute magnitude of complex features
        magnitudes = torch.norm(x, dim=-1)  # [B, L, D]
        
        # Compute sequence-level magnitude features
        mean_magnitudes = torch.mean(magnitudes, dim=1)  # [B, D]
        
        # Compute variance features to capture complexity
        var_magnitudes = torch.var(magnitudes, dim=1)  # [B, D]
        
        # Concatenate both features
        features = torch.cat([mean_magnitudes, var_magnitudes], dim=-1)  # [B, D*2]
        
        # Estimate complexity score with both magnitude and variance information
        complexity = self.probe(features)  # [B, 1]
        
        # Scale to [0,1] with steeper sigmoid for better separation
        return torch.sigmoid(2.0 * complexity).squeeze(-1)  # [B]

class FeatureRouter(nn.Module):
    """Routes features through different pathways based on complexity"""
    def __init__(self, config):
        super().__init__()
        d_model = config.d_model
        self.attention = nn.MultiheadAttention(d_model, 1)
        # Add projection layers with proper kronecker_rank
        self.input_proj = ComplexLinear(
            d_model, 
            d_model, 
            factorized=config.factorized_linear,
            kronecker_rank=config.kronecker_rank
        )
        self.output_proj = ComplexLinear(
            d_model, 
            d_model,
            factorized=config.factorized_linear,
            kronecker_rank=config.kronecker_rank
        )
        
    def forward(self, x: torch.Tensor, pos_enc: nn.Module) -> torch.Tensor:
        # Apply positional encoding conditionally based on feature importance
        encoded = pos_enc(x)  # [B, L, D, 2]
        
        # Project input and extract real parts for attention
        projected = self.input_proj(x)
        x_real = projected[..., 0]  # [B, L, D]
        
        # Compute attention scores
        attn_output, _ = self.attention(x_real, x_real, x_real)
        
        # Scale positional encoding by attention
        scale = torch.sigmoid(attn_output).unsqueeze(-1)  # [B, L, D, 1]
        encoded = encoded * scale
        
        # Project output
        return self.output_proj(encoded)