import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import InfiniToeplitz
from .complex.linear import ComplexLinear
from .complex.norms import ComplexLayerNorm
from .routing import FeatureRouter

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create position matrix: (max_len, 1)
        pos = torch.arange(max_len).float().unsqueeze(1)
        
        # Create div_term for efficient computation
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term[:d_model//2])
        pe[:, 1::2] = torch.cos(pos * div_term[:d_model//2])
        
        # Register as buffer
        self.register_buffer('pe', pe)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model, 2)
        Returns:
            Complex positional encoding of shape (1, seq_len, d_model, 2)
        """
        seq_len = x.size(1)
        # Add zero imaginary component to positional encoding 
        pe = torch.stack([
            self.pe[:seq_len],
            torch.zeros_like(self.pe[:seq_len])
        ], dim=-1)
            
        return pe.unsqueeze(0)  # Add batch dimension
    
class DynamicActivation(nn.Module):
    def __init__(self, dim: int, hidden_size: int = None):
        super().__init__()
        self.dim = dim
        h = hidden_size or (dim // 8)
        
        # Complex adaptation parameters
        self.adaptation_rate = nn.Parameter(torch.tensor(0.1, dtype=torch.cfloat))
        
        # Complex-aware GRU cells
        self.gru_mag = nn.GRUCell(2, h)
        self.gru_phase = nn.GRUCell(2, h)
        
        # Complex transformation layers with kronecker_rank=4 (small since these are 1D projections)
        self.out_mag = nn.Sequential(
            nn.ReLU(),
            ComplexLinear(h, 1, factorized=True, kronecker_rank=4),
            nn.Sigmoid()
        )
        self.out_phase = nn.Sequential(
            nn.ReLU(),
            ComplexLinear(h, 1, factorized=True, kronecker_rank=4),
            nn.Tanh()
        )
        
        # Complex mixing parameters
        self.mag_mix = nn.Parameter(torch.ones(1, dtype=torch.cfloat))
        self.phase_mix = nn.Parameter(torch.ones(1, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Complex input tensor of shape (B, L, D, 2)
        Returns:
            Activated complex tensor of same shape
        """
        B, L, D, _ = x.shape
        
        # Convert to polar form
        xr, xi = x.unbind(-1)
        mag = torch.sqrt(xr**2 + xi**2 + 1e-6)  # (B, L, D)
        phase = torch.atan2(xi, xr)  # (B, L, D)
        
        # Complex statistics
        stats = torch.stack([mag.mean(dim=-1), phase.mean(dim=-1)], dim=-1)
        stats = stats.view(B*L, 2)
        
        # Process through complex GRU pathways
        h_mag = self.gru_mag(stats, torch.zeros(B*L, self.gru_mag.hidden_size, device=x.device))
        h_ph = self.gru_phase(stats, torch.zeros(B*L, self.gru_phase.hidden_size, device=x.device))
        
        # Generate complex adjustments
        mag_adj = self.out_mag(h_mag).view(B, L, D, 1)
        phase_adj = self.out_phase(h_ph).view(B, L, D, 1)
        
        # Apply complex transformations
        new_mag = mag.unsqueeze(-1) * (1 + (mag_adj * self.mag_mix * self.adaptation_rate)).real
        new_phase = phase.unsqueeze(-1) + (phase_adj * self.phase_mix * self.adaptation_rate).real
        
        # Convert back to complex
        real = new_mag * torch.cos(new_phase)
        imag = new_mag * torch.sin(new_phase)
        
        return torch.stack([real.squeeze(-1), imag.squeeze(-1)], dim=-1)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear1 = ComplexLinear(
            config.d_model, 
            config.dim_ff, 
            factorized=config.factorized_linear,
            kronecker_rank=config.kronecker_rank
        )
        self.linear2 = ComplexLinear(
            config.dim_ff, 
            config.d_model, 
            factorized=config.factorized_linear,
            kronecker_rank=config.kronecker_rank
        )
        self.activation = DynamicActivation(config.dim_ff, config.hidden_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.attn = InfiniToeplitz(
            embed_dim=config.d_model,
            num_heads=config.nhead,
            factorized=config.factorized_linear,
            chunk_size=getattr(config, 'infini_local_window', 512),
            mem_gamma=0.9
        )

        self.ff = FeedForward(config)
        self.norm1 = ComplexLayerNorm(config.d_model)
        self.norm2 = ComplexLayerNorm(config.d_model)

    def forward(self, x):
        # Shape handling is done in attention module
        x_attn = self.attn(x)
        x = self.norm1(x + x_attn)
        x = self.norm2(x + self.ff(x))
        return x