import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import fft
from typing import Optional, List, Dict, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import weakref
import math

from .ops import *
from .config import ModelConfig

def complex_checkpoint(function, *args):
    """Checkpointing wrapper that handles complex tensors"""
    # Convert complex inputs to real/imag pairs
    real_args = []
    for arg in args:
        if torch.is_tensor(arg) and arg.is_complex():
            real_args.extend([arg.real, arg.imag])
        else:
            real_args.append(arg)
    
    # Define a wrapper that reconstructs complex tensors
    def wrapper(*real_args_flat):
        reconstructed_args = []
        i = 0
        for arg in args:
            if torch.is_tensor(arg) and arg.is_complex():
                real_part = real_args_flat[i]
                imag_part = real_args_flat[i+1]
                reconstructed_args.append(torch.complex(real_part, imag_part))
                i += 2
            else:
                reconstructed_args.append(real_args_flat[i])
                i += 1
        return function(*reconstructed_args)
    
    # Apply checkpointing to the wrapper
    return torch.utils.checkpoint.checkpoint(wrapper, *real_args, use_reentrant=False)

class TaskComplexityEstimator(nn.Module):
    def __init__(
        self,
        threshold: float = 1e-3,
        eps: float = 1e-6,
        feature_dim: int = 32,
        use_fft: bool = True
    ):
        """
        threshold: any multivector component with |value| > threshold
                   counts as "active" toward complexity.
        """
        super().__init__()
        self.threshold = threshold
        self.eps       = eps
        self.use_fft   = use_fft
        from .config import ModelConfig
        from .ops import MultivectorOps
        # Minimal config for feature_net: input_dim=4, mlp_dim=feature_dim, output_dim=1
        feature_config = ModelConfig(input_dim=4, mlp_dim=feature_dim, output_dim=1, n_heads=1, n_layers=1, dropout=0.0)
        self.feature_net = ComplexMLP(feature_config, MultivectorOps())
        # JIT compile expensive computations
        self.get_spectral_norm = torch.jit.script(self._get_spectral_norm)
        self.get_rank_estimate = torch.jit.script(self._get_rank_estimate)
    
    @torch.jit.script
    def _get_spectral_norm(self, x: torch.Tensor, eps: float) -> torch.Tensor:
        """Estimate spectral norm using power iteration"""
        if x.ndim < 2:
            return torch.ones(1, device=x.device)
        
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        feat_size = x_flat.shape[1]
        
        # Initialize u randomly on correct device
        u = torch.randn((feat_size, 1), device=x.device)
        u_norm = torch.norm(u) + eps
        u = u / u_norm
        
        # Power iteration with fixed steps
        v: Optional[torch.Tensor] = None
        for _ in range(3):  # Usually 3 iterations is sufficient
            # v = x @ u
            v = torch.matmul(x_flat, u)
            v_norm = torch.norm(v) + eps
            v = v / v_norm
            
            # u = x.T @ v
            u = torch.matmul(x_flat.transpose(0, 1), v)
            u_norm = torch.norm(u) + eps
            u = u / u_norm
        
        # Final power iteration estimate
        if v is None:  # Satisfy TorchScript type checking
            v = torch.matmul(x_flat, u)
        
        final_term = torch.matmul(x_flat, u)
        result = torch.abs(torch.sum(v * final_term, dim=0))
        return result
    
    @torch.jit.script 
    def _get_rank_estimate(self, x: torch.Tensor, eps: float) -> torch.Tensor:
        """Estimate effective rank using eigenvalue decomposition"""
        if x.ndim < 2:
            return torch.ones(1, device=x.device)
            
        # Reshape and handle large matrices with projection
        x_flat = x.reshape(x.shape[0], -1)
        feat_size = x_flat.shape[1]
        
        if feat_size > 128:  # Random projection for large matrices
            proj = torch.randn(feat_size, 128, device=x.device)
            proj_norm = torch.norm(proj, dim=0, keepdim=True) + eps
            proj = proj / proj_norm
            x_flat = torch.matmul(x_flat, proj)
            
        # Compute Gram matrix for stability
        gram = torch.matmul(x_flat.transpose(0, 1), x_flat)
        gram = gram / (x_flat.shape[0] + eps)  # Normalize by batch size
        
        # Get eigenvalues through SVD
        s = torch.linalg.svdvals(gram)
        s = torch.clamp(s, min=eps)  # Ensure numerical stability
        
        # Normalize singular values to probability distribution
        total = torch.sum(s) + eps
        s = s / total
        
        # Compute entropy-based rank estimate
        entropy = -torch.sum(s * torch.log(s + eps))
        rank_estimate = torch.exp(entropy) / x_flat.shape[1]
        
        return rank_estimate
    
    def forward(self, mv: torch.Tensor, ops: Optional[MultivectorOps] = None) -> torch.Tensor:
        """
        Estimate task complexity from complex multivector input
        Returns: [B, L] or [B] complexity scores in [0,1]
        """
        if ops is None:
            ops = MultivectorOps()
            
        with torch.no_grad():
            # Get base features using scripted functions
            spectral = self.get_spectral_norm(mv, self.eps)
            rank = self.get_rank_estimate(mv, self.eps)
            
            # Compute sparsity using ops for complex magnitude
            mv_abs = ops.dot_product(mv, mv).sqrt()
            sparsity = torch.mean((mv_abs > self.threshold).float(), dim=-1)
            
            # Optional frequency domain features using ops
            if self.use_fft and mv.dim() > 2:
                fft_result = ops.fft_sync(mv)
                freq_feat = torch.mean(ops.dot_product(fft_result, fft_result).sqrt(), dim=-1)
            else:
                freq_feat = torch.zeros_like(spectral)
            
            # Combine features in complex domain
            features = torch.stack([
                spectral, rank, sparsity, freq_feat
            ], dim=-1).to(torch.complex64)
            
            # Get final complexity score using complex MLP
            scores = self.feature_net(features, ops).real.sigmoid()
            return torch.clamp(scores, 0.0, 1.0)  # [B, L]

class FeatureRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        threshold: float = 0.5,
        estimator: Optional[TaskComplexityEstimator] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden_dim or (d_model // 2)
        self.threshold = threshold
        self.estimator = estimator or TaskComplexityEstimator()
        from .config import ModelConfig
        from .ops import MultivectorOps
        # Minimal config for router: input_dim=2, mlp_dim=hidden, output_dim=1
        router_config = ModelConfig(input_dim=2, mlp_dim=self.hidden, output_dim=1, n_heads=1, n_layers=1, dropout=0.0)
        self.complex_mlp = ComplexMLP(router_config, MultivectorOps())

    async def forward(self, mv: torch.Tensor, ops: Optional[MultivectorOps] = None) -> torch.Tensor:
        """
        mv: Complex tensor of shape [B, L, D] or [B, L, D, mv_dim]
        Returns: Complex tensor of shape [B, L, 1]
        """
        if ops is None:
            ops = MultivectorOps()
            
        # Get complexity estimation
        with torch.no_grad():
            comp_score = self.estimator(mv, ops)  # [B, L]
            # Get magnitude using ops
            mag = ops.dot_product(mv, mv).sqrt().mean(dim=-1)  # [B, L]
            # Stack features as complex tensor
            feats = torch.stack([comp_score, mag], dim=-1).to(torch.complex64)  # [B, L, 2]

        # Route using complex MLP
        gate = self.complex_mlp(feats, ops)  # [B, L, 1]
        return gate

    def forward_sync(self, mv: torch.Tensor, ops: Optional[MultivectorOps] = None) -> torch.Tensor:
        if ops is None:
            ops = MultivectorOps()
            
        comp_score = self.estimator(mv, ops)
        mag = ops.dot_product(mv, mv).sqrt().mean(dim=-1)
        feats = torch.stack([comp_score, mag], dim=-1).to(torch.complex64)
        return self.complex_mlp(feats, ops)

class ComplexDropout(nn.Module):
    def __init__(self, p=0.1, seed=42):
        super().__init__()
        self.p = p
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        
        # Use the real part shape for dropout mask size
        real_tensor = x.real if x.is_complex() else x
        
        # Create dropout mask with generator to keep reproducibility local
        mask = (torch.rand(real_tensor.shape, generator=self.generator, device=real_tensor.device) > self.p).to(real_tensor.dtype)
        
        if x.is_complex():
            # Scale mask to maintain expectation
            scale = 1 / (1 - self.p)
            real = x.real * mask * scale
            imag = x.imag * mask * scale
            return torch.complex(real, imag)
        else:
            return x * mask / (1 - self.p)

    def _make_mask(self, x_real):
        cpu_rng_state = torch.get_rng_state()
        torch.manual_seed(self.seed)
        mask = (torch.rand_like(x_real) > self.p).to(x_real.dtype)
        torch.set_rng_state(cpu_rng_state)
        return mask

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # Initialize with the correct normalized shape - just normalize over the last dimension
        self.ln_r = nn.LayerNorm(dim, eps=eps)  # Changed to normalize over last dimension only
        self.ln_i = nn.LayerNorm(dim, eps=eps)  # Changed to normalize over last dimension only

    def forward(self, x: torch.Tensor):
        # Handle both real and complex inputs
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))
            
        # Get input shape
        batch_size, seq_len, dim = x.shape
        
        # Ensure we're normalizing over the correct dimension
        if dim != self.dim:
            raise ValueError(f"Expected input dimension {self.dim}, but got {dim}")
            
        # Reshape to [batch_size * seq_len, dim] for normalization
        x_reshaped = x.reshape(-1, dim)
        
        # Apply layer norm to real and imaginary parts separately
        real_norm = self.ln_r(x_reshaped.real)
        imag_norm = self.ln_i(x_reshaped.imag)
        
        # Reshape back to original shape
        real_norm = real_norm.reshape(batch_size, seq_len, dim)
        imag_norm = imag_norm.reshape(batch_size, seq_len, dim)
        
        return torch.complex(real_norm, imag_norm)

class LinearLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Initialize weights with proper scaling
        # Note: weight shape is [in_features, out_features] for F.linear
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim, dtype=torch.complex64) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_dim, dtype=torch.complex64))
        
    def forward(self, x: torch.Tensor, ops: Optional[MultivectorOps] = None) -> torch.Tensor:
        if ops is None:
            ops = MultivectorOps()
            
        # Ensure input is complex
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))
            
        # Handle batched inputs correctly
        if x.dim() > 2:
            batch_shape = x.shape[:-1]
            x_reshaped = x.reshape(-1, x.shape[-1])
            # F.linear expects weight shape [out_features, in_features]
            out = F.linear(x_reshaped, self.weight.t(), self.bias)
            return out.reshape(*batch_shape, -1)
        else:
            return F.linear(x, self.weight.t(), self.bias)

class ComplexInputProjection(nn.Module):
    def __init__(self, config, ops):
        super().__init__()
        self.config = config
        self.ops = ops
        self.linear = LinearLayer(in_dim=1, out_dim=config.input_dim)
        self.norm = LayerNorm(config.input_dim)
        self.act = ComplexPReLU(config)

    def forward(self, x: torch.Tensor, ops=None) -> torch.Tensor:
        ops = ops or self.ops
        if ops is None:
            ops = MultivectorOps()
            
        # Convert input to complex if real
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))
            
        # Project using complex linear layer
        x = self.linear(x, ops=ops)
        x = self.norm(x)
        return self.act(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.max_len = config.max_seq_len
        
        # Create geometric progression of frequencies for embeddings
        freqs = 1.0 / (10000 ** (torch.arange(0, self.d_model, dtype=torch.float32) / self.d_model))
        self.register_buffer('freqs', freqs)
        print(f"[PositionalEncoding] I got initialized successfully with geometric frequencies: min={freqs.min().item():.5f}, max={freqs.max().item():.5f}! :D")

    def forward(self, x: torch.Tensor, ops: Optional[MultivectorOps] = None) -> torch.Tensor:
        """
        Args:
            x: Complex tensor of shape [batch_size, seq_len, embed_dim]
            ops: Optional MultivectorOps instance
        Returns:
            Complex tensor of same shape with positional information
        """
        if ops is None:
            ops = MultivectorOps()
            
        batch_size, seq_len, embed_dim = x.shape
        positions = torch.arange(0, seq_len, device=x.device, dtype=torch.float32)
        angles = positions.unsqueeze(1) * self.freqs.unsqueeze(0)

        # Create complex phase tensor directly
        encoding = torch.exp(1j * angles)  # [seq_len, embed_dim]
        
        # Expand for batch dimension and add to input
        encoding = encoding.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Add positional encoding using ops for complex addition
        return ops.matmul_sync(x, encoding)

class ComplexPReLU(nn.Module):
    def __init__(self, config_or_dim):
        super().__init__()
        if hasattr(config_or_dim, 'input_dim'):
            input_dim = config_or_dim.input_dim
        else:
            input_dim = int(config_or_dim)
        self.input_dim = input_dim
        self.pos_weight = nn.Parameter(torch.ones(input_dim))
        self.neg_weight = nn.Parameter(torch.ones(input_dim) * 0.25)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        r = torch.abs(z)
        phi = torch.angle(z)
        r_out = self.pos_weight * r
        return r_out * torch.exp(1j * phi)

class DynamicActivation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.cprelu = ComplexPReLU(dim)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.complex64))
        self.beta = nn.Parameter(torch.zeros(1, dtype=torch.complex64))
        self.dropout = ComplexDropout(0.3)
        print("[DynamicActivation] I got initialized successfully! :D")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activated = self.cprelu(x)
        scaled = self.alpha * activated + self.beta
        return self.dropout(scaled)

class InfiniToeplitz(nn.Module):
    def __init__(self, config: ModelConfig, ops: MultivectorOps):
        super().__init__()
        self.config = config
        self.ops = ops
        self.num_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        # Projection layers for Q, K, V
        self.q_proj = LinearLayer(config.d_model, config.d_model)
        self.k_proj = LinearLayer(config.d_model, config.d_model)
        self.v_proj = LinearLayer(config.d_model, config.d_model)

        # Initialize structured matrix parameters and gates
        self.key_params = nn.ParameterList()
        self.value_params = nn.ParameterList()
        self.key_gate = nn.Parameter(torch.ones(config.n_heads, 1, dtype=torch.float32) * 0.5)
        self.value_gate = nn.Parameter(torch.ones(config.n_heads, 1, dtype=torch.float32) * 0.5)

        for _ in range(config.n_heads):
            # Initialize key parameters
            key_param = torch.randn(self.head_dim, dtype=torch.complex64) * 0.02
            self.key_params.append(nn.Parameter(key_param))

            # Initialize value parameters
            value_param = torch.randn(self.head_dim, dtype=torch.complex64) * 0.02
            self.value_params.append(nn.Parameter(value_param))

        # Output projection
        self.out_proj = LinearLayer(config.d_model, config.d_model)
        self.norm_q = LayerNorm(config.d_model)
        self.norm_k = LayerNorm(config.d_model)
        self.norm_v = LayerNorm(config.d_model)

    def _circulant_matmul(self, x, c):
        """Handles both real and complex inputs with proper output dtype"""
        # Ensure complex dtype
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))
        if not c.is_complex():
            c = torch.complex(c, torch.zeros_like(c))
        
        # FFT processing
        X = torch.fft.fft(x)
        C = torch.fft.fft(c)
        
        # Direct complex multiplication
        out = torch.fft.ifft(X * C)
        
        # Ensure output matches input dtype
        return out.to(x.dtype)

    def forward(self, x, ops=None, tokens=None):
        ops = ops or self.ops
        batch_size, seq_len, _ = x.shape
        
        # Project and normalize
        q = self.q_proj(self.norm_q(x), ops)
        k = self.k_proj(self.norm_k(x), ops)
        v = self.v_proj(self.norm_v(x), ops)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Initialize states
        key_state = [param.clone().expand(batch_size, -1) for param in self.key_params]
        value_state = [param.clone().expand(batch_size, -1) for param in self.value_params]
        
        outputs = []
        for i in range(seq_len):
            q_i = q[:, :, i, :]
            k_i = k[:, :, i, :]
            v_i = v[:, :, i, :]
            head_outputs = []
            
            for h in range(self.num_heads):
                # Update states
                delta_k = k_i[:, h]
                key_state[h] = (1 - self.key_gate[h]) * key_state[h] + self.key_gate[h] * delta_k
                
                delta_v = v_i[:, h]
                value_state[h] = (1 - self.value_gate[h]) * value_state[h] + self.value_gate[h] * delta_v
                
                # Compute attention using circulant multiplication
                attn = self._circulant_matmul(q_i[:, h], key_state[h])
                attn_scores = torch.complex(
                    F.softmax(attn.real, dim=-1),
                    torch.zeros_like(attn.real)
                )
                
                # Compute head output
                out = self._circulant_matmul(attn_scores, value_state[h])
                head_outputs.append(out)
            
            combined = torch.stack(head_outputs, dim=1)  # [batch, heads, head_dim]
            outputs.append(combined.reshape(batch_size, -1))  # Flatten head dims
        
        output = torch.stack(outputs, dim=1)  # [batch, seq_len, embed_dim]
        return self.out_proj(output, ops)

class ComplexMLP(nn.Module):
    def __init__(self, config, ops):
        super().__init__()
        self.config = config
        self.ops = ops
        # Use d_model for input/output dimensions
        self.norm1 = LayerNorm(config.d_model)  # For input normalization
        self.norm2 = LayerNorm(config.d_ff)    # For intermediate normalization
        # First linear layer: d_model -> d_ff
        self.lin1a = LinearLayer(config.d_model, config.d_ff)
        self.lin1b = LinearLayer(config.d_model, config.d_ff)
        self.act = DynamicActivation(config.d_ff)
        self.res_gate = nn.Parameter(torch.randn(1, dtype=torch.complex64))
        # Second linear layer: d_ff -> d_model
        self.lin2 = LinearLayer(config.d_ff, config.d_model)
        self.drop = ComplexDropout(config.dropout)
        self.scale = nn.Parameter(torch.ones(1, dtype=torch.float32) * 0.02)
        print("[ComplexMLP] I got initialized successfully! :D")

    def forward(self, x, ops=None):
        if ops is None:
            ops = self.ops
        residual = x  # [batch_size, seq_len, d_model]
        x = self.norm1(x)  # [batch_size, seq_len, d_model]
        xa = self.lin1a(x, ops)  # [batch_size, seq_len, d_ff]
        xb = self.lin1b(x, ops)  # [batch_size, seq_len, d_ff]
        x = self.act(xa * torch.sigmoid(self.res_gate) + xb * (1 - torch.sigmoid(self.res_gate)))  # [batch_size, seq_len, d_ff]
        x = self.norm2(x)  # [batch_size, seq_len, d_ff]
        x = self.lin2(x, ops)  # [batch_size, seq_len, d_model]
        return self.drop(x * self.scale) + residual  # [batch_size, seq_len, d_model]

class TransformerLayer(nn.Module):
    def __init__(self, config: ModelConfig, ops: MultivectorOps, concept_graph=None):
        super().__init__()
        self.config = config
        self.ops = ops
        embed_dim = config.d_model
        num_heads = config.n_heads
        mlp_dim = config.mlp_dim
        self.attn_norm = LayerNorm(embed_dim)
        self.mlp_norm = LayerNorm(embed_dim)
        self.attn = InfiniToeplitz(config, ops)
        self.mlp = ComplexMLP(config, ops)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32) * 0.02)
        self.drop = ComplexDropout(config.dropout)
        # Store concept_graph directly since it's already a weak reference
        self.concept_graph = concept_graph
        print("[TransformerLayer] I got initialized successfully! :D")

    def forward(self, x: torch.Tensor, ops: Optional[MultivectorOps] = None) -> torch.Tensor:
        if ops is None:
            ops = self.ops
        attn_out = self.attn(self.attn_norm(x), ops)
        x = x + self.drop(attn_out * self.alpha)
        mlp_out = self.mlp(self.mlp_norm(x), ops)
        return x + self.drop(mlp_out * self.alpha)

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig, ops: Optional[MultivectorOps] = None, concept_graph=None):
        super().__init__()
        self.config = config
        self.ops = ops or MultivectorOps()
        self.concept_graph = concept_graph
        
        # Input embedding and positional encoding
        self.embedding = ComplexLinear(config.input_dim, config.d_model)
        self.pos_encoding = PositionalEncoding(config)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerLayer(config, self.ops, concept_graph)
            for _ in range(config.n_layers)
        ])
        
        # Output projection
        self.output = ComplexLinear(config.d_model, config.output_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, ComplexLinear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Input embedding
        x = self.embedding(x)
        x = self.pos_encoding(x, self.ops)
        
        # Process through layers
        for block in self.blocks:
            x = block(x, self.ops)
            
        # Output projection
        return self.output(x)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = LinearLayer(d_model, d_ff)
        self.linear2 = LinearLayer(d_ff, d_model)
        self.activation = nn.GELU()
        self.norm = LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return self.norm(x + residual)

class ComplexLinear(nn.Module):
    """Complex-valued linear layer with geometric algebra support."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights with proper scaling for complex numbers
        # Note: weight shape is [in_features, out_features] for F.linear
        self.weight = nn.Parameter(torch.randn(in_features, out_features, dtype=torch.complex64) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.complex64)) if bias else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))
            
        # Handle batched inputs correctly
        if x.dim() > 2:
            batch_shape = x.shape[:-1]
            x_reshaped = x.reshape(-1, x.shape[-1])
            # F.linear expects weight shape [out_features, in_features]
            out = F.linear(x_reshaped, self.weight.t(), self.bias)
            return out.reshape(*batch_shape, -1)
        else:
            return F.linear(x, self.weight.t(), self.bias)

class ComplexLoss(nn.Module):
    """Custom loss function that handles complex numbers and incorporates concept graph regularization."""
    def __init__(self, concept_graph=None, alpha=0.1, beta=0.01):
        super().__init__()
        self.concept_graph = concept_graph
        self.alpha = alpha  # Weight for concept graph regularization
        self.beta = beta   # Weight for complex number regularization
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between complex predictions and targets.
        
        Args:
            pred: Complex tensor of shape [batch_size, seq_len, dim]
            target: Complex tensor of shape [batch_size, seq_len, dim]
            
        Returns:
            Total loss combining MSE, concept graph regularization, and complex number regularization
        """
        # Ensure inputs are complex
        if not pred.is_complex():
            pred = torch.complex(pred, torch.zeros_like(pred))
        if not target.is_complex():
            target = torch.complex(target, torch.zeros_like(target))
            
        # Compute MSE loss for real and imaginary parts separately
        real_loss = F.mse_loss(pred.real, target.real)
        imag_loss = F.mse_loss(pred.imag, target.imag)
        mse_loss = real_loss + imag_loss
        
        # Add complex number regularization to encourage meaningful phase
        phase_reg = torch.mean(torch.abs(torch.angle(pred)))  # Penalize large phase angles
        
        # Add concept graph regularization if available
        concept_loss = torch.tensor(0.0, device=pred.device)
        if self.concept_graph is not None:
            # Get concept embeddings from the graph
            concepts = self.concept_graph.get_concepts()
            if concepts is not None and len(concepts) > 0:
                # Compute cosine similarity between predictions and concept embeddings
                pred_norm = F.normalize(pred.reshape(-1, pred.shape[-1]), dim=-1)
                concept_norm = F.normalize(concepts, dim=-1)
                similarity = torch.matmul(pred_norm, concept_norm.t())
                
                # Encourage predictions to align with relevant concepts
                concept_loss = -torch.mean(torch.max(similarity, dim=-1)[0])
        
        # Combine losses
        total_loss = mse_loss + self.alpha * concept_loss + self.beta * phase_reg
        
        return total_loss
