import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import fft
from typing import Optional, List, Dict, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import *

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
                   counts as “active” toward complexity.
        """
        super().__init__()
        self.threshold = threshold
        self.eps       = eps
        self.use_fft   = use_fft
        
        # Complex feature extraction layers
        self.feature_net = ComplexMLP(4, 1, feature_dim)
        
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
        
        # Complex MLP for routing
        self.complex_mlp = ComplexMLP(2, 1, self.hidden)

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
        self.ln_r = nn.LayerNorm(dim, eps)
        self.ln_i = nn.LayerNorm(dim, eps)

    def forward(self, x: torch.Tensor):
        return torch.complex(self.ln_r(x.real), self.ln_i(x.imag))

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=None):
        super().__init__()
        r = rank or min(in_dim, out_dim) // 2
        self.A = nn.Parameter(torch.randn(in_dim, r, dtype=torch.complex64)*.02)
        self.B = nn.Parameter(torch.randn(r, out_dim, dtype=torch.complex64)*.02)
        self.bias = nn.Parameter(torch.zeros(out_dim, dtype=torch.complex64))
        print("[LinearLayer] I got initialized successfully! :D")

    def forward(self, x, ops):
        o = x.view(-1, x.size(-1))
        m = ops.matmul_sync(o, self.A)
        y = ops.matmul_sync(m, self.B)
        return (y + self.bias).view(*x.shape[:-1], -1)

class ComplexInputProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = LinearLayer(in_dim, out_dim)
        self.norm = LayerNorm(out_dim)
        self.act = ComplexPReLU(out_dim)

    def forward(self, x: torch.Tensor, ops: Optional[MultivectorOps] = None) -> torch.Tensor:
        if ops is None:
            ops = MultivectorOps()
            
        # Convert input to complex if real
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))
            
        # Project using complex linear layer
        x = self.linear(x, ops)
        x = self.norm(x)
        return self.act(x)
    
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        # Create geometric progression of frequencies for complex embeddings
        freqs = 1.0 / (10000 ** (torch.arange(0, embed_dim, dtype=torch.float32) / embed_dim))
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
    def __init__(self, dim):
        super().__init__()
        self.pos_weight = nn.Parameter(torch.ones(dim))
        self.neg_weight = nn.Parameter(torch.ones(dim) * 0.25)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        r = torch.abs(z)
        phi = torch.angle(z)

        # Vectorized positive/negative split on magnitude
        # PReLU applies pos_weight when r>0 else neg_weight (which is rare because magnitude>=0)
        # We can just do: r_out = torch.where(r > 0, pos_weight * r, neg_weight * r)
        # But since r is always >=0, neg part is effectively zero, so use clamp for numerical stability
        r_out = self.pos_weight * r

        return r_out * torch.exp(1j * phi)

class DynamicActivation(nn.Module):
    def __init__(self, dim):
        super().__init__()
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
    def __init__(self, embed_dim, num_heads, key_structure='toeplitz', value_structure='circulant', concept_graph=None):
        super().__init__()
        
        # Validate dimensions
        if embed_dim < num_heads:
            raise ValueError(f"embed_dim ({embed_dim}) must be >= num_heads ({num_heads})")
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
            
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.key_structure = key_structure
        self.value_structure = value_structure
        self.concept_graph = concept_graph

        # Projection layers for Q, K, V
        self.q_proj = LinearLayer(embed_dim, embed_dim)
        self.k_proj = LinearLayer(embed_dim, embed_dim)
        self.v_proj = LinearLayer(embed_dim, embed_dim)

        # Initialize structured matrix parameters and gates
        self.key_params = nn.ParameterList()
        self.value_params = nn.ParameterList()
        self.key_gate = nn.Parameter(torch.ones(num_heads, 1, dtype=torch.float32) * 0.5)
        self.value_gate = nn.Parameter(torch.ones(num_heads, 1, dtype=torch.float32) * 0.5)

        for _ in range(num_heads):
            if key_structure == 'toeplitz':
                key_param = torch.randn(2 * self.head_dim - 1, dtype=torch.complex64) * 0.02
            elif key_structure == 'circulant':
                key_param = torch.randn(self.head_dim, dtype=torch.complex64) * 0.02
            self.key_params.append(nn.Parameter(key_param))

            if value_structure == 'circulant':
                value_param = torch.randn(self.head_dim, dtype=torch.complex64) * 0.02
            self.value_params.append(nn.Parameter(value_param))

        # Output projection
        self.out_proj = LinearLayer(embed_dim, embed_dim)
        self.norm_q = LayerNorm(embed_dim)
        self.norm_k = LayerNorm(embed_dim)
        self.norm_v = LayerNorm(embed_dim)

        print(f"[InfiniToeplitz] I got initialized successfully with {num_heads} heads, key: {key_structure}, value: {value_structure}! :D")

    def _project_key_update(self, x, output_dim):
        """
        Projects complex-valued input to target dimension using FFT-based method
        Args:
            x: complex-valued tensor [..., input_dim]
            output_dim: target output dimension
        Returns:
            complex-valued tensor [..., output_dim]
        """
        # Ensure input is properly shaped [..., features]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Get original feature dimension
        input_dim = x.size(-1)
        
        # Split into real and imaginary components
        x_real, x_imag = x.real, x.imag
        
        # Pad or truncate to target dimension
        if output_dim > input_dim:
            pad_size = output_dim - input_dim
            x_real_pad = F.pad(x_real, (0, pad_size))
            x_imag_pad = F.pad(x_imag, (0, pad_size))
        else:
            x_real_pad = x_real[..., :output_dim]
            x_imag_pad = x_imag[..., :output_dim]
        
        # Generate random projection matrix (real-valued)
        rand_matrix = torch.randn(output_dim, device=x.device)
        
        # Process real component
        x_real_fft = torch.fft.rfft(x_real_pad)
        rand_fft = torch.fft.rfft(rand_matrix)
        min_dim = min(x_real_fft.size(-1), rand_fft.size(-1))
        proj_real = torch.fft.irfft(x_real_fft[..., :min_dim] * rand_fft[..., :min_dim], n=output_dim)
        
        # Process imaginary component
        x_imag_fft = torch.fft.rfft(x_imag_pad)
        proj_imag = torch.fft.irfft(x_imag_fft[..., :min_dim] * rand_fft[..., :min_dim], n=output_dim)
        
        # Combine back into complex tensor
        return torch.complex(proj_real, proj_imag)

    def _toeplitz_matmul(self, x, t):
        # x: [batch, head_dim], t: [2*head_dim-1]
        n = self.head_dim
        
        # Pad input to length m+n-1 for linear convolution
        x_padded = F.pad(x, (0, n-1))
        
        # Pad Toeplitz matrix diagonal to same length
        t_padded = F.pad(t, (0, x_padded.size(-1) - t.size(-1)))
        
        # FFT multiplication
        X = torch.fft.fft(x_padded)
        T = torch.fft.fft(t_padded)
        product = torch.fft.ifft(X * T)[..., :n]  # Keep only valid part
        
        return product

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

    def forward(self, x, ops, tokens=None):
        batch_size, seq_len, _ = x.shape[:3]
        q = self.q_proj(self.norm_q(x), ops)
        k = self.k_proj(self.norm_k(x), ops)
        v = self.v_proj(self.norm_v(x), ops)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
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
                if self.key_structure == 'toeplitz':
                    delta_k = self._project_key_update(k_i[:, h], 2*self.head_dim-1)
                else:  # circulant
                    delta_k = k_i[:, h]
                key_state[h] = (1 - self.key_gate[h]) * key_state[h] + self.key_gate[h] * delta_k
                if self.value_structure == 'circulant':
                    delta_v = v_i[:, h]
                    value_state[h] = (1 - self.value_gate[h]) * value_state[h] + self.value_gate[h] * delta_v
                # Compute attention
                if self.key_structure == 'toeplitz':
                    attn = self._toeplitz_matmul(q_i[:, h], key_state[h])
                else:
                    attn = self._circulant_matmul(q_i[:, h], key_state[h])
                attn_scores = torch.complex(
                    F.softmax(attn.real, dim=-1),
                    torch.zeros_like(attn.real)
                )
                # --- Nuanced Concept Extraction & Relationships ---
                if self.concept_graph is not None and tokens is not None:
                    token_id = tokens[:, i] if tokens.dim() == 2 else None
                    if token_id is not None:
                        for b in range(batch_size):
                            cname = f"token_{int(token_id[b])}"
                            # Add concept and associate embedding
                            self.concept_graph.add_concept(cname)
                            self.concept_graph.associate_token(cname, int(token_id[b]), embedding=k_i[b, h].detach())
                            # Extract top attended neighbors for richer relationships
                            attn_vals = attn_scores[b].real.detach().cpu().numpy()
                            top_indices = attn_vals.argsort()[-3:][::-1]  # top 3 attended
                            for idx in top_indices:
                                if idx == b:  # skip self
                                    continue
                                neighbor_token = tokens[b, idx] if tokens.dim() == 2 else None
                                if neighbor_token is not None:
                                    nname = f"token_{int(neighbor_token)}"
                                    # Add/strengthen edge with relationship type
                                    rel_type = "attention" if attn_vals[idx] > 0.5 else "co-occurrence"
                                    confidence = float(attn_vals[idx])
                                    self.concept_graph.add_concept(nname)
                                    self.concept_graph.add_edge(cname, nname, weight=confidence, relationship=rel_type, confidence=confidence)
                # --- Advanced Energy Update ---
                if self.concept_graph is not None and tokens is not None:
                    for b in range(batch_size):
                        token_id = tokens[b, i] if tokens.dim() == 2 else None
                        if token_id is not None:
                            cname = f"token_{int(token_id)}"
                            c = self.concept_graph.get(cname)
                            if c is not None:
                                # Use non-linear bounding for energy
                                attn_sum = attn_scores[b].real.abs().sum().item()
                                c.energy = float(torch.tanh(torch.tensor(c.energy + attn_sum)).item() * 10.0)
                                c.access_count += 1
                # Compute head output
                if self.value_structure == 'circulant':
                    out = self._circulant_matmul(attn_scores, value_state[h])
                else:
                    out = attn_scores  # Fallback
                head_outputs.append(out)
            combined = torch.stack(head_outputs, dim=1)  # [batch, heads, head_dim]
            outputs.append(combined.reshape(batch_size, -1))  # Flatten head dims
        output = torch.stack(outputs, dim=1)  # [batch, seq_len, embed_dim]
        # --- Propagate and decay energy after each forward pass ---
        if self.concept_graph is not None:
            self.concept_graph.propagate_energy()
            self.concept_graph.decay()
        return self.out_proj.forward(output, ops)

class ComplexMLP(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim):
        super().__init__()
        self.norm1 = LayerNorm(in_dim)
        self.norm2 = LayerNorm(h_dim)
        
        # Parallel feature pathways
        self.lin1a = LinearLayer(in_dim, h_dim)
        self.lin1b = LinearLayer(in_dim, h_dim)
        self.act = DynamicActivation(h_dim)
        
        # Gated residual connection
        self.res_gate = nn.Parameter(torch.randn(1, dtype=torch.complex64))
        self.lin2 = LinearLayer(h_dim, out_dim)
        self.drop = ComplexDropout(0.3)
        
        # Learnable scale parameters
        self.scale = nn.Parameter(torch.ones(1, dtype=torch.float32) * 0.02)
        print("[ComplexMLP] I got initialized successfully! :D")

    def forward(self, x, ops):
        residual = x
        x = self.norm1(x)
        
        # Parallel feature processing
        xa = self.lin1a(x, ops)
        xb = self.lin1b(x, ops)
        x = self.act(xa * torch.sigmoid(self.res_gate) + xb * (1 - torch.sigmoid(self.res_gate)))
        
        x = self.norm2(x)
        x = self.lin2(x, ops)
        return self.drop(x * self.scale) + residual

class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, concept_graph=None):
        super().__init__()
        self.attn_norm = LayerNorm(embed_dim)
        self.mlp_norm = LayerNorm(embed_dim)
        self.attn = InfiniToeplitz(embed_dim, num_heads, concept_graph=concept_graph)
        self.mlp = ComplexMLP(embed_dim, embed_dim, mlp_dim)
        self.alpha = nn.Parameter(torch.ones(1, dtype=torch.float32) * 0.02)
        self.drop = ComplexDropout(0.3)
        self.concept_graph = concept_graph
        print("[TransformerLayer] I got initialized successfully! :D")

    def forward(self, x, ops, tokens=None):
        attn_out = self.attn(self.attn_norm(x), ops, tokens=tokens)
        x = x + self.drop(attn_out * self.alpha)
        mlp_out = self.mlp(self.mlp_norm(x), ops)
        return x + self.drop(mlp_out * self.alpha)

class Transformer(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, mlp_dim, output_dim=1, concept_graph=None):
        super().__init__()
        self.input_proj = ComplexInputProjection(in_dim=1, out_dim=embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.layers = nn.ModuleList([
            TransformerLayer(embed_dim, num_heads, mlp_dim, concept_graph=concept_graph)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(embed_dim)
        self.head = LinearLayer(embed_dim, output_dim)
        self.concept_graph = concept_graph
        print(f"[Transformer] I got initialized successfully with {num_layers} layers and positional encoding! :D")

    def _convert_state_dict(self, state_dict):
        """Convert old state dict format to new complex-valued format"""
        new_state = {}
        
        # Handle input projection conversion
        if 'input_proj.real_proj.weight' in state_dict:
            real_w = state_dict['input_proj.real_proj.weight']
            imag_w = state_dict['input_proj.imag_proj.weight']
            in_dim, out_dim = real_w.shape
            # Convert to complex linear params
            new_state['input_proj.linear.A'] = torch.complex(
                real_w[:, :out_dim//2], 
                imag_w[:, :out_dim//2]
            )
            new_state['input_proj.linear.B'] = torch.eye(out_dim, dtype=torch.complex64)
            new_state['input_proj.linear.bias'] = torch.complex(
                state_dict['input_proj.real_proj.bias'],
                state_dict['input_proj.imag_proj.bias']
            )
            
            # Convert layer norm params
            w = state_dict['input_proj.norm.weight']
            b = state_dict['input_proj.norm.bias']
            new_state['input_proj.norm.ln_r.weight'] = w
            new_state['input_proj.norm.ln_r.bias'] = b
            new_state['input_proj.norm.ln_i.weight'] = w.clone()
            new_state['input_proj.norm.ln_i.bias'] = b.clone()
            
            # Add activation params
            new_state['input_proj.act.pos_weight'] = torch.ones(out_dim)
            new_state['input_proj.act.neg_weight'] = torch.ones(out_dim) * 0.25
        
        # Handle positional encoding
        if 'positional_encoding.freqs' in state_dict:
            old_freqs = state_dict['positional_encoding.freqs']
            embed_dim = self.positional_encoding.embed_dim
            new_state['positional_encoding.freqs'] = F.interpolate(
                old_freqs.unsqueeze(0).unsqueeze(0),
                size=embed_dim,
                mode='linear'
            ).squeeze()
        
        # Copy remaining keys
        for k, v in state_dict.items():
            if k not in new_state and not k.startswith('input_proj.') and not k.startswith('positional_encoding.'):
                new_state[k] = v
                
        return new_state

    def load_state_dict(self, state_dict, strict=True):
        """Override to handle conversion of old state dicts"""
        converted_state = self._convert_state_dict(state_dict)
        return super().load_state_dict(converted_state, strict=False)

    # --- ConceptGraph to Bytecode utility ---
    def concept_graph_to_bytecode(self, start_concept=None, max_hops=5):
        """
        Walk the concept graph as an AST and emit a list of bytecode instructions for the VM.
        """
        if self.concept_graph is None:
            return []
        cg = self.concept_graph
        visited = set()
        bytecode = []
        def walk(name, hops):
            if name in visited or hops > max_hops:
                return
            visited.add(name)
            c = cg.get(name)
            if c is None:
                return
            # Example: emit LOADC for concept, then recursively for neighbors
            bytecode.append(("LOADC", [name]))
            for conn in cg.neighbors(name):
                # Relationship can affect opcode
                if conn.relationship == "attention":
                    bytecode.append(("ADD_EDGE", [name, conn.target, str(conn.weight)]))
                elif conn.relationship == "co-occurrence":
                    bytecode.append(("ADD_EDGE", [name, conn.target, str(conn.weight)]))
                # Recursively walk
                walk(conn.target, hops + 1)
        # Start from most central or provided concept
        if start_concept is None:
            central = cg.centrality(k=1)
            if central:
                start_concept = central[0][0]
            else:
                return []
        walk(start_concept, 0)
        return bytecode
