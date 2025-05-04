import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops import OptimizedComplexOps

class ComplexLayerNorm(nn.Module):
    """LayerNorm applied separately to real and imag parts."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.ln_real = nn.LayerNorm(dim, eps=eps)
        self.ln_imag = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., dim], complex dtype
        real = self.ln_real(x.real)
        imag = self.ln_imag(x.imag)
        return torch.complex(real, imag)

class LinearLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, ops: OptimizedComplexOps):
        super().__init__()
        self.ops = ops
        # complex weight: shape (in_dim, out_dim)
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim, dtype=torch.complex64) * 0.02)
        self.bias   = nn.Parameter(torch.zeros(out_dim,     dtype=torch.complex64))

    async def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, in_dim) or (batch, in_dim)
        # flatten batch and seq for matmul
        orig_shape = x.shape
        flat = x.view(-1, x.shape[-1])
        y = await self.ops.matmul(flat, self.weight)
        y = y + self.bias
        return y.view(*orig_shape[:-1], -1)

class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, ops: OptimizedComplexOps):
        super().__init__()
        self.lin1 = LinearLayer(dim, hidden_dim, ops)
        self.lin2 = LinearLayer(hidden_dim, dim, ops)

    async def forward(self, x: torch.Tensor) -> torch.Tensor:
        # simple complex GELU: apply to real and imag separately
        z = await self.lin1(x)
        real = F.gelu(z.real)
        imag = F.gelu(z.imag)
        z = torch.complex(real, imag)
        return await self.lin2(z)

class AttentionLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int, ops: OptimizedComplexOps):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.qkv      = LinearLayer(dim, dim * 3, ops)
        self.out_proj = LinearLayer(dim, dim, ops)

    async def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, dim)
        B, S, D = x.shape
        # project QKV
        qkv = await self.qkv(x)                  # (B, S, 3*D)
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)              # each (B, S, heads, head_dim)

        # transpose for matmul: (B, heads, S, head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # scaled dot-product: Q @ K^H
        # flatten B*heads for ops.matmul
        q_flat = q.reshape(-1, S, self.head_dim)               # (B*h, S, d)
        k_flat = k.reshape(-1, S, self.head_dim)
        # conj-transpose K
        k_flat_h = k_flat.conj().transpose(-2, -1)             # (B*h, d, S)
        # scores: (B*h, S, S)
        scores = await self.qkv.ops.matmul(q_flat, k_flat_h)  
        scores = scores / (self.head_dim ** 0.5)

        # softmax on real part (imag zeroed for attention weights)
        attn = F.softmax(scores.real, dim=-1)
        attn = attn.type_as(scores)                           # cast back to complex

        # attend V
        v_flat = v.reshape(-1, S, self.head_dim)               # (B*h, S, d)
        ctx = await self.qkv.ops.matmul(attn, v_flat)          # (B*h, S, d)
        ctx = ctx.view(B, self.num_heads, S, self.head_dim)
        ctx = ctx.permute(0, 2, 1, 3).reshape(B, S, D)         # (B, S, D)

        return await self.out_proj(ctx)

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_dim: int, ops: OptimizedComplexOps):
        super().__init__()
        self.attn = AttentionLayer(dim, num_heads, ops)
        self.ln1  = ComplexLayerNorm(dim)
        self.ffn  = FeedForward(dim, mlp_dim, ops)
        self.ln2  = ComplexLayerNorm(dim)

    async def forward(self, x: torch.Tensor) -> torch.Tensor:
        # multi-head attention + residual
        attn_out = await self.attn(self.ln1(x))
        x = x + attn_out
        # feed-forward + residual
        ffn_out = await self.ffn(self.ln2(x))
        return x + ffn_out
