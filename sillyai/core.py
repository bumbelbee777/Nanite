import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA
from typing import Optional

from .ops import OptimizedComplexOps

class ComplexLayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.ln_real = nn.LayerNorm(dim, eps=eps)
        self.ln_imag = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real = self.ln_real(x.real)
        imag = self.ln_imag(x.imag)
        return torch.complex(real, imag)

class LinearLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, ops: OptimizedComplexOps, rank: Optional[int] = None):
        super().__init__()
        self.ops = ops
        self.rank = rank or min(in_dim, out_dim) // 2
        self.A = nn.Parameter(torch.randn(in_dim, self.rank, dtype=torch.complex64) * 0.02)
        self.B = nn.Parameter(torch.randn(self.rank, out_dim, dtype=torch.complex64) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_dim, dtype=torch.complex64))

    async def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim)
        orig = x
        x_flat = x.view(-1, x.shape[-1])  # [N, in_dim]
        mid = await self.ops.matmul(x_flat, self.A)  # [N, rank]
        out = await self.ops.matmul(mid, self.B)     # [N, out_dim]
        out = out + self.bias
        return out.view(*orig.shape[:-1], self.B.shape[1])

class DynamicActivation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dtype=torch.complex64) * 0.02)
        self.bias   = nn.Parameter(torch.zeros(dim, dtype=torch.complex64))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, L, D]
        r   = torch.abs(z)
        phi = torch.angle(z)
        w_r, w_i = self.weight.real, self.weight.imag
        b_r, b_i = self.bias.real,   self.bias.imag

        out_r   = F.relu(r) + (r - F.relu(r)) * w_r + b_r
        out_phi = phi + b_i
        return out_r * torch.exp(1j * out_phi)

class AttentionLayer(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 ops: OptimizedComplexOps,
                 use_toeplitz: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.ops = ops
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.qkv       = LinearLayer(dim, dim * 3, ops)
        self.out_proj  = LinearLayer(dim, dim, ops)
        self.use_toeplitz = use_toeplitz

    async def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        # project QKV
        qkv = await self.qkv(x)  # [B, S, 3D]
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # reshape to (B*H, S, head_dim)
        def merge_heads(t):
            return t.permute(0,2,1,3).reshape(-1, S, self.head_dim)
        qh, kh, vh = map(merge_heads, (q,k,v))

        if self.use_toeplitz:
            # -- InfiniToeplitz/streaming --
            # 1) stream FFT(q) and FFT(k) via ops.fft
            q_f = await self.ops.fft(qh, dim=1)
            k_f = await self.ops.fft(kh, dim=1)
            # 2) multiply & IFFT back
            prod = q_f * torch.conj(k_f)
            corr = await self.ops.ifft(prod, dim=1)
            scores = corr.real
        else:
            # standard QK^H
            k_t = kh.conj().transpose(-2,-1)
            scores = (await self.ops.matmul(qh, k_t)).real

        scores = scores / (self.head_dim**0.5)
        attn   = F.softmax(scores, dim=-1).type_as(scores)

        ctx = await self.ops.matmul(attn, vh)  # [B*h, S, d]
        # reshape back to [B, S, D]
        ctx = ctx.view(B, self.num_heads, S, self.head_dim) \
                 .permute(0,2,1,3).reshape(B, S, D)

        return await self.out_proj(ctx)

class TransformerBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_dim: int,
                 ops: OptimizedComplexOps,
                 router: nn.Module,
                 use_learnable_act: bool = True):
        super().__init__()
        self.ln1  = ComplexLayerNorm(dim)
        self.attn = AttentionLayer(dim, num_heads, ops)
        self.ln2  = ComplexLayerNorm(dim)
        self.use_learnable_act = use_learnable_act

        # two FFN branches: static (GELU) vs dynamic (PReLU)
        self.ffn_static = nn.Sequential(
            LinearLayer(dim, mlp_dim, ops),
            nn.GELU(),
            LinearLayer(mlp_dim, dim, ops),
        )
        self.ffn_dynamic = nn.Sequential(
            LinearLayer(dim, mlp_dim, ops),
            DynamicActivation(mlp_dim),
            LinearLayer(mlp_dim, dim, ops),
        )

        self.router = router  # should output [B, L, 1] complexity

    async def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D] complex
        # 1) attention block
        attn_out = await self.attn(self.ln1(x))
        x = x + attn_out

        # 2) decide path per token
        B, L, D = x.shape
        complexity = self.router(x)              # [B, L, 1]
        mask_high  = (complexity > 0.5).float()  # static vs dynamic

        # 3) apply LN then split into two FFNs
        y = self.ln2(x)
        out_static = self.ffn_static(y)   # uses GELU
        out_dyn    = await self._apply_dynamic_ffn(y)

        # mix outputs based on complexity
        # broadcast mask to D dims:
        m = mask_high.expand(B, L, D)
        ffn_out = out_dyn * m + out_static * (1 - m)

        return x + ffn_out

    async def _apply_dynamic_ffn(self, y: torch.Tensor) -> torch.Tensor:
        # manual async-forward through the dynamic branch
        y1 = await self.ffn_dynamic[0](y)         # LinearLayer
        y2 =        self.ffn_dynamic[1](y1)       # DynamicActivation (sync)
        y3 = await self.ffn_dynamic[2](y2)        # LinearLayer
        return y3

def build_transformer(num_layers: int,
                      dim: int,
                      num_heads: int,
                      mlp_dim: int,
                      ops: OptimizedComplexOps,
                      router: nn.Module):
    layers = []
    for _ in range(num_layers):
        layers.append(
            TransformerBlock(dim, num_heads, mlp_dim, ops, router)
        )
    return nn.ModuleList(layers)
