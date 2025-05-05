import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .ops import MultivectorOps

class TaskComplexityEstimator(nn.Module):
    def __init__(
        self,
        threshold: float = 1e-3,
        eps: float = 1e-6
    ):
        """
        threshold: any multivector component with |value| > threshold
                   counts as “active” toward complexity.
        """
        super().__init__()
        self.threshold = threshold
        self.eps       = eps

    def forward(self, mv: torch.Tensor) -> torch.Tensor:
        """
        mv: [B, L, D] or [B, L, D, mv_dim] multivector tensor.
        If 4-D, we collapse to [..., channel] for complexity.
        Returns: [B, L] in [0,1].
        """
        # collapse extra dims
        if mv.dim() == 4:
            # flatten feature × grade dims
            B, L, D, G = mv.shape
            flat = mv.view(B, L, D * G)
        else:
            flat = mv  # assume already [B,L,D]
        # 1) “Density” = fraction of components > threshold
        density = (flat.abs() > self.threshold).float().mean(dim=-1)

        # 2) “Variance” = per-token variance of magnitude
        mag = flat.abs()
        var = mag.var(dim=-1) / (mag.mean(dim=-1) + self.eps)

        # normalize var to [0,1] by soft clipping
        var_score = torch.tanh(var)

        # combine
        score = (density + var_score) / 2
        return score.clamp(0, 1)  # [B, L]

class FeatureRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        threshold: float = 0.5,
        estimator: Optional[TaskComplexityEstimator] = None,
    ):
        """
        d_model: your token feature dimension
        hidden_dim: width of the internal routing MLP (default = d_model//2)
        threshold: default decision threshold (not used directly in forward)
        estimator: you can supply your own TaskComplexityEstimator
        """
        super().__init__()
        self.d_model   = d_model
        self.hidden    = hidden_dim or (d_model // 2)
        self.threshold = threshold

        self.estimator = estimator or TaskComplexityEstimator()

        # A tiny MLP that operates on [B, L, 2] → [B, L, 1]
        # input features: [complexity_score, mean_magnitude]
        self.mlp = nn.Sequential(
            nn.Linear(2, self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, 1),
        )

    async def forward(self, mv: torch.Tensor) -> torch.Tensor:
        """
        mv: [B, L, D] or [B, L, D, mv_dim] multivector tensor
        Returns: [B, L, 1] in (0,1)—higher means “use dynamic branch”
        """
        # 1) estimation
        with torch.no_grad():
            comp_score = self.estimator(mv)  # [B, L]
            # we also provide mean magnitude as extra signal
            if mv.dim() == 4:
                mag = mv.abs().mean(dim=(-1, -2))  # [B,L]
            else:
                mag = mv.abs().mean(dim=-1)       # [B,L]
            feats = torch.stack([comp_score, mag], dim=-1)  # [B, L, 2]

        # 2) routing MLP
        gate = self.mlp(feats)  # [B, L, 1]
        return torch.sigmoid(gate)  # [B, L, 1]

    def forward_sync(self, mv: torch.Tensor) -> torch.Tensor:
        # identical logic sync
        comp_score = self.estimator(mv)
        if mv.dim() == 4:
            mag = mv.abs().mean(dim=(-1, -2))
        else:
            mag = mv.abs().mean(dim=-1)
        feats = torch.stack([comp_score, mag], dim=-1)
        gate  = self.mlp(feats)
        return torch.sigmoid(gate)

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
    def __init__(self, in_dim: int, out_dim: int, ops: MultivectorOps, rank: Optional[int] = None):
        super().__init__()
        self.ops = ops
        self.rank = rank or min(in_dim, out_dim) // 2
        self.A = nn.Parameter(torch.randn(in_dim, self.rank, dtype=torch.complex64) * 0.02)
        self.B = nn.Parameter(torch.randn(self.rank, out_dim, dtype=torch.complex64) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_dim, dtype=torch.complex64))

    async def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x
        x_flat = x.view(-1, x.shape[-1])                  # [N, in_dim]
        mid = await self.ops.matmul(x_flat, self.A)       # [N, rank]
        out = await self.ops.matmul(mid, self.B)          # [N, out_dim]
        out = out + self.bias
        return out.view(*orig.shape[:-1], self.B.shape[1])

    def forward_sync(self, x: torch.Tensor) -> torch.Tensor:
        orig = x
        x_flat = x.view(-1, x.shape[-1])
        mid = self.ops.matmul_sync(x_flat, self.A)
        out = self.ops.matmul_sync(mid, self.B)
        out = out + self.bias
        return out.view(*orig.shape[:-1], self.B.shape[1])

class DynamicActivation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dtype=torch.complex64) * 0.02)
        self.bias   = nn.Parameter(torch.zeros(dim, dtype=torch.complex64))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
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
                 ops: MultivectorOps,      # <= swapped in
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
        qkv = await self.qkv(x)                           # [B, S, 3D]
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        def merge_heads(t):
            return t.permute(0,2,1,3).reshape(-1, S, self.head_dim)
        qh, kh, vh = map(merge_heads, (q,k,v))

        if self.use_toeplitz:
            q_f = await self.ops.fft(qh, dim=1)
            k_f = await self.ops.fft(kh, dim=1)
            prod = q_f * torch.conj(k_f)
            corr = await self.ops.ifft(prod, dim=1)
            scores = corr.real
        else:
            k_t = kh.conj().transpose(-2,-1)
            scores = (await self.ops.matmul(qh, k_t)).real

        scores = scores / (self.head_dim**0.5)
        attn   = F.softmax(scores, dim=-1).type_as(scores)

        ctx = await self.ops.matmul(attn, vh)             # [B*h, S, d]
        ctx = ctx.view(B, self.num_heads, S, self.head_dim) \
                 .permute(0,2,1,3).reshape(B, S, D)

        return await self.out_proj(ctx)

    def forward_sync(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv.forward_sync(x)
        qkv = qkv.view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        def merge_heads(t):
            return t.permute(0,2,1,3).reshape(-1, S, self.head_dim)
        qh, kh, vh = map(merge_heads, (q,k,v))

        if self.use_toeplitz:
            q_f = self.ops.fft_sync(qh, dim=1)
            k_f = self.ops.fft_sync(kh, dim=1)
            prod = q_f * torch.conj(k_f)
            corr = self.ops.ifft_sync(prod, dim=1)
            scores = corr.real
        else:
            k_t = kh.conj().transpose(-2,-1)
            scores = (self.ops.matmul_sync(qh, k_t)).real

        scores = scores / (self.head_dim**0.5)
        attn   = F.softmax(scores, dim=-1).type_as(scores)

        ctx = self.ops.matmul_sync(attn, vh)
        ctx = ctx.view(B, self.num_heads, S, self.head_dim) \
                 .permute(0,2,1,3).reshape(B, S, D)

        return self.out_proj.forward_sync(ctx)

class TransformerBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 mlp_dim: int,
                 ops: MultivectorOps,      # <= swapped in
                 router: nn.Module,
                 use_learnable_act: bool = True):
        super().__init__()
        self.ln1  = ComplexLayerNorm(dim)
        self.attn = AttentionLayer(dim, num_heads, ops)
        self.ln2  = ComplexLayerNorm(dim)
        self.use_learnable_act = use_learnable_act

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

        self.router = router

    async def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = await self.attn(self.ln1(x))
        x = x + attn_out

        complexity = self.router(x)              
        mask_high  = (complexity > 0.5).float()

        y = self.ln2(x)
        out_static = self.ffn_static(y)
        out_dyn    = await self._apply_dynamic_ffn(y)

        m = mask_high.expand_as(out_dyn)
        ffn_out = out_dyn * m + out_static * (1 - m)
        return x + ffn_out

    async def _apply_dynamic_ffn(self, y: torch.Tensor) -> torch.Tensor:
        y1 = await self.ffn_dynamic[0](y)
        y2 =        self.ffn_dynamic[1](y1)
        y3 = await self.ffn_dynamic[2](y2)
        return y3

    def forward_sync(self, x: torch.Tensor) -> torch.Tensor:
        x_in = x
        attn_out = self.attn.forward_sync(self.ln1(x_in))
        x = x_in + attn_out

        complexity = self.router(x)              
        mask_high  = (complexity > 0.5).float()

        y = self.ln2(x)
        out_static = self.ffn_static(y)
        out_dyn    = self._apply_dynamic_ffn_sync(y)

        m = mask_high.expand_as(out_dyn)
        ffn_out = out_dyn * m + out_static * (1 - m)
        return x + ffn_out

    def _apply_dynamic_ffn_sync(self, y: torch.Tensor) -> torch.Tensor:
        y1 = self.ffn_dynamic[0].forward_sync(y)
        y2 = self.ffn_dynamic[1](y1)
        y3 = self.ffn_dynamic[2].forward_sync(y2)
        return y3

def build_transformer(num_layers: int,
                      dim: int,
                      num_heads: int,
                      mlp_dim: int,
                      ops: MultivectorOps,
                      router: nn.Module):
    return nn.ModuleList([
        TransformerBlock(dim, num_heads, mlp_dim, ops, router)
        for _ in range(num_layers)
    ])
