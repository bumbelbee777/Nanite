import asyncio
import torch
import torch.nn as nn
import torch.optim as optim
import onnx

from .config import ModelConfig
from .ops import MultivectorOps
from .core import (
    TransformerBlock,
    ComplexLayerNorm,
    LinearLayer,
    FeatureRouter
)

class SillyAI(nn.Module):
    def __init__(self, cfg: ModelConfig, num_classes: int):
        """
        A silly complex-valued multimodal model, fully async under the hood,
        with dynamic activation routing and InfiniToeplitz attention.
        """
        super().__init__()
        self.cfg = cfg

        # 1) Shared MV Ops (caching, SVD, lookahead, Clifford support)
        self.ops = MultivectorOps(
            cache_max_bytes=cfg.cache_max_bytes
        )

        # 2) Shared Feature Router
        #    (now takes only d_model and optional estimator/hidden_dim)
        self.router = FeatureRouter(
            d_model=cfg.dim
        )

        # 3) Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=cfg.dim,
                num_heads=cfg.num_heads,
                mlp_dim=cfg.mlp_dim,
                ops=self.ops,
                router=self.router,
                use_learnable_act=True,
            )
            for _ in range(cfg.num_layers)
        ])

        # 4) Output head: layer‐norm + classifier
        self.ln_out     = ComplexLayerNorm(cfg.dim)
        self.classifier = LinearLayer(cfg.dim, num_classes, self.ops)

        # 5) Move to device & dtype
        device = torch.device(cfg.device or "cpu")
        dtype  = torch.complex64 if cfg.dtype == "complex64" else torch.complex128
        for module in [*self.blocks, self.ln_out, self.classifier, self.router]:
            module.to(device=device, dtype=dtype)
        self.device = device

        # 6) Optimizer & loss
        self.optimizer = optim.Adam(self.parameters())
        self.criterion = nn.CrossEntropyLoss()

    def parameters(self):
        params = []
        for blk in self.blocks:
            params += list(blk.parameters())
        params += list(self.ln_out.parameters())
        params += list(self.classifier.parameters())
        params += list(self.router.parameters())
        return params

    def complex_to_real_input(self, x: torch.Tensor) -> torch.Tensor:
        """Split a complex tensor [B, S, D] → real [B, S, 2D]."""
        return torch.cat([x.real, x.imag], dim=-1)

    def recombine_real_input(self, x: torch.Tensor) -> torch.Tensor:
        """Recombine real [B, S, 2D] → complex [B, S, D]."""
        D = x.shape[-1] // 2
        return torch.complex(x[..., :D], x[..., D:])

    async def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Async forward through transformer blocks → real logits.
        """
        # Accept either complex [B,S,D] or real-concat [B,S,2D].
        if x.dtype in (torch.float32, torch.float64) and x.shape[-1] == 2*self.cfg.dim:
            x = self.recombine_real_input(x)

        # x is now complex [B, S, D]
        for blk in self.blocks:
            x = await blk(x)

        x = self.ln_out(x)               # complex [B,S,D]
        x = await self.classifier(x)     # complex [B,S,C]
        return x.real                     # real logits [B,S,C]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sync wrapper for inference/training."""
        return asyncio.get_event_loop().run_until_complete(self._forward(x))

    def forward_onnx(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pure‐sync forward for ONNX: expects real [B,S,2D], returns real logits.
        """
        # recombine into complex
        D = self.cfg.dim
        x = torch.complex(x[..., :D], x[..., D:])
        for blk in self.blocks:
            x = blk.forward_sync(x)
        x = self.ln_out(x)
        x = self.classifier.forward_sync(x)
        return x.real

    async def train_step_async(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.train_mode(True)
        self.optimizer.zero_grad()

        logits = await self._forward(x)
        # if sequence‐level labels:
        if logits.dim() == 3 and y.dim() == 1:
            logits = logits[:, 0, :]

        loss = self.criterion(logits, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        return asyncio.get_event_loop().run_until_complete(
            self.train_step_async(x, y)
        )

    def train_mode(self, enabled: bool = True):
        for module in [*self.blocks, self.ln_out, self.classifier, self.router]:
            module.train(enabled)

    def train(self):
        self.train_mode(True)

    def eval_mode(self):
        self.train_mode(False)

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")
        return path

    def load(self, path: str):
        self.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
        return path

    def to_onnx(self, path: str):
        """
        Export the ONNX‐sync path (`forward_onnx`).
        """
        dummy = torch.randn(1, 10, 2*self.cfg.dim, device=self.device)
        torch.onnx.export(
            self.forward_onnx,    # sync entrypoint
            dummy,
            path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"Model exported to {path}")
        return path

    def load_onnx(self, path: str):
        model = onnx.load(path)
        onnx.checker.check_model(model)
        print(f"ONNX model loaded from {path}")
        return path

    def __repr__(self):
        return f"SillyAI(dim={self.cfg.dim}, num_heads={self.cfg.num_heads}, " \
               f"num_layers={self.cfg.num_layers}, mlp_dim={self.cfg.mlp_dim})"
    
    def __str__(self):
        return f"SillyAI Model\n" \
            f"  - Dimension: {self.cfg.dim}\n" \
            f"  - Number of Heads: {self.cfg.num_heads}\n" \
            f"  - Number of Layers: {self.cfg.num_layers}\n" \
            f"  - MLP Dimension: {self.cfg.mlp_dim}\n" \
            f"  - Device: {self.device}\n" \
            f"  - Dtype: {self.cfg.dtype}\n" \
            f"  - Dropout: {self.cfg.dropout}\n" \
            f"  - Precision: {self.cfg.precision}\n" \
            f"  - Cache Max Bytes: {self.cfg.cache_max_bytes}\n" \
            f"  - Decomp Threshold: {self.cfg.decomp_threshold}\n" \
            f"  - Decomp Gain Ratio: {self.cfg.decomp_gain_ratio}\n" \
            f"  - Optimizer: {self.optimizer}\n" \
            f"  - Loss Function: {self.criterion}\n"