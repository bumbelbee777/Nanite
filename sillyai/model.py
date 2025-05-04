import asyncio

import torch
import torch.nn as nn
import torch.optim as optim
import onnx

from .config import ModelConfig
from .ops import OptimizedComplexOps
from .core import (
    TransformerBlock,
    ComplexLayerNorm,
    LinearLayer,
    FeatureRouter
)

class SillyAI:
    def __init__(self, cfg: ModelConfig, num_classes: int):
        """
        A silly complex-valued multimodal model, fully async under the hood,
        with dynamic activation routing and InfiniToeplitz attention.
        """
        self.cfg = cfg

        # 1) Shared Optimized Ops
        self.ops = OptimizedComplexOps(
            cache_max_bytes=cfg.cache_max_bytes,
            quant_precision=cfg.precision,
            decomp_threshold=cfg.decomp_threshold,
            decomp_gain_ratio=cfg.decomp_gain_ratio,
        )

        # 2) Shared Feature Router (decides static vs dynamic FFN per token)
        self.router = FeatureRouter(
            d_model=cfg.dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout
        )

        # 3) Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=cfg.dim,
                num_heads=cfg.num_heads,
                mlp_dim=cfg.mlp_dim,
                ops=self.ops,
                router=self.router,
                use_learnable_act=True,      # toggle branch inside each block
            )
            for _ in range(cfg.num_layers)
        ])

        # 4) Output head: layer-norm + linear to real logits
        self.ln_out     = ComplexLayerNorm(cfg.dim)
        self.classifier = LinearLayer(cfg.dim, num_classes, self.ops)

        # 5) Move everything to the desired device & dtype
        device = torch.device(cfg.device or "cpu")
        dtype  = torch.complex64 if cfg.dtype == "complex64" else torch.complex128
        for module in [*self.blocks, self.ln_out, self.classifier, self.router]:
            module.to(device=device, dtype=dtype)
        self.device = device

        # 6) Optimizer & criterion
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

    def parameters(self):
        """Gather all parameters for the optimizer."""
        params = []
        for blk in self.blocks:
            params += list(blk.parameters())
        params += list(self.ln_out.parameters())
        params += list(self.classifier.parameters())
        # router has no learnable params for routing but include if it does in future
        params += list(self.router.parameters())
        return params

    async def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Async forward through transformer blocks,
        then produce real logits.
        """
        # x: complex tensor of shape [B, S, D]
        for blk in self.blocks:
            x = await blk(x)

        # final norm + classifier
        x = self.ln_out(x)               # complex [B, S, D]
        x = await self.classifier(x)     # complex [B, S, C]
        return x.real                     # real logits [B, S, C]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Synchronous wrapper around the async forward.
        """
        return asyncio.get_event_loop().run_until_complete(self._forward(x))

    async def train_step_async(self,
                               x: torch.Tensor,
                               y: torch.Tensor
                              ) -> float:
        """
        An async train step over one batch.

        x: complex input [B, S, D]
        y: integer class labels [B, S] or [B,]
        """
        self.train_mode(True)
        self.optimizer.zero_grad()

        logits = await self._forward(x)  # [B, S, C]
        # if we're doing sequence classification with per-seq labels
        if logits.dim() == 3 and y.dim() == 1:
            logits = logits[:, 0, :]     # [B, C]

        loss = self.criterion(logits, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Synchronous wrapper around train_step_async.
        """
        return asyncio.get_event_loop().run_until_complete(
            self.train_step_async(x, y)
        )

    def train_mode(self, enabled: bool = True):
        """Toggle train/eval on all submodules."""
        for module in [*self.blocks, self.ln_out, self.classifier, self.router]:
            module.train(enabled)

    def eval_mode(self):
        """Set all submodules to eval mode."""
        self.train_mode(False)

    def save(self, path: str):
        """Save the model to a file."""
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")
        return path
    
    def load(self, path: str):
        """Load the model from a file."""
        self.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
        return path
    
    def to_onnx(self, path: str):
        """Export the model to ONNX format."""
        dummy_input = torch.randn(1, 10, self.cfg.dim, device=self.device)
        torch.onnx.export(
            self,
            dummy_input,
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
        """Load the model from ONNX format."""
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        print(f"Model loaded from {path}")
        return path
