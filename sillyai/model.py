import asyncio
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .config import ModelConfig
from .ops import OptimizedComplexOps, PrecisionLevel
from .core import TransformerBlock, ComplexLayerNorm, LinearLayer

class SillyAI:
    def __init__(self, cfg: ModelConfig, num_classes: int):
        """
        A “silly” end-to-end complex-valued transformer with
        a final classifier head.
        """
        self.cfg = cfg

        # 1) Shared optimized ops
        self.ops = OptimizedComplexOps(
            cache_max_bytes=cfg.cache_max_bytes,
            quant_precision=cfg.precision,
            decomp_threshold=cfg.decomp_threshold,
            decomp_gain_ratio=cfg.decomp_gain_ratio,
        )

        # 2) Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.dim, cfg.num_heads, cfg.mlp_dim, self.ops)
            for _ in range(cfg.num_layers)
        ])

        # 3) Output head: layer-norm + linear to real logits
        self.ln_out = ComplexLayerNorm(cfg.dim)
        self.classifier = LinearLayer(cfg.dim, num_classes, self.ops)

        # 4) Move to device & dtype
        device = torch.device(cfg.device or "cpu")
        dtype  = torch.complex64 if cfg.dtype == "complex64" else torch.complex128
        self.device = device
        for module in [*self.blocks, self.ln_out, self.classifier]:
            module.to(device=device, dtype=dtype)

        # 5) Optimizer & criterion (real-valued targets)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()

    def parameters(self):
        """Gather all parameters for the optimizer."""
        params = []
        for blk in self.blocks:
            params += list(blk.parameters())
        params += list(self.ln_out.parameters())
        params += list(self.classifier.parameters())
        return params

    async def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Async forward through transformer blocks,
        then produce real logits.
        """
        # x: complex tensor (B, S, D)
        for blk in self.blocks:
            x = await blk(x)

        # final norm
        x = self.ln_out(x)               # complex (B, S, D)
        # classifier: complex → complex
        x = await self.classifier(x)     # (B, S, num_classes)
        # we produce real logits by taking real part
        return x.real                     # (B, S, num_classes)

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

        x: complex input (B, S, D)
        y: integer class labels (B, S) or (B,) depending on task
        """
        self.train_mode(True)
        self.optimizer.zero_grad()

        # forward
        logits = await self._forward(x)               # (B, S, C)
        # if labels are (B,) but logits are (B, S, C), take e.g. first token
        if logits.dim() == 3 and y.dim() == 1:
            logits = logits[:, 0, :]                  # (B, C)

        loss = self.criterion(logits, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_step(self,
                   x: torch.Tensor,
                   y: torch.Tensor
                   ) -> float:
        """
        Synchronous wrapper around train_step_async.
        """
        return asyncio.get_event_loop().run_until_complete(
            self.train_step_async(x, y)
        )

    def train_mode(self, enabled: bool = True):
        """Toggle train/eval on all submodules."""
        for module in [*self.blocks, self.ln_out, self.classifier]:
            module.train(enabled)

    def eval_mode(self):
        self.train_mode(False)
