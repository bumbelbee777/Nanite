import os
import torch

from .model import SillyAI
from .config import ModelConfig

def main():
    # Load configuration
    config = ModelConfig(
        dim=128,
        num_heads=8,
        mlp_dim=512,
        num_layers=6,
        dropout=0.1,
        precision="complex64",
        cache_max_bytes=1 << 26,
        decomp_threshold=1_000_000,
        decomp_gain_ratio=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="complex64"
    )

    # Initialize model
    model = SillyAI(config, num_classes=10)

    # Print model summary
    print(model)
    print(f"Model is on device: {model.device}")
    print(f"Model dtype: {model.dtype}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Model cache max bytes: {model.ops.cache_max_bytes}")
    print(f"Model precision: {model.ops.quant_precision}")