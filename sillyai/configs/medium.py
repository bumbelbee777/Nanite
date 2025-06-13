from dataclasses import field

import torch

from sillyai.config import Modality, ModelConfig, PrecisionLevel


class MediumModelConfig(ModelConfig):
    """Configuration for a medium-sized SillyAI model."""

    # Input/Output dimensions
    input_dim: int = 1
    output_dim: int = 1
    d_model: int = 256
    d_ff: int = 1024
    mlp_dim: int = 1024

    # Transformer architecture
    n_heads: int = 8
    n_layers: int = 4
    max_seq_len: int = 128
    dropout: float = 0.1

    # Device and optimization
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: PrecisionLevel = PrecisionLevel.FP8

    # Concept graph parameters
    concept_graph_size: int = 8192
    concept_decay_rate: float = 0.1

    # Modality support
    supported_modalities: set[Modality] = field(
        default_factory=lambda: {Modality.TEXT, Modality.IMAGE},
    )

    # Plugin configuration
    enabled_plugins: list = field(default_factory=lambda: ["trainer", "visualizer"])
