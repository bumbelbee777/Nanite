from dataclasses import field

import torch

from sillyai.config import Modality, ModelConfig, PrecisionLevel


class LargeModelConfig(ModelConfig):
    """Configuration for a large SillyAI model."""

    # Input/Output dimensions
    input_dim: int = 1
    output_dim: int = 1
    d_model: int = 512
    d_ff: int = 2048
    mlp_dim: int = 2048

    # Transformer architecture
    n_heads: int = 8
    n_layers: int = 6
    max_seq_len: int = 256
    dropout: float = 0.1

    # Device and optimization
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: PrecisionLevel = PrecisionLevel.FP16
    mixed_precision: bool = True

    # Concept graph parameters
    concept_graph_size: int = 16384
    concept_decay_rate: float = 0.1

    # Modality support
    supported_modalities: set[Modality] = field(
        default_factory=lambda: {Modality.TEXT, Modality.IMAGE, Modality.AUDIO},
    )

    # Plugin configuration
    enabled_plugins: list = field(
        default_factory=lambda: ["trainer", "visualizer", "profiler"],
    )
