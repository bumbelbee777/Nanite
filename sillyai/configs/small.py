from dataclasses import field

from sillyai.config import Modality, ModelConfig, PrecisionLevel


class SmallModelConfig(ModelConfig):
    """Configuration for a small SillyAI model."""

    # Input/Output dimensions
    input_dim: int = 1
    output_dim: int = 1
    d_model: int = 64
    d_ff: int = 256
    mlp_dim: int = 256

    # Transformer architecture
    n_heads: int = 4
    n_layers: int = 2
    max_seq_len: int = 64
    dropout: float = 0.1

    # Device and optimization
    device: str = "cpu"
    precision: PrecisionLevel = PrecisionLevel.TERNARY

    # Concept graph parameters
    concept_graph_size: int = 1024
    concept_decay_rate: float = 0.1

    # Modality support
    supported_modalities: set[Modality] = field(default_factory=lambda: {Modality.TEXT})

    # Plugin configuration
    enabled_plugins: list = field(default_factory=lambda: ["trainer"])
