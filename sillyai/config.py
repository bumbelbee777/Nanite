from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union
from enum import Enum
import torch

class PrecisionLevel(Enum):
    TERNARY, INT4, FP4, FP8, FP16 = range(1, 6)

@dataclass
class ModelConfig:
    # Core model dimensions
    input_dim: int
    output_dim: int
    d_model: int = field(default=None)  # Will be set to input_dim if None
    d_ff: int = field(default=None)     # Will be set to 4 * d_model if None
    mlp_dim: int = field(default=None)  # Will be set to d_ff if None
    
    # Transformer architecture
    num_heads: int = 8
    num_layers: int = 6
    max_seq_len: int = 512
    dropout: float = 0.1
    
    # Complex number handling
    precision: PrecisionLevel = PrecisionLevel.TERNARY
    
    # Memory and caching
    cache_max_bytes: int = 1 << 26  # 64MB
    concept_graph_size: int = 1000  # Maximum number of concepts in the graph
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Plugin configuration
    enabled_plugins: List[str] = field(default_factory=list)
    
    # Device and dtype
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    dtype: str = "complex64"

    
    def __post_init__(self):
        # Set default values for d_model and d_ff
        if self.d_model is None:
            self.d_model = self.input_dim
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.mlp_dim is None:
            self.mlp_dim = self.d_ff
            
        # Validate dimensions
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        assert self.input_dim > 0, "input_dim must be positive"
        assert self.output_dim > 0, "output_dim must be positive"
        assert self.concept_graph_size > 0, "concept_graph_size must be positive"
