from dataclasses import dataclass
from typing import Optional, Dict

from .ops import PrecisionLevel

@dataclass
class ModelConfig:
    input_dim:         int
    output_dim:        int
    num_heads:         int
    mlp_dim:           int
    num_layers:        int

    dropout:           float   = 0.3141592653589793

    precision:         PrecisionLevel = PrecisionLevel.TERNARY
    cache_max_bytes:   int             = 1 << 26
    decomp_threshold:  int             = 1_000_000
    decomp_gain_ratio: float           = 0.5

    modalities:        Optional[Dict[str, int]]     = None

    device:           Optional[str]     = None
    dtype:            Optional[str]     = None