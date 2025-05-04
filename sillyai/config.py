from dataclasses import dataclass
from typing import Optional

from .ops import PrecisionLevel

@dataclass
class ModelConfig:
    dim:               int
    num_heads:         int
    mlp_dim:           int
    num_layers:        int

    dropout:           float   = 0.1

    precision:         PrecisionLevel = PrecisionLevel.INT4
    cache_max_bytes:   int             = 1 << 26
    decomp_threshold:  int             = 1_000_000
    decomp_gain_ratio: float           = 0.5

    device:           Optional[str]     = None
    dtype:            Optional[str]     = None