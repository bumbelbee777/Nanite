from .model import SillyAI
from .config import ModelConfig, Modality, PrecisionLevel
from .ops import MultivectorOps
from .plugins.trainer import SillyAITrainerPlugin
from .plugins.visualizer import SillyAIVisualizerPlugin

__all__ = [
    "SillyAI",
    "ModelConfig",
    "Modality",
    "PrecisionLevel",
    "MultivectorOps",
    "SillyAITrainerPlugin",
    "SillyAIVisualizerPlugin",
]
