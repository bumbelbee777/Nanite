from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union, Set
from enum import Enum
import torch


class PrecisionLevel(Enum):
    TERNARY = "ternary"
    INT4 = "int4"
    FP4 = "fp4"
    FP8 = "fp8"
    FP16 = "fp16"


class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


@dataclass
class ModelConfig:
    """Configuration for SillyAI model."""

    # Input/Output dimensions
    input_dim: int = 1
    output_dim: int = 1
    d_model: int = 512
    d_ff: int = 2048
    mlp_dim: int = 2048  # Dimension for MLP layers

    # Vocabulary and tokenization
    vocab_size: int = 32000
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3

    # Transformer architecture
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    max_seq_len: int = 1024

    # Complex periodic function parameters
    num_basis: int = 32  # Number of basis functions
    basis_freq_min: float = 0.1  # Minimum frequency for basis functions
    basis_freq_max: float = 10.0  # Maximum frequency for basis functions

    # Image processing parameters
    image_size: int = 224  # Input image size
    image_channels: int = 3  # Number of image channels

    # Audio processing parameters
    audio_sample_rate: int = 44100  # Audio sample rate
    audio_channels: int = 1  # Number of audio channels (1=mono, 2=stereo)
    audio_n_fft: int = 2048  # FFT window size
    audio_hop_length: int = 512  # Number of samples between windows
    audio_n_mels: int = 128  # Number of mel bands
    audio_max_length: int = 30  # Maximum audio length in seconds

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_epochs: int = 100
    early_stop_patience: int = 5
    gradient_clip_val: float = 1.0

    # Concept graph parameters
    concept_graph_size: int = 10000
    concept_embedding_dim: int = 256
    concept_decay_rate: float = 0.1

    # Device and optimization
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: PrecisionLevel = PrecisionLevel.FP8
    mixed_precision: bool = True
    gradient_clip: float = 1.0

    # Modality support
    supported_modalities: Set[Modality] = field(default_factory=lambda: {Modality.TEXT})

    # Plugin configuration
    enabled_plugins: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.enabled_plugins:
            self.enabled_plugins = ["trainer", "visualizer"]

        # Validate modality-specific parameters
        if Modality.IMAGE in self.supported_modalities:
            if self.image_size <= 0:
                raise ValueError(
                    "image_size must be positive when IMAGE modality is supported"
                )
            if self.image_channels not in [1, 3]:
                raise ValueError("image_channels must be 1 (grayscale) or 3 (RGB)")

        if Modality.AUDIO in self.supported_modalities:
            if self.audio_sample_rate <= 0:
                raise ValueError(
                    "audio_sample_rate must be positive when AUDIO modality is supported"
                )
            if self.audio_channels not in [1, 2]:
                raise ValueError("audio_channels must be 1 (mono) or 2 (stereo)")
            if self.audio_n_fft <= 0:
                raise ValueError("audio_n_fft must be positive")
            if self.audio_hop_length <= 0:
                raise ValueError("audio_hop_length must be positive")
            if self.audio_n_mels <= 0:
                raise ValueError("audio_n_mels must be positive")
            if self.audio_max_length <= 0:
                raise ValueError("audio_max_length must be positive")

        # Validate transformer parameters
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_ff < self.d_model:
            raise ValueError("d_ff must be greater than or equal to d_model")
        if self.mlp_dim < self.d_model:
            raise ValueError("mlp_dim must be greater than or equal to d_model")

        # Validate vocabulary parameters
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if not all(
            isinstance(x, int)
            for x in [
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
                self.unk_token_id,
            ]
        ):
            raise ValueError("token IDs must be integers")

        # Validate concept graph parameters
        if self.concept_graph_size <= 0:
            raise ValueError("concept_graph_size must be positive")
        if not 0 <= self.concept_decay_rate <= 1:
            raise ValueError("concept_decay_rate must be between 0 and 1")
