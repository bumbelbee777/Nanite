import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
from PIL import Image
import torchvision.transforms as transforms

class ImageModality(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        output_dim: int = 256,
        input_size: Optional[Union[int, Tuple[int, int]]] = None,
        dropout_rate: float = 0.3,
        use_batchnorm: bool = True,
        activation: str = 'gelu',
        adaptive_pooling: bool = True,
        min_resolution: int = 32
    ):
        """
        Args:
            input_size: Optional (height, width). If None, will adapt to input.
                       If int, will use (size, size). Can be overridden during forward pass.
            adaptive_pooling: If True, uses adaptive pooling to handle varying resolutions
            min_resolution: Minimum resolution for safety checks
        """
        super().__init__()
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.adaptive_pooling = adaptive_pooling
        self.min_resolution = min_resolution
        
        # Set default/initial input size
        self.input_size = self._validate_input_size(input_size) if input_size else None
        
        # Core CNN architecture
        self.conv_blocks = nn.Sequential(
            self._conv_block(input_channels, 64, 3),
            self._conv_block(64, 128, 3),
            self._conv_block(128, 256, 3),
            self._conv_block(256, 512, 3)
        )
        
        # Adaptive pooling if enabled
        if adaptive_pooling:
            self.pool = nn.AdaptiveAvgPool2d((4, 4))
            self.fc = nn.Linear(512 * 4 * 4, output_dim)
        else:
            self.pool = nn.MaxPool2d(2)
            self.fc = None  # Will be initialized on first forward
            
        self.dropout = nn.Dropout2d(dropout_rate)
        self.activation = self._get_activation(activation)

    def _conv_block(self, in_c, out_c, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_c),
            self._get_activation('leaky_relu'),
            nn.MaxPool2d(2)
        )

    def _get_activation(self, name):
        activations = {
            'relu': nn.ReLU(),
            'leaky': nn.LeakyReLU(0.1),
            'gelu': nn.GELU(),
            'silu': nn.SiLU()
        }
        return activations.get(name.lower(), nn.ReLU())

    def _validate_input_size(self, size):
        if isinstance(size, int):
            return (size, size)
        assert len(size) == 2 and all(s >= self.min_resolution for s in size), \
            f"Input size must be >= {self.min_resolution} in both dimensions"
        return tuple(size)

    def forward(self, x: torch.Tensor, input_size: Optional[Tuple[int, int]] = None):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            input_size: Optional manual override (height, width)
        """
        if input_size:
            self.input_size = self._validate_input_size(input_size)
            
        # Safety checks
        assert x.dim() == 4, "Input must be 4D tensor"
        if self.input_size and not self.adaptive_pooling:
            assert x.shape[-2:] == self.input_size, \
                f"Input size {x.shape[-2:]} doesn't match expected {self.input_size}"
        
        # Forward pass
        x = self.conv_blocks(x)
        
        if not self.adaptive_pooling:
            # Initialize FC layer on first run if needed
            if self.fc is None:
                self._init_fc(x)
            x = x.flatten(1)
        else:
            x = self.pool(x).flatten(1)
            
        return self.fc(x)

    def _init_fc(self, sample_tensor):
        with torch.no_grad():
            flattened_dim = sample_tensor.flatten(1).shape[1]
        self.fc = nn.Linear(flattened_dim, self.output_dim).to(sample_tensor.device)
        nn.init.kaiming_normal_(self.fc.weight)

    @staticmethod
    def load_image(
        image_path: str,
        target_size: Optional[Tuple[int, int]] = None,
        crop_to_size: Optional[Tuple[int, int]] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Flexible image loading with auto-resizing and optional cropping
        
        Args:
            target_size: Resize to (H, W) while maintaining aspect ratio
            crop_to_size: Center crop after resizing (if different from target_size)
        """
        img = Image.open(image_path).convert('RGB')
        
        # Resize first (maintaining aspect ratio)
        if target_size:
            img.thumbnail((target_size[1], target_size[0]))  # PIL uses (W, H)
        
        # Center crop if requested
        if crop_to_size:
            crop = transforms.CenterCrop(crop_to_size)
            img = crop(img)
        
        # Convert to tensor
        transform = [transforms.ToTensor()]
        if normalize:
            transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225]))
        return transforms.Compose(transform)(img).unsqueeze(0)

    def get_expected_shape(self) -> Tuple[int, int, int]:
        """Returns expected (channels, height, width) for input verification"""
        return (self.input_channels, *self.input_size) if self.input_size else None