import re
import math
from gensim.models import KeyedVectors
from typing import List, Dict, Optional, Union
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
from PIL import Image
import torchvision.transforms as transforms

from .config import ModelConfig
from .core import LinearLayer

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

class Word2VecTokenizer:
    """
    Enhanced word-level tokenizer with pretrained Word2Vec embeddings.
    
    Features:
    - Handles both binary (.bin) and text (.txt) Word2Vec formats
    - Customizable unknown token handling
    - Efficient batch processing
    - PyTorch integration
    - Subword fallback mechanism
    """
    
    def __init__(
        self,
        w2v_path: Union[str, Path],
        unk_token: str = "<UNK>",
        pad_token: str = "<PAD>",
        lowercase: bool = True,
        subword_fallback: bool = False,
        device: str = "cpu"
    ):
        """
        Args:
            w2v_path: Path to Word2Vec model file
            unk_token: Token for unknown words
            pad_token: Token for padding
            lowercase: Convert text to lowercase
            subword_fallback: Try subword matching for OOV words
            device: Target device for PyTorch tensors
        """
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.lowercase = lowercase
        self.subword_fallback = subword_fallback
        self.device = device
        
        # Load Word2Vec model
        self._load_word2vec(w2v_path)
        self._build_vocab()
        self._create_embedding_matrix()
        
    def _load_word2vec(self, path: Union[str, Path]):
        """Handle different Word2Vec formats"""
        path = str(path)
        binary = path.endswith('.bin')
        
        try:
            self.model = KeyedVectors.load_word2vec_format(path, binary=binary)
            self.vector_size = self.model.vector_size
        except Exception as e:
            raise ValueError(f"Failed to load Word2Vec model from {path}: {str(e)}")

    def _build_vocab(self):
        """Build vocabulary with special tokens"""
        self.index_to_word = [self.pad_token, self.unk_token] + list(self.model.index_to_key)
        self.word_to_index = {w: i for i, w in enumerate(self.index_to_word)}
        
        # For subword fallback
        if self.subword_fallback:
            self._build_subword_index()

    def _build_subword_index(self):
        """Create n-gram index for subword fallback"""
        self.subword_index = {}
        for word in self.model.index_to_key:
            for i in range(len(word)):
                ngram = word[i:i+3]  # Use 3-grams
                if ngram not in self.subword_index:
                    self.subword_index[ngram] = []
                self.subword_index[ngram].append(word)

    def _create_embedding_matrix(self):
        """Initialize embedding matrix with special tokens"""
        # Pad token gets zeros, UNK gets random initialization
        pad_vec = np.zeros(self.vector_size, dtype=np.float32)
        unk_vec = np.random.normal(size=self.vector_size).astype(np.float32)
        
        # Stack all embeddings
        self.embedding_matrix = np.vstack([
            pad_vec,
            unk_vec,
            np.array([self.model[w] for w in self.model.index_to_key])
        ])
        
        # Convert to PyTorch tensor
        self.embedding_tensor = torch.from_numpy(self.embedding_matrix).to(self.device)

    def tokenize(self, text: str) -> List[str]:
        """Enhanced tokenization with optional subword fallback"""
        if self.lowercase:
            text = text.lower()
            
        # Improved regex pattern
        tokens = re.findall(r"\b\w+(?:'\w+)?\b", text)
        
        if self.subword_fallback:
            tokens = [self._handle_subword(tok) for tok in tokens]
            
        return tokens

    def _handle_subword(self, word: str) -> str:
        """Attempt subword matching for OOV words"""
        if word in self.word_to_index:
            return word
            
        # Try n-gram matching
        for i in range(len(word)):
            ngram = word[i:i+3]
            if ngram in self.subword_index:
                candidates = self.subword_index[ngram]
                # Return first match (could implement better scoring)
                return candidates[0]
                
        return self.unk_token

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = False,
        return_tensors: str = None
    ) -> Union[List[int], torch.Tensor]:
        """
        Enhanced encoding with padding options
        
        Args:
            text: Input text to encode
            max_length: Truncate/pad to this length
            padding: Whether to pad sequences
            return_tensors: 'pt' for PyTorch tensor, None for list
            
        Returns:
            List of indices or PyTorch tensor
        """
        tokens = self.tokenize(text)
        unk_idx = self.word_to_index[self.unk_token]
        indices = [self.word_to_index.get(tok, unk_idx) for tok in tokens]
        
        # Handle truncation/padding
        if max_length is not None:
            if len(indices) > max_length:
                indices = indices[:max_length]
            elif padding:
                pad_idx = self.word_to_index[self.pad_token]
                indices += [pad_idx] * (max_length - len(indices))
                
        if return_tensors == 'pt':
            return torch.tensor(indices, device=self.device)
        return indices

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Batch encoding for multiple texts
        
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        encoded = [self.encode(text, max_length, padding, 'pt') for text in texts]
        input_ids = torch.stack(encoded)
        
        # Create attention mask
        attention_mask = (input_ids != self.word_to_index[self.pad_token]).long()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

    def decode(self, indices: Union[List[int], torch.Tensor]) -> List[str]:
        """Decode indices back to text"""
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
            
        max_idx = len(self.index_to_word) - 1
        return [
            self.index_to_word[i] if 0 <= i <= max_idx else self.unk_token
            for i in indices
            if i != self.word_to_index[self.pad_token]  # Skip padding
        ]

    def embed(
        self,
        inputs: Union[str, List[int], torch.Tensor],
        return_tensors: str = 'pt'
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Flexible embedding lookup
        
        Args:
            inputs: Can be text, indices, or tensor
            return_tensors: 'pt' for PyTorch, 'np' for numpy
            
        Returns:
            Embeddings matrix (num_tokens, dim)
        """
        if isinstance(inputs, str):
            indices = self.encode(inputs, return_tensors='pt')
        elif isinstance(inputs, list):
            indices = torch.tensor(inputs, device=self.device)
        else:
            indices = inputs.to(self.device)
            
        embeddings = self.embedding_tensor[indices]
        
        if return_tensors == 'np':
            return embeddings.cpu().numpy()
        return embeddings

    def get_vocab_size(self) -> int:
        """Total vocabulary size including special tokens"""
        return len(self.index_to_word)

    def get_embedding_matrix(self) -> torch.Tensor:
        """Get the full embedding matrix as PyTorch tensor"""
        return self.embedding_tensor

    def save_vocab(self, path: Union[str, Path]):
        """Save vocabulary to file"""
        with open(path, 'w') as f:
            for word in self.index_to_word:
                f.write(f"{word}\n")

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        vocab_path: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """
        Create tokenizer from pretrained files
        
        Args:
            path: Path to Word2Vec model
            vocab_path: Optional custom vocabulary file
            **kwargs: Additional init arguments
        """
        tokenizer = cls(path, **kwargs)
        
        if vocab_path is not None:
            with open(vocab_path) as f:
                custom_vocab = [line.strip() for line in f]
            tokenizer.index_to_word = custom_vocab
            tokenizer.word_to_index = {w: i for i, w in enumerate(custom_vocab)}
            
        return tokenizer

class ModalityManager(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.modalities = nn.ModuleDict()
        
    def add_modality(self, name: str, modality_config: dict):
        if name == 'text':
            self._init_text_modality(modality_config)
        elif name == 'image':
            self._init_image_modality(modality_config)
            
    def _init_text_modality(self, config):
        self.modalities['text_tokenizer'] = Word2VecTokenizer(
            w2v_path=config['w2v_path'],
            unk_token=config.get('unk_token', '<UNK>'),
            pad_token=config.get('pad_token', '<PAD>'),
            lowercase=config.get('lowercase', True),
            device=self.config.device
        )
        self.modalities['text_proj'] = LinearLayer(
            self.modalities['text_tokenizer'].get_embedding_matrix().shape[1],
            self.config.d_model,
            factorized=self.config.factorized_linear
        )
        
    def _init_image_modality(self, config):
        self.modalities['image_encoder'] = ImageModality(
            input_channels=config.get('input_channels', 3),
            output_dim=self.config.d_model,
            input_size=config.get('input_size', (224, 224)),
            dropout_rate=config.get('dropout_rate', 0.1)
        )
        self.modalities['image_proj'] = LinearLayer(
            self.config.d_model,
            self.config.d_model,
            factorized=self.config.factorized_linear
        )
        
    def process_text(self, text):
        if isinstance(text, str):
            text = [text]
        encoded = self.modalities['text_tokenizer'].batch_encode(text)
        text_emb = self.modalities['text_tokenizer'].embed(encoded['input_ids'])
        return self.modalities['text_proj'](text_emb)
        
    def process_image(self, images):
        img_features = self.modalities['image_encoder'](images)
        return self.modalities['image_proj'](img_features)