import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Union
import numpy as np
from scipy.fft import dct, idct
import cv2

class ComplexBasisSet(nn.Module):
    """Learnable basis set of complex periodic functions."""
    def __init__(self, num_basis: int = 32, max_freq: float = 10.0):
        super().__init__()
        self.num_basis = num_basis
        self.max_freq = max_freq
        
        # Initialize basis frequencies and phases
        self.frequencies = nn.Parameter(torch.linspace(0.1, max_freq, num_basis))
        self.phases = nn.Parameter(torch.zeros(num_basis))
        self.amplitudes = nn.Parameter(torch.ones(num_basis))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate complex periodic functions for input x."""
        # x shape: [batch_size, seq_len]
        # Output shape: [batch_size, seq_len, num_basis]
        x_expanded = x.unsqueeze(-1)  # [batch_size, seq_len, 1]
        freqs = self.frequencies.unsqueeze(0).unsqueeze(0)  # [1, 1, num_basis]
        phases = self.phases.unsqueeze(0).unsqueeze(0)  # [1, 1, num_basis]
        amps = self.amplitudes.unsqueeze(0).unsqueeze(0)  # [1, 1, num_basis]
        
        # Generate complex periodic functions
        complex_funcs = amps * torch.exp(1j * (freqs * x_expanded + phases))
        return complex_funcs

class ComplexTokenEmbedding(nn.Module):
    """Convert tokens to complex periodic function representations."""
    def __init__(self, vocab_size: int, num_basis: int = 32, embedding_dim: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_basis = num_basis
        self.embedding_dim = embedding_dim
        
        # Basis set for periodic functions
        self.basis_set = ComplexBasisSet(num_basis)
        
        # Learnable weights for each token
        self.token_weights = nn.Parameter(torch.randn(vocab_size, num_basis, embedding_dim, dtype=torch.complex64))
        
        # Projection layers
        self.proj_in = nn.Linear(embedding_dim, num_basis)
        self.proj_out = nn.Linear(num_basis, embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to complex periodic function representations."""
        # x shape: [batch_size, seq_len]
        batch_size, seq_len = x.shape
        
        # Get basis functions
        x_norm = x.float() / self.vocab_size  # Normalize to [0, 1]
        basis_funcs = self.basis_set(x_norm)  # [batch_size, seq_len, num_basis]
        
        # Get token weights
        token_weights = self.token_weights[x]  # [batch_size, seq_len, num_basis, embedding_dim]
        
        # Combine basis functions with token weights
        complex_repr = torch.einsum('bsn,bsnd->bsd', basis_funcs, token_weights)
        
        return complex_repr

class ComplexTokenizer:
    """Tokenizer that converts text to complex periodic function representations."""
    def __init__(self, word2vec_path: Optional[str] = None, num_basis: int = 32):
        self.num_basis = num_basis
        self.word2idx = {}
        self.idx2word = []
        
        if word2vec_path:
            self._load_word2vec(word2vec_path)
            
    def _load_word2vec(self, path: str):
        """Load word2vec embeddings and convert to complex periodic functions."""
        # Load word2vec embeddings
        embeddings = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.array([float(x) for x in values[1:]], dtype=np.float32)
                embeddings[word] = vector
                
        # Convert to complex periodic functions
        for word, vec in embeddings.items():
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
                self.idx2word.append(word)
                
    def encode(self, text: str) -> torch.Tensor:
        """Convert text to token IDs."""
        words = text.lower().split()
        return torch.tensor([self.word2idx.get(word, 0) for word in words])
        
    def decode(self, ids: torch.Tensor) -> str:
        """Convert token IDs back to text."""
        return ' '.join([self.idx2word[idx] for idx in ids])

class ImageToComplex(nn.Module):
    """Convert images to complex periodic function representations."""
    def __init__(self, num_basis: int = 32, image_size: int = 224):
        super().__init__()
        self.num_basis = num_basis
        self.image_size = image_size
        self.basis_set = ComplexBasisSet(num_basis)
        
    def _preprocess_image(self, image: Union[str, bytes, np.ndarray]) -> torch.Tensor:
        """Convert image to tensor."""
        if isinstance(image, str):
            # Load from file
            img = cv2.imread(image)
        elif isinstance(image, bytes):
            # Load from bytes
            img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        else:
            img = image
            
        # Convert to RGB and resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img).float() / 255.0
        return img_tensor
        
    def _dct_transform(self, img: torch.Tensor) -> torch.Tensor:
        """Apply DCT to image."""
        # Convert to numpy for scipy DCT
        img_np = img.numpy()
        
        # Apply 2D DCT
        dct_coeffs = dct(dct(img_np, axis=0), axis=1)
        
        # Convert back to tensor
        return torch.from_numpy(dct_coeffs)
        
    def forward(self, image: Union[str, bytes, np.ndarray]) -> torch.Tensor:
        """Convert image to complex periodic function representation."""
        # Preprocess image
        img_tensor = self._preprocess_image(image)
        
        # Apply DCT
        dct_coeffs = self._dct_transform(img_tensor)
        
        # Generate basis functions
        x = torch.linspace(0, 1, self.image_size)
        basis_funcs = self.basis_set(x)
        
        # Combine DCT coefficients with basis functions
        complex_repr = torch.einsum('ij,ijk->ik', dct_coeffs, basis_funcs)
        
        return complex_repr 