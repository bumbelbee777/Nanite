import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from scipy.fft import dct, idct
import cv2
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
from threading import Lock
import time
import numba
from numba import jit, prange

from .ops import MultivectorOps
from .shared import get_shared_ops


@jit(nopython=True, parallel=True)
def _compute_basis_functions(
    x: np.ndarray, frequencies: np.ndarray, phases: np.ndarray, amplitudes: np.ndarray
) -> np.ndarray:
    """Compute complex-valued basis functions for token embedding."""
    n = len(x)
    result = np.zeros(n, dtype=np.complex64)
    for i in prange(n):
        for j in range(len(frequencies)):
            result[i] += amplitudes[j] * np.exp(
                1j * (frequencies[j] * x[i] + phases[j])
            )
    return result


@dataclass
class TokenFunction:
    """Complex-valued periodic function for token representation."""

    num_basis: int
    frequencies: Optional[np.ndarray] = None
    phases: Optional[np.ndarray] = None
    amplitudes: Optional[np.ndarray] = None
    embedding: Optional[np.ndarray] = None

    def to_tensor(self) -> torch.Tensor:
        """Convert function to tensor representation."""
        if self.embedding is None:
            return torch.zeros(self.num_basis, dtype=torch.complex64)
        return torch.from_numpy(self.embedding).to(torch.complex64)


class ComplexTokenStream:
    """Lock-free token stream with complex-valued periodic functions."""

    def __init__(
        self, buffer_size: int = 1024, num_workers: int = 4, num_basis: int = 32
    ):
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        self.num_basis = num_basis

        # Initialize queues
        self.token_queue = queue.Queue(maxsize=buffer_size)
        self.function_queue = queue.Queue(maxsize=buffer_size)

        # Initialize thread pool
        self.thread_pool = ThreadPoolExecutor(max_workers=num_workers)

        # Initialize token storage
        self.token_functions: Dict[int, TokenFunction] = {}
        self.token_lock = Lock()

    def push_tokens(
        self, tokens: List[int], functions: Optional[List[TokenFunction]] = None
    ):
        """Push tokens and their functions to the stream."""
        for i, token in enumerate(tokens):
            # Add token to queue
            while not self.token_queue.full():
                try:
                    self.token_queue.put(token, block=False)
                    break
                except queue.Full:
                    time.sleep(0.001)  # Spin wait

            # Add function if provided
            if functions and i < len(functions):
                with self.token_lock:
                    self.token_functions[token] = functions[i]

    def process_stream(
        self, batch_size: int = 32
    ) -> Tuple[List[int], List[TokenFunction]]:
        """Process token stream in batches."""
        tokens = []
        functions = []

        # Get tokens from queue
        while len(tokens) < batch_size:
            try:
                token = self.token_queue.get_nowait()
                tokens.append(token)

                # Get function if available
                with self.token_lock:
                    if token in self.token_functions:
                        functions.append(self.token_functions[token])
                    else:
                        functions.append(TokenFunction(self.num_basis))
            except queue.Empty:
                break

        return tokens, functions

    def close(self):
        """Close token stream."""
        self.thread_pool.shutdown()


class ComplexTokenizer(nn.Module):
    """Unified tokenizer with complex-valued periodic functions."""

    def __init__(
        self,
        w2v_path: Optional[str] = None,
        num_basis: int = 32,
        max_vocab_size: int = 50000,
        cache_size: int = 10000,
        quantize: bool = True,
        quantize_bits: int = 8,
        use_mmap: bool = True,
        chunk_size: int = 100000,
        subword_fallback: bool = True,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        buffer_size: int = 1024,
        device: str = "cpu",
    ):
        super().__init__()

        # Core parameters
        self.w2v_path = w2v_path
        self.num_basis = num_basis
        self.max_vocab_size = max_vocab_size
        self.cache_size = cache_size
        self.quantize = quantize
        self.quantize_bits = quantize_bits
        self.use_mmap = use_mmap
        self.chunk_size = chunk_size
        self.subword_fallback = subword_fallback
        self.device = device

        # Initialize data structures
        self.word2idx = {}
        self.idx2word = []
        self._model = None
        self._embedding = None

        # Initialize components
        self.basis_set = ComplexBasisSet(num_basis)
        self.token_embedding = ComplexTokenEmbedding(max_vocab_size, num_basis)
        self.token_stream = ComplexTokenStream(buffer_size, num_workers, num_basis)

        # Load word2vec if provided
        if w2v_path:
            self._load_word2vec(w2v_path)

    def _load_word2vec(self, path: str):
        """Load word2vec embeddings and convert to complex periodic functions."""
        # Load word2vec embeddings
        embeddings = {}
        with open(path, "r", encoding="utf-8") as f:
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

                # Create token function
                func = self._create_token_function(word, vec)
                self.token_stream.token_functions[self.word2idx[word]] = func

    def _create_token_function(self, word: str, vec: np.ndarray) -> TokenFunction:
        """Create complex-valued periodic function for a token."""
        func = TokenFunction(self.num_basis)

        # Use DCT to get frequency components
        dct_coeffs = dct(vec)

        # Set frequencies and phases
        func.frequencies = np.linspace(0.1, self.num_basis, self.num_basis)
        func.phases = np.angle(dct_coeffs[: self.num_basis])
        func.amplitudes = np.abs(dct_coeffs[: self.num_basis])

        # Create complex embedding
        x = np.linspace(0, 1, self.num_basis)
        func.embedding = _compute_basis_functions(
            x, func.frequencies, func.phases, func.amplitudes
        )

        return func

    def encode(self, text: str) -> Tuple[torch.Tensor, List[TokenFunction]]:
        """Convert text to token IDs and functions."""
        words = text.lower().split()
        ids = []
        functions = []

        for word in words:
            if word in self.word2idx:
                idx = self.word2idx[word]
                ids.append(idx)

                # Get function from stream
                with self.token_stream.token_lock:
                    if idx in self.token_stream.token_functions:
                        functions.append(self.token_stream.token_functions[idx])
                    else:
                        functions.append(TokenFunction(self.num_basis))
            else:
                # Handle unknown tokens
                ids.append(0)  # Use first token as unknown
                functions.append(TokenFunction(self.num_basis))

        return torch.tensor(ids, device=self.device), functions

    def decode(
        self, ids: torch.Tensor, functions: Optional[List[TokenFunction]] = None
    ) -> str:
        """Convert token IDs and functions back to text."""
        words = []
        for i, idx in enumerate(ids):
            word = self.idx2word[idx] if idx < len(self.idx2word) else "<unk>"
            words.append(word)

            # Update function if provided
            if functions and i < len(functions):
                with self.token_stream.token_lock:
                    self.token_stream.token_functions[idx] = functions[i]

        return " ".join(words)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through tokenizer."""
        # Get token functions
        functions = []
        for idx in x:
            with self.token_stream.token_lock:
                if idx in self.token_stream.token_functions:
                    functions.append(self.token_stream.token_functions[idx])
                else:
                    functions.append(TokenFunction(self.num_basis))

        # Convert functions to tensor
        func_tensors = [func.to_tensor() for func in functions]
        return torch.stack(func_tensors)


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

    def __init__(
        self,
        vocab_size: int,
        num_basis: int = 32,
        embedding_dim: int = 128,
        ops: Optional[MultivectorOps] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_basis = num_basis
        self.embedding_dim = embedding_dim
        self.ops = ops or get_shared_ops()

        # Basis set for periodic functions
        self.basis_set = ComplexBasisSet(num_basis)

        # Initialize with smaller chunks to avoid memory issues
        chunk_size = min(1000, vocab_size)  # Process 1000 tokens at a time
        self.token_weights = nn.ParameterList()

        for i in range(0, vocab_size, chunk_size):
            chunk_vocab_size = min(chunk_size, vocab_size - i)
            # Initialize with smaller values and proper scaling
            weights = (
                torch.randn(
                    chunk_vocab_size, num_basis, embedding_dim, dtype=torch.complex64
                )
                * 0.02
            )
            self.token_weights.append(nn.Parameter(weights))

        # Projection layers with reduced dimensions
        self.proj_in = nn.Linear(embedding_dim, num_basis)
        self.proj_out = nn.Linear(num_basis, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to complex periodic function representations."""
        # x shape: [batch_size, seq_len]
        batch_size, seq_len = x.shape

        # Get basis functions
        x_norm = x.float() / self.vocab_size  # Normalize to [0, 1]
        basis_funcs = self.basis_set(x_norm)  # [batch_size, seq_len, num_basis]

        # Process tokens in chunks to save memory
        complex_repr = []
        chunk_size = len(self.token_weights[0])

        for i in range(0, self.vocab_size, chunk_size):
            # Get mask for current chunk
            mask = (x >= i) & (x < i + chunk_size)
            if not mask.any():
                continue

            # Get token weights for chunk
            chunk_idx = i // chunk_size
            token_weights = self.token_weights[chunk_idx]

            # Adjust indices for chunk
            chunk_x = x[mask] - i

            # Get weights for masked tokens
            weights = token_weights[chunk_x]

            # Get basis functions for masked tokens
            chunk_basis = basis_funcs[mask]

            # Use MultivectorOps for complex tensor operations
            chunk_repr = self.ops.complex_matmul(chunk_basis, weights)

            # Create output tensor for chunk
            chunk_output = torch.zeros(
                mask.shape + (self.embedding_dim,),
                dtype=torch.complex64,
                device=x.device,
            )
            chunk_output[mask] = chunk_repr
            complex_repr.append(chunk_output)

        # Combine chunks using MultivectorOps
        return self.ops.complex_sum(complex_repr)


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
        complex_repr = torch.einsum("ij,ijk->ik", dct_coeffs, basis_funcs)

        return complex_repr
