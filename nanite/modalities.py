import gc
import multiprocessing as mp
import re
import time
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
import mmap
import pickle
import hashlib
from pathlib import Path

import bitarray
import numpy as np
import torch
import torch.nn as nn
from numba import jit
from scipy.fft import dct
import gensim
from gensim.models import KeyedVectors

from .complex_tokens import (
    ComplexBasisSet,
    ComplexTokenEmbedding,
    ComplexTokenStream,
    TokenFunction,
    _compute_basis_functions,
)
from .config import ModelConfig, PrecisionLevel
from .core import LinearLayer
from .ops import MixedPrecisionRouter, MultivectorOps, Quantizer, TensorCache, TensorHasher, PRECISION_CONFIGS
from .shared import get_shared_ops
from .utils import RingBuffer


@jit(nopython=True)
def _process_tokens_numba(tokens: np.ndarray, vocab: np.ndarray) -> np.ndarray:
    """Process tokens using Numba for faster tokenization."""
    n = len(tokens)
    result = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(len(vocab)):
            if tokens[i] == vocab[j]:
                result[i] = j
                break
    return result


@dataclass
class TokenView:
    """Bit-packed token view for zero-copy slicing."""

    data: np.ndarray  # Bit-packed token data
    start: int  # Start bit position
    length: int  # Length in bits

    def to_tokens(self) -> list[int]:
        """Convert bit-packed view to token list."""
        tokens = []
        for i in range(self.length):
            if self.data[self.start + i]:
                tokens.append(i)
        return tokens

    def slice(self, start: int, length: int) -> "TokenView":
        """Create a new view into the same data."""
        return TokenView(self.data, self.start + start, length)


class ConcurrentBloomTrie:
    """Concurrent bloom filter trie for fast token lookup."""

    def __init__(self, size: int, num_hashes: int = 3):
        self.size = size
        self.num_hashes = num_hashes
        self.bits = bitarray.bitarray(size)
        self.bits.setall(0)

    def _hash(self, key: str, seed: int) -> int:
        """Hash function with seed."""
        return hash(key + str(seed)) % self.size

    def add(self, key: str):
        """Add key to trie."""
        for i in range(self.num_hashes):
            pos = self._hash(key, i)
            self.bits[pos] = 1

    def contains(self, key: str) -> bool:
        """Check if key is in trie."""
        for i in range(self.num_hashes):
            pos = self._hash(key, i)
            if not self.bits[pos]:
                return False
        return True


class AoSoA:
    """Array of Struct of Arrays for SIMD-friendly token storage."""

    def __init__(self, num_tokens: int, vec_size: int):
        self.num_tokens = num_tokens
        self.vec_size = vec_size
        self.data = np.zeros((num_tokens, vec_size), dtype=np.float32)
        self.masks = np.zeros(num_tokens, dtype=np.bool_)

    def set_token(self, idx: int, vec: np.ndarray, mask: bool = True):
        """Set token vector and mask."""
        self.data[idx] = vec
        self.masks[idx] = mask

    def get_token(self, idx: int) -> tuple[np.ndarray, bool]:
        """Get token vector and mask."""
        return self.data[idx], self.masks[idx]

    def batch_get(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Get multiple tokens efficiently."""
        return self.data[indices], self.masks[indices]


class TokenStream:
    """Lock-free token stream with ring buffer and SIMD processing."""

    def __init__(self, buffer_size: int = 1024, num_workers: int = 4):
        self.buffer = RingBuffer(buffer_size)
        self.workers = ThreadPoolExecutor(max_workers=num_workers)
        self.views: list[TokenView] = []

    def push_tokens(self, tokens: list[int]):
        """Push tokens to stream."""
        for token in tokens:
            while not self.buffer.push(token):
                time.sleep(0.001)  # Spin wait

    def process_stream(self, batch_size: int = 32):
        """Process token stream in batches."""
        batch = []
        while len(batch) < batch_size:
            token = self.buffer.pop()
            if token is None:
                break
            batch.append(token)
        return batch

    def create_view(self, start: int, length: int) -> TokenView:
        """Create view into token stream."""
        view = TokenView(self.buffer.buffer, start, length)
        self.views.append(view)
        return view

    def close(self):
        """Close token stream."""
        self.workers.shutdown()


class ImageModality(nn.Module):
    """
    Complex-valued image modality leveraging JPEG's signal processing nature.
    
    This modality exploits the fact that JPEG compression works by:
    1. Converting RGB to YCbCr color space
    2. Applying 2D DCT to extract frequency components
    3. Quantizing coefficients to achieve compression
    
    We use this same pipeline but preserve the complex-valued nature of the
    frequency domain representation for Nanite's multivector operations.
    """
    
    def __init__(
        self,
        ops: MultivectorOps,
        input_channels: int = 3,
        output_dim: int = 256,
        block_size: int = 8,  # JPEG standard block size
        num_freq_bands: int = 64,  # 8x8 DCT coefficients
        use_quantization: bool = True,
        quantize_bits: int = 4,
        use_cache: bool = True,
        cache_size: int = 1024,
        use_deduplication: bool = True,
        num_workers: int = 4,
        device: str = "cpu",
    ):
        super().__init__()
        
        self.ops = ops
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.block_size = block_size
        self.num_freq_bands = num_freq_bands
        self.use_quantization = use_quantization
        self.quantize_bits = quantize_bits
        self.use_cache = use_cache
        self.cache_size = cache_size
        self.use_deduplication = use_deduplication
        self.num_workers = num_workers
        self.device = device
        
        # Initialize optimized data structures
        self._init_optimized_structures()
        
        # DCT basis matrices for fast computation
        self._init_dct_basis()
        
        # Color space conversion matrices (RGB to YCbCr)
        self._init_color_conversion()
        
        # Complex-valued frequency band projections
        self._init_frequency_projections()
        
        # Quantization and compression components
        self._init_compression_components()
        
        print(f"[ImageModality] Initialized with {self.num_freq_bands} frequency bands, "
              f"block size {self.block_size}, quantization: {self.use_quantization}")

    def _init_optimized_structures(self):
        """Initialize lock-free, async, and optimized data structures."""
        # Tensor cache for deduplication and memory efficiency
        if self.use_cache:
            self.tensor_cache = TensorCache(max_bytes=self.cache_size * 1024 * 1024)  # MB
            self.cache_initialized = False
        else:
            self.tensor_cache = None
            
        # Tensor hasher for deduplication
        if self.use_deduplication:
            self.tensor_hasher = TensorHasher(strategy="auto")
            self.hash_to_tensor = {}  # In-memory hash table
        else:
            self.tensor_hasher = None
            
        # Concurrent bloom filter for fast membership testing
        self.bloom_filter = ConcurrentBloomTrie(size=10000, num_hashes=3)
        
        # Array of Structures of Arrays for efficient memory layout
        self.aosoa = AoSoA(num_tokens=self.num_freq_bands, vec_size=self.output_dim)
        
        # Token stream for async processing
        self.token_stream = TokenStream(buffer_size=512, num_workers=self.num_workers)

    def _init_dct_basis(self):
        """Initialize DCT basis matrices for fast computation."""
        # Create 2D DCT basis matrices
        self.dct_basis_u = torch.zeros(self.block_size, self.block_size, dtype=torch.complex64)
        self.dct_basis_v = torch.zeros(self.block_size, self.block_size, dtype=torch.complex64)
        
        for i in range(self.block_size):
            for j in range(self.block_size):
                # DCT basis functions with complex phase
                phase_u = torch.tensor(2 * torch.pi * i * j / self.block_size)
                phase_v = torch.tensor(2 * torch.pi * i * j / self.block_size)
                
                self.dct_basis_u[i, j] = torch.complex(
                    torch.cos(phase_u), torch.sin(phase_u)
                )
                self.dct_basis_v[i, j] = torch.complex(
                    torch.cos(phase_v), torch.sin(phase_v)
                )
        
        # Normalize basis matrices
        self.dct_basis_u /= torch.sqrt(torch.tensor(self.block_size, dtype=torch.complex64))
        self.dct_basis_v /= torch.sqrt(torch.tensor(self.block_size, dtype=torch.complex64))

    def _init_color_conversion(self):
        """Initialize RGB to YCbCr color space conversion matrices."""
        # Standard RGB to YCbCr conversion matrix
        self.rgb_to_ycbcr = torch.tensor([
            [0.299, 0.587, 0.114],
            [-0.169, -0.331, 0.500],
            [0.500, -0.419, -0.081]
        ], dtype=torch.complex64)
        
        # YCbCr to RGB conversion matrix
        self.ycbcr_to_rgb = torch.inverse(self.rgb_to_ycbcr)

    def _init_frequency_projections(self):
        """Initialize complex-valued frequency band projections."""
        # Create learnable projections for each frequency band
        self.freq_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.block_size * self.block_size, self.output_dim // 4, dtype=torch.complex64),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.output_dim // 4, self.output_dim, dtype=torch.complex64)
            ) for _ in range(self.num_freq_bands)
        ])
        
        # Frequency band importance weights (learnable)
        self.freq_weights = nn.Parameter(
            torch.randn(self.num_freq_bands, dtype=torch.complex64) * 0.02
        )

    def _init_compression_components(self):
        """Initialize quantization and compression components."""
        if self.use_quantization:
            # Quantizer for frequency coefficients
            self.quantizer = Quantizer(
                qmin=-(2**(self.quantize_bits-1)),
                qmax=2**(self.quantize_bits-1)-1,
                init_scale=1.0,
                mode="complex"
            )
            
            # Mixed precision router for adaptive quantization
            self.precision_router = MixedPrecisionRouter(
                small=1024, low_rank=16, large=1_000_000
            )

    async def _start_cache_if_needed(self):
        """Start tensor cache if not already started."""
        if self.tensor_cache and not self.cache_initialized:
            await self.tensor_cache.start()
            self.cache_initialized = True

    def _compute_image_hash(self, x: torch.Tensor) -> int:
        """Compute fast hash of image tensor for deduplication."""
        if self.tensor_hasher:
            return self.tensor_hasher.hash_tensor(x)
        return hash(x.flatten().numpy().tobytes())

    def _check_deduplication(self, image_hash: int, x: torch.Tensor) -> torch.Tensor | None:
        """Check if tensor exists in deduplication cache."""
        if not self.use_deduplication:
            return None
            
        if self.bloom_filter.contains(str(image_hash)):
            if image_hash in self.hash_to_tensor:
                return self.hash_to_tensor[image_hash]
        return None

    def _store_deduplication(self, image_hash: int, result: torch.Tensor):
        """Store result in deduplication cache."""
        if not self.use_deduplication:
            return
            
        self.bloom_filter.add(str(image_hash))
        self.hash_to_tensor[image_hash] = result.detach().clone()

    def _rgb_to_ycbcr(self, x: torch.Tensor) -> torch.Tensor:
        """Convert RGB to YCbCr color space."""
        # x: [B, 3, H, W] -> [B, 3, H, W]
        B, C, H, W = x.shape
        
        # Reshape for matrix multiplication
        x_flat = x.permute(0, 2, 3, 1).reshape(-1, 3)  # [B*H*W, 3]
        x_flat = x_flat.to(torch.complex64)  # Ensure complex dtype
        # Apply color conversion
        ycbcr_flat = torch.matmul(x_flat, self.rgb_to_ycbcr.T)  # [B*H*W, 3]
        
        # Reshape back
        ycbcr = ycbcr_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)  # [B, 3, H, W]
        
        return ycbcr

    def _extract_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """Extract blocks from image for DCT processing."""
        # x: [B, C, H, W] -> [B*num_blocks, C, block_size, block_size]
        B, C, H, W = x.shape
        
        # Pad if necessary to make dimensions divisible by block_size
        pad_h = (self.block_size - H % self.block_size) % self.block_size
        pad_w = (self.block_size - W % self.block_size) % self.block_size
        
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            H, W = H + pad_h, W + pad_w
        
        # Extract blocks
        blocks = x.unfold(2, self.block_size, self.block_size).unfold(3, self.block_size, self.block_size)
        blocks = blocks.contiguous().view(B, C, -1, self.block_size, self.block_size)
        blocks = blocks.permute(0, 2, 1, 3, 4).contiguous().view(-1, C, self.block_size, self.block_size)
        
        return blocks

    def _apply_2d_dct(self, blocks: torch.Tensor) -> torch.Tensor:
        """Apply 2D DCT to blocks to extract frequency components."""
        # blocks: [B*num_blocks, C, block_size, block_size]
        # -> [B*num_blocks, C, block_size, block_size] (complex)
        
        B_num_blocks, C, H, W = blocks.shape
        
        # Convert to complex
        if not blocks.is_complex():
            blocks = torch.complex(blocks, torch.zeros_like(blocks))
        
        # Apply 2D DCT using basis matrices
        # DCT = U^T * X * V
        dct_blocks = torch.zeros_like(blocks)
        
        for i in range(B_num_blocks):
            for j in range(C):
                block = blocks[i, j]  # [block_size, block_size]
                
                # Apply 2D DCT
                dct_block = torch.matmul(
                    torch.matmul(self.dct_basis_u.T, block),
                    self.dct_basis_v
                )
                
                dct_blocks[i, j] = dct_block
        
        return dct_blocks

    def _quantize_frequencies(self, dct_blocks: torch.Tensor) -> torch.Tensor:
        """Quantize DCT coefficients for compression."""
        if not self.use_quantization:
            return dct_blocks
            
        # Get precision level based on tensor characteristics
        precision = self.precision_router.get_precision(dct_blocks)
        
        if precision == PrecisionLevel.INT4:
            # Apply quantization
            dct_blocks = self.quantizer(dct_blocks)
        
        return dct_blocks

    def _extract_frequency_bands(self, dct_blocks: torch.Tensor) -> torch.Tensor:
        """Extract frequency bands from DCT coefficients."""
        # dct_blocks: [B*num_blocks, C, block_size, block_size]
        # -> [B*num_blocks, num_freq_bands, output_dim]
        
        B_num_blocks, C, H, W = dct_blocks.shape
        
        # Flatten DCT coefficients
        dct_flat = dct_blocks.reshape(B_num_blocks, C, -1)  # [B*num_blocks, C, block_size^2]
        
        # Extract frequency bands (zigzag pattern for JPEG-like ordering)
        freq_bands = []
        for i in range(self.num_freq_bands):
            if i < dct_flat.shape[-1]:
                # Get frequency band coefficients
                band_coeffs = dct_flat[:, :, i]  # [B*num_blocks, C]
                
                # Apply frequency-specific projection
                band_projected = self.freq_projections[i](band_coeffs)  # [B*num_blocks, output_dim]
                
                # Apply frequency weight
                band_weighted = band_projected * self.freq_weights[i]
                
                freq_bands.append(band_weighted)
            else:
                # Pad with zeros if more bands than coefficients
                freq_bands.append(torch.zeros(B_num_blocks, self.output_dim, dtype=torch.complex64))
        
        # Stack frequency bands
        result = torch.stack(freq_bands, dim=1)  # [B*num_blocks, num_freq_bands, output_dim]
        
        return result

    def _aggregate_blocks(self, block_features: torch.Tensor, original_shape: tuple) -> torch.Tensor:
        """Aggregate block features back to image-level representation."""
        # block_features: [B*num_blocks, num_freq_bands, output_dim]
        # -> [B, output_dim]
        
        B, C, H, W = original_shape
        
        # Calculate number of blocks
        num_blocks_h = (H + self.block_size - 1) // self.block_size
        num_blocks_w = (W + self.block_size - 1) // self.block_size
        num_blocks = num_blocks_h * num_blocks_w
        
        # Reshape to separate batch and blocks
        block_features = block_features.reshape(B, num_blocks, -1, self.output_dim)
        
        # Aggregate across blocks using weighted average
        # Higher frequency bands get lower weights (JPEG-like)
        freq_weights = torch.softmax(torch.arange(self.num_freq_bands, dtype=torch.float32), dim=0)
        freq_weights = freq_weights.to(block_features.device)
        
        # Weighted aggregation
        weighted_features = block_features * freq_weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        aggregated = weighted_features.sum(dim=(1, 2))  # [B, output_dim]
        
        return aggregated

    async def forward(
        self,
        x: torch.Tensor,
        input_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        Process image through complex-valued JPEG-inspired pipeline.
        
        Args:
            x: Input tensor [B, C, H, W] (real-valued)
            input_size: Optional input size for resizing
            
        Returns:
            Complex multivector representation [B, output_dim, mv_dim]
        """
        # Start cache if needed
        await self._start_cache_if_needed()
        
        # Compute hash for deduplication
        image_hash = self._compute_image_hash(x)
        
        # Check deduplication cache
        cached_result = self._check_deduplication(image_hash, x)
        if cached_result is not None:
            return cached_result
        
        # Store original shape
        original_shape = x.shape
        
        # 1. Color space conversion (RGB to YCbCr)
        ycbcr = self._rgb_to_ycbcr(x)
        
        # 2. Extract blocks for DCT processing
        blocks = self._extract_blocks(ycbcr)
        
        # 3. Apply 2D DCT to extract frequency components
        dct_blocks = self._apply_2d_dct(blocks)
        
        # 4. Quantize frequency coefficients
        dct_blocks = self._quantize_frequencies(dct_blocks)
        
        # 5. Extract frequency bands
        freq_features = self._extract_frequency_bands(dct_blocks)
        
        # 6. Aggregate blocks back to image-level
        image_features = self._aggregate_blocks(freq_features, original_shape)
        
        # 7. Convert to multivector representation (identity, already complex)
        mv = image_features if torch.is_complex(image_features) else torch.complex(image_features, torch.zeros_like(image_features))
        
        # Store in deduplication cache
        self._store_deduplication(image_hash, mv)
        
        # Store in tensor cache if enabled
        if self.tensor_cache:
            await self.tensor_cache.put(f"image_{image_hash}", mv, "cpu")
        
        return mv

    def forward_sync(self, x: torch.Tensor) -> torch.Tensor:
        """Synchronous version for ONNX export and testing."""
        # For sync version, we'll use a simplified pipeline
        # that doesn't require async operations
        
        # Store original shape
        original_shape = x.shape
        
        # 1. Color space conversion
        ycbcr = self._rgb_to_ycbcr(x)
        
        # 2. Extract blocks
        blocks = self._extract_blocks(ycbcr)
        
        # 3. Apply 2D DCT (simplified for sync)
        if not blocks.is_complex():
            blocks = torch.complex(blocks, torch.zeros_like(blocks))
        
        # Use FFT as approximation of DCT for speed
        dct_blocks = torch.fft.fft2(blocks, dim=(-2, -1))
        
        # 4. Extract frequency bands (simplified)
        B_num_blocks, C, H, W = dct_blocks.shape
        dct_flat = dct_blocks.reshape(B_num_blocks, C, -1)
        
        # Take first num_freq_bands coefficients
        freq_coeffs = dct_flat[:, :, :self.num_freq_bands]  # [B*num_blocks, C, num_freq_bands]
        
        # Simple projection to output dimension
        freq_features = freq_coeffs.mean(dim=1)  # [B*num_blocks, num_freq_bands]
        
        # Project to output dimension
        projection = torch.randn(self.num_freq_bands, self.output_dim, dtype=torch.complex64)
        block_features = torch.matmul(freq_features, projection)  # [B*num_blocks, output_dim]
        
        # 5. Aggregate blocks
        B, C, H, W = original_shape
        num_blocks_h = (H + self.block_size - 1) // self.block_size
        num_blocks_w = (W + self.block_size - 1) // self.block_size
        num_blocks = num_blocks_h * num_blocks_w
        
        block_features = block_features.reshape(B, num_blocks, self.output_dim)
        image_features = block_features.mean(dim=1)  # [B, output_dim]
        
        # 6. Convert to multivector (identity, already complex)
        mv = image_features if torch.is_complex(image_features) else torch.complex(image_features, torch.zeros_like(image_features))
        
        return mv

    async def cleanup(self):
        """Cleanup resources."""
        if self.tensor_cache and self.cache_initialized:
            await self.tensor_cache.stop()
            self.cache_initialized = False
        
        if self.token_stream:
            self.token_stream.close()
        
        # Clear deduplication cache
        if self.use_deduplication:
            self.hash_to_tensor.clear()
            self.bloom_filter = ConcurrentBloomTrie(size=10000, num_hashes=3)

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            if hasattr(self, 'token_stream') and self.token_stream:
                self.token_stream.close()
        except:
            pass


class SubwordTokenizer(nn.Module):
    """Subword tokenizer with parallel processing and complex embeddings."""

    def __init__(
        self,
        w2v_path: str | None = None,
        vocab_size: int = 32000,
        min_frequency: int = 2,
        max_subword_length: int = 4,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        device: str = "cpu",  # Force CPU for better async performance
        ops: MultivectorOps | None = None,
        embedding_dim: int = 256,
        num_basis: int = 32,
        quantize: bool = True,
        quantize_bits: int = 4,
        use_mmap: bool = True,
        num_workers: int = 1,
        buffer_size: int = 512,
        w2v_quantize_bits: int = 8,  # FP8 quantization for Word2Vec
        w2v_cache_size: int = 10000,  # Number of vectors to cache in memory
        w2v_mmap_threshold: int = 50000,  # Use mmap if vocab > this size
    ):
        super().__init__()
        print("[SubwordTokenizer] Initializing...")

        self.device = device
        # Use shared ops instance
        self.ops = ops or get_shared_ops()
            
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.max_subword_length = max_subword_length
        self.embedding_dim = embedding_dim
        self.num_basis = num_basis
        self.quantize = quantize
        self.quantize_bits = quantize_bits
        self.use_mmap = use_mmap
        self.num_workers = num_workers
        self.buffer_size = buffer_size
        
        # Word2Vec specific parameters
        self.w2v_path = w2v_path
        self.w2v_quantize_bits = w2v_quantize_bits
        self.w2v_cache_size = w2v_cache_size
        self.w2v_mmap_threshold = w2v_mmap_threshold
        
        # Word2Vec storage
        self._w2v_model = None
        self._w2v_vectors = None
        self._w2v_quantized = None
        self._w2v_mmap_file = None
        self._w2v_mmap_data = None
        self._w2v_cache = {}
        self._w2v_vocab_mapping = {}
        self._w2v_loaded = False
        self._w2v_quantized_loaded = False

        # Special tokens
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        # Initialize components
        print("[SubwordTokenizer] Setting up basic components...")
        self.vocab = {
            unk_token: 0,
            pad_token: 1,
            bos_token: 2,
            eos_token: 3
        }
        
        # Add common single characters
        for char in "abcdefghijklmnopqrstuvwxyz":
            self.vocab[char] = len(self.vocab)
        
        # Add digits
        for digit in "0123456789":
            self.vocab[digit] = len(self.vocab)
            
        # Add common punctuation and symbols
        common_symbols = (
            ".,!?;:'\"()[]{}<>-_+=*/\\@#$%^&"  # Basic punctuation
            "°²³±×÷"  # Mathematical symbols
            "€£¥¢"  # Currency symbols
            "©®™"  # Legal symbols
            "→←↑↓↔↕"  # Arrows
            "•●○◆◇■□"  # Shapes
            "✓✔✗✘"  # Checkmarks
            "★☆"  # Stars
            "☀☁☂☃☎☏"  # Weather and phone
            "☺☻"  # Smileys
            "♠♣♥♦"  # Cards
            "♫♪"  # Music
            "∞≠≈≤≥"  # Math
            "∑∏√∫"  # More math
            "αβγδε"  # Greek letters
            "…"  # Ellipsis
            "—–"  # Dashes
            "«»‹›"  # Quotes
            "§¶"  # Section
            "†‡"  # Dagger
            "µ"  # Micro
            "¶"  # Paragraph
            "·"  # Middle dot
            "¿¡"  # Spanish punctuation
            "«»"  # French quotes
            "„"  # German quotes
            "‟"  # More quotes
            "′″‴"  # Prime
            "‱"  # Per ten thousand
            "‰"  # Per mille
            "‽"  # Interrobang
            "⁰¹²³⁴⁵⁶⁷⁸⁹"  # Superscript
            "₀₁₂₃₄₅₆₇₈₉"  # Subscript
            "⁺⁻⁼"  # Superscript operators
            "₊₋₌"  # Subscript operators
        )
        for symbol in common_symbols:
            self.vocab[symbol] = len(self.vocab)
            
        # Add common contractions and special words
        common_words = {
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
            # Add common units and measurements
            "km", "m", "cm", "mm", "kg", "g", "mg", "l", "ml", "km²", "m²", "cm²",
            "km³", "m³", "cm³", "°C", "°F", "K", "Hz", "kHz", "MHz", "GHz",
            "W", "kW", "MW", "GW", "V", "kV", "A", "mA", "Ω", "kΩ", "MΩ",
            "s", "ms", "μs", "ns", "min", "h", "day", "week", "month", "year",
            "byte", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB",
            "bit", "kbit", "Mbit", "Gbit", "Tbit", "Pbit", "Ebit", "Zbit", "Ybit",
            "bps", "kbps", "Mbps", "Gbps", "Tbps", "Pbps", "Ebps", "Zbps", "Ybps"
        }
        for word in common_words:
            self.vocab[word] = len(self.vocab)
            
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.subword_trie = ConcurrentBloomTrie(vocab_size)
        self.token_stream = TokenStream(buffer_size, num_workers)
        self.basis_set = ComplexBasisSet(num_basis, embedding_dim)

        # Initialize embeddings
        print("[SubwordTokenizer] Initializing embeddings...")
        self.embeddings = nn.Parameter(
            torch.randn(vocab_size, embedding_dim, dtype=torch.complex64) * 0.02
        )
        
        print("[SubwordTokenizer] Setting up parallel processing...")
        self._init_parallel_processing()

        # Load Word2Vec if path provided
        if self.w2v_path:
            print(f"[SubwordTokenizer] Word2Vec path provided: {self.w2v_path}")
            self._load_w2v_model()

        print("[SubwordTokenizer] Initialization complete!")

    def _load_w2v_model(self):
        """Load Word2Vec model with memory-efficient handling."""
        if not self.w2v_path or self._w2v_loaded:
            return
            
        try:
            print(f"[SubwordTokenizer] Loading Word2Vec model from {self.w2v_path}...")
            
            # Check if file exists
            if not Path(self.w2v_path).exists():
                print(f"[SubwordTokenizer] Warning: Word2Vec file not found at {self.w2v_path}")
                return
            
            # Load Word2Vec model
            if self.w2v_path.endswith('.bin'):
                # Binary format
                self._w2v_model = KeyedVectors.load_word2vec_format(
                    self.w2v_path, 
                    binary=True,
                    limit=self.vocab_size  # Limit to vocab size for memory efficiency
                )
            else:
                # Text format or other formats
                self._w2v_model = KeyedVectors.load(self.w2v_path)
            
            print(f"[SubwordTokenizer] Loaded Word2Vec model with {len(self._w2v_model.key_to_index)} vectors")
            
            # Quantize and cache Word2Vec vectors
            self._quantize_w2v_vectors()
            
            # Build vocabulary mapping
            self._build_w2v_vocab_mapping()
            
            self._w2v_loaded = True
            print("[SubwordTokenizer] Word2Vec model loaded and quantized successfully!")
            
        except Exception as e:
            print(f"[SubwordTokenizer] Error loading Word2Vec model: {e}")
            print("[SubwordTokenizer] Falling back to random embeddings...")

    def _quantize_w2v_vectors(self):
        """Quantize Word2Vec vectors to FP8 for memory efficiency."""
        if not self._w2v_model or self._w2v_quantized_loaded:
            return
            
        try:
            print(f"[SubwordTokenizer] Quantizing Word2Vec vectors to {self.w2v_quantize_bits}-bit precision...")
            
            # Get all vectors
            vectors = self._w2v_model.vectors  # [vocab_size, embedding_dim]
            
            # Quantize to FP8 or specified bit precision
            if self.w2v_quantize_bits == 8:
                # FP8 quantization
                max_val = np.max(np.abs(vectors))
                scale = 127.0 / max_val if max_val > 0 else 1.0
                quantized = np.round(vectors * scale).astype(np.int8)
                self._w2v_quantized = {
                    'vectors': quantized,
                    'scale': scale,
                    'dtype': 'int8'
                }
            else:
                # Custom bit quantization
                max_val = np.max(np.abs(vectors))
                scale = (2**(self.w2v_quantize_bits - 1) - 1) / max_val if max_val > 0 else 1.0
                quantized = np.round(vectors * scale).astype(np.int16)
                self._w2v_quantized = {
                    'vectors': quantized,
                    'scale': scale,
                    'dtype': f'int{self.w2v_quantize_bits}'
                }
            
            # Setup memory mapping if vocab is large
            if len(self._w2v_model.key_to_index) > self.w2v_mmap_threshold:
                self._setup_w2v_mmap()
            
            self._w2v_quantized_loaded = True
            print(f"[SubwordTokenizer] Quantized {len(vectors)} vectors to {self.w2v_quantize_bits}-bit precision")
            
        except Exception as e:
            print(f"[SubwordTokenizer] Error quantizing Word2Vec vectors: {e}")

    def _setup_w2v_mmap(self):
        """Setup memory mapping for large Word2Vec vocabularies."""
        try:
            print("[SubwordTokenizer] Setting up memory mapping for large vocabulary...")
            
            # Create temporary file for mmap
            mmap_path = Path(self.w2v_path).parent / f"w2v_mmap_{hashlib.md5(self.w2v_path.encode()).hexdigest()[:8]}.bin"
            
            # Save quantized vectors to file
            with open(mmap_path, 'wb') as f:
                pickle.dump(self._w2v_quantized, f)
            
            # Memory map the file
            self._w2v_mmap_file = open(mmap_path, 'rb')
            self._w2v_mmap_data = mmap.mmap(
                self._w2v_mmap_file.fileno(), 
                0, 
                access=mmap.ACCESS_READ
            )
            
            # Load the mapping data
            self._w2v_quantized = pickle.loads(self._w2v_mmap_data)
            
            print(f"[SubwordTokenizer] Memory mapping setup for {mmap_path}")
            
        except Exception as e:
            print(f"[SubwordTokenizer] Error setting up memory mapping: {e}")

    def _build_w2v_vocab_mapping(self):
        """Build mapping between our vocab and Word2Vec vocab."""
        if not self._w2v_model:
            return
            
        try:
            print("[SubwordTokenizer] Building Word2Vec vocabulary mapping...")
            
            w2v_vocab = set(self._w2v_model.key_to_index.keys())
            our_vocab = set(self.vocab.keys())
            
            # Find intersection
            common_words = w2v_vocab.intersection(our_vocab)
            
            # Build mapping
            for word in common_words:
                if word in self.vocab and word in self._w2v_model.key_to_index:
                    our_idx = self.vocab[word]
                    w2v_idx = self._w2v_model.key_to_index[word]
                    self._w2v_vocab_mapping[our_idx] = w2v_idx
            
            print(f"[SubwordTokenizer] Mapped {len(self._w2v_vocab_mapping)} common words")
            
        except Exception as e:
            print(f"[SubwordTokenizer] Error building vocabulary mapping: {e}")

    def _get_w2v_vector(self, word: str) -> torch.Tensor | None:
        """Get Word2Vec vector for a word with caching and lazy loading."""
        if not self._w2v_model or word not in self._w2v_model.key_to_index:
            return None
        
        try:
            # Check cache first
            if word in self._w2v_cache:
                return self._w2v_cache[word]
            
            # Get vector index
            w2v_idx = self._w2v_model.key_to_index[word]
            
            # Get quantized vector
            if self._w2v_quantized:
                quantized_vec = self._w2v_quantized['vectors'][w2v_idx]
                scale = self._w2v_quantized['scale']
                
                # Dequantize
                vec = quantized_vec.astype(np.float32) / scale
            else:
                # Fallback to original vectors
                vec = self._w2v_model.vectors[w2v_idx]
            
            # Convert to complex tensor
            vec_tensor = torch.tensor(vec, dtype=torch.complex64, device=self.device)
            
            # Cache if we have space
            if len(self._w2v_cache) < self.w2v_cache_size:
                self._w2v_cache[word] = vec_tensor
            
            return vec_tensor
            
        except Exception as e:
            print(f"[SubwordTokenizer] Error getting Word2Vec vector for '{word}': {e}")
            return None

    def _init_embeddings(self):
        """Initialize or update embeddings with Word2Vec."""
        print("[SubwordTokenizer] Initializing embeddings with Word2Vec...")
        try:
            if self._w2v_model is not None and self._w2v_quantized_loaded:
                print("[SubwordTokenizer] Using quantized Word2Vec model for embeddings...")
                
                # Initialize embeddings tensor
                new_embeddings = torch.zeros(
                    self.vocab_size,
                    self.embedding_dim,
                    dtype=torch.complex64,
                    device=self.device
                )
                
                # Initialize special tokens with random values
                for token in [self.unk_token, self.pad_token, self.bos_token, self.eos_token]:
                    if token in self.vocab:
                        idx = self.vocab[token]
                        new_embeddings[idx] = torch.randn(
                            self.embedding_dim,
                            dtype=torch.complex64,
                            device=self.device
                        ) * 0.02
                
                # Initialize other tokens with Word2Vec vectors
                initialized_count = 0
                for token, idx in self.vocab.items():
                    if token not in [self.unk_token, self.pad_token, self.bos_token, self.eos_token]:
                        w2v_vec = self._get_w2v_vector(token)
                        if w2v_vec is not None:
                            # Ensure correct dimension
                            if w2v_vec.shape[0] != self.embedding_dim:
                                # Pad or truncate to match embedding_dim
                                if w2v_vec.shape[0] < self.embedding_dim:
                                    padding = torch.zeros(self.embedding_dim - w2v_vec.shape[0], dtype=torch.complex64, device=self.device)
                                    w2v_vec = torch.cat([w2v_vec, padding])
                                else:
                                    w2v_vec = w2v_vec[:self.embedding_dim]
                            
                            new_embeddings[idx] = w2v_vec
                            initialized_count += 1
                        else:
                            # Fallback to random initialization
                            new_embeddings[idx] = torch.randn(
                                self.embedding_dim,
                                dtype=torch.complex64,
                                device=self.device
                            ) * 0.02
                
                # Update embeddings
                self.embeddings.data = new_embeddings
                print(f"[SubwordTokenizer] Initialized {initialized_count} embeddings from Word2Vec model")
                
            else:
                print("[SubwordTokenizer] Using random initialization for embeddings")
                # Random initialization is already done in __init__
                
        except Exception as e:
            print(f"[SubwordTokenizer] Error initializing embeddings: {e}")
            raise

    def update_embeddings_from_w2v(self, learning_rate: float = 0.001):
        """Update embeddings based on Word2Vec reference model with quantization."""
        if not self._w2v_model or not self._w2v_quantized_loaded:
            return

        try:
            print("[SubwordTokenizer] Updating embeddings from Word2Vec...")
            
            # Get reference vectors from Word2Vec
            w2v_vectors = []
            valid_indices = []
            
            for token, idx in self.vocab.items():
                if token not in [self.unk_token, self.pad_token, self.bos_token, self.eos_token]:
                    w2v_vec = self._get_w2v_vector(token)
                    if w2v_vec is not None:
                        w2v_vectors.append(w2v_vec)
                        valid_indices.append(idx)
            
            if not w2v_vectors:
                print("[SubwordTokenizer] No valid Word2Vec vectors found for update")
                return
            
            # Stack vectors
            w2v_tensor = torch.stack(w2v_vectors)  # [num_valid, embedding_dim]
            current_embeddings = self.embeddings[valid_indices]  # [num_valid, embedding_dim]
            
            # Compute cosine similarity loss
            norm_embeddings = torch.nn.functional.normalize(current_embeddings, p=2, dim=-1)
            norm_w2v = torch.nn.functional.normalize(w2v_tensor, p=2, dim=-1)
            
            # Compute cosine similarity matrix
            similarity = torch.matmul(norm_embeddings, norm_w2v.t())
            
            # Convert to distance
            distance = 1 - similarity
            
            # Compute gradients
            grad = torch.matmul(distance, norm_w2v)
            
            # Update embeddings
            with torch.no_grad():
                self.embeddings.data[valid_indices] -= learning_rate * grad
                
            print(f"[SubwordTokenizer] Updated {len(valid_indices)} embeddings from Word2Vec")
            
        except Exception as e:
            print(f"[SubwordTokenizer] Error updating embeddings from Word2Vec: {e}")

    def get_w2v_stats(self) -> dict:
        """Get statistics about Word2Vec usage."""
        stats = {
            'w2v_loaded': self._w2v_loaded,
            'w2v_quantized': self._w2v_quantized_loaded,
            'w2v_vocab_size': len(self._w2v_model.key_to_index) if self._w2v_model else 0,
            'w2v_cache_size': len(self._w2v_cache),
            'w2v_mapped_words': len(self._w2v_vocab_mapping),
            'quantization_bits': self.w2v_quantize_bits,
            'memory_mapped': self._w2v_mmap_data is not None
        }
        return stats

    def print_w2v_stats(self):
        """Print Word2Vec statistics."""
        stats = self.get_w2v_stats()
        print("\n" + "=" * 50)
        print("📚 Word2Vec Statistics")
        print("=" * 50)
        print(f"🔗 Loaded: {stats['w2v_loaded']}")
        print(f"⚡ Quantized: {stats['w2v_quantized']}")
        print(f"📖 Vocabulary Size: {stats['w2v_vocab_size']:,}")
        print(f"💾 Cache Size: {stats['w2v_cache_size']:,}")
        print(f"🔗 Mapped Words: {stats['w2v_mapped_words']:,}")
        print(f"🎯 Quantization Bits: {stats['quantization_bits']}")
        print(f"🗺️ Memory Mapped: {stats['memory_mapped']}")
        print("=" * 50 + "\n")

    def _init_parallel_processing(self):
        """Initialize parallel processing components."""
        print("[SubwordTokenizer] Creating process pool...")
        try:
            # Use ThreadPoolExecutor instead of ProcessPoolExecutor to avoid pickling issues
            self.process_pool = ThreadPoolExecutor(max_workers=self.num_workers)
            self.thread_pool = ThreadPoolExecutor(max_workers=self.num_workers)
            self.lock = Lock()
            print("[SubwordTokenizer] Process pool created successfully")
        except Exception as e:
            print(f"[SubwordTokenizer] Error creating process pool: {e}")
            raise

    def _build_vocab(self, texts: list[str]):
        """Build vocabulary from texts."""
        print("[SubwordTokenizer] Building vocabulary...")
        word_freq = {}
        
        # Count word frequencies
        print("[SubwordTokenizer] Counting word frequencies...")
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Add special tokens
        self.vocab = {
            self.unk_token: 0,
            self.pad_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }
        
        # Add words that appear frequently enough
        for word, freq in word_freq.items():
            if freq >= self.min_frequency:
                self.vocab[word] = len(self.vocab)
                
        print(f"[SubwordTokenizer] Final vocab: {list(self.vocab.keys())[:50]} ... total {len(self.vocab)} tokens")

    def _tokenize_word(self, word: str) -> list[str]:
        """Tokenize a single word into subwords."""
        # Handle single characters and digits
        if len(word) == 1:
            if word in self.vocab:
                return [word]
            return [self.unk_token]
            
        # Try to find the longest matching subword
        subwords = []
        start = 0
        while start < len(word):
            found = False
            for end in range(min(start + self.max_subword_length, len(word)), start, -1):
                subword = word[start:end]
                if subword in self.vocab:
                    subwords.append(subword)
                    start = end
                    found = True
                    break
            if not found:
                subwords.append(self.unk_token)
                start += 1
                
        return subwords

    def _tokenize_text(self, text: str) -> list[str]:
        """Tokenize text into subwords."""
        print(f"[SubwordTokenizer] Tokenizing text: {text[:50]}...")
        tokens = []
        words = text.split()
        for word in words:
            subwords = self._tokenize_word(word)
            tokens.extend(subwords)
        print(f"[SubwordTokenizer] Tokens: {tokens[:50]} ... total {len(tokens)}")
        return tokens

    def tokenize(self, text: str) -> tuple[list[str], list[TokenFunction]]:
        """Tokenize text and return tokens with their functions."""
        print("[SubwordTokenizer] Starting tokenization...")
        tokens = self._tokenize_text(text)
        functions = []
        
        print("[SubwordTokenizer] Creating token functions...")
        for token in tokens:
            if token in self.vocab:
                vec = self.embeddings[self.vocab[token]].detach()
                functions.append(self._create_token_function(token, vec))
            else:
                vec = self.embeddings[0].detach()  # Use UNK token embedding
                functions.append(self._create_token_function(self.unk_token, vec))

        print("[SubwordTokenizer] Tokenization complete!")
        return tokens, functions

    async def encode(
        self,
        text: str,
        max_length: int | None = None,
        padding: bool = False,
        return_tensors: str | None = None,
    ) -> tuple[torch.Tensor, list[TokenFunction]]:
        """Encode text to token IDs and functions.

        Args:
            text: Input text to encode
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            return_tensors: Return format ('pt', 'np', etc.)

        Returns:
            Tuple of (token_ids, token_functions)
        """
        print("[SubwordTokenizer] Starting encode...")
        
        # Tokenize text
        print("[SubwordTokenizer] Starting tokenization...")
        tokens, functions = self.tokenize(text)
        print(f"[SubwordTokenizer] Tokenized into {len(tokens)} tokens")
        
        # Convert tokens to IDs
        print("[SubwordTokenizer] Converting tokens to IDs...")
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab[self.unk_token])
        
        # Add special tokens
        print("[SubwordTokenizer] Adding special tokens...")
        token_ids = [self.vocab[self.bos_token]] + token_ids + [self.vocab[self.eos_token]]
        
        # Truncate if needed
        if max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
        # Pad if needed
        if padding and max_length:
            while len(token_ids) < max_length:
                token_ids.append(self.vocab[self.pad_token])
        
        # Convert to tensor
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        
        print("[SubwordTokenizer] Encode complete!")
        return token_tensor, functions

    async def batch_encode(
        self,
        texts: list[str],
        max_length: int | None = None,
        padding: bool = False,
        return_tensors: str | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode a batch of texts."""
        print(f"[SubwordTokenizer] Starting batch_encode with {len(texts)} texts...")
        try:
            # Process texts in parallel using thread pool
            print("[SubwordTokenizer] Processing texts in parallel...")
            futures = []
            for text in texts:
                future = self.thread_pool.submit(self.tokenize, text)
                futures.append(future)
            
            # Collect results
            print("[SubwordTokenizer] Collecting results...")
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    results.append(result)
                except Exception as e:
                    print(f"[SubwordTokenizer] Error processing text: {e}")
                    # Create fallback result with detach() to avoid gradient issues
                    unk_embedding = self.embeddings[0].detach()
                    results.append((torch.tensor([self.vocab[self.unk_token]], dtype=torch.long), [self._create_token_function(self.unk_token, unk_embedding)]))
            
            # Convert to IDs and handle padding
            print("[SubwordTokenizer] Converting to IDs and handling padding...")
            ids_list = []
            functions_list = []
            max_len = max(len(tokens) for tokens, _ in results)
            
            for tokens, functions in results:
                # Convert to IDs
                ids = [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]
                
                # Add special tokens
                ids = [self.vocab[self.bos_token]] + ids + [self.vocab[self.eos_token]]
                functions = [self._create_token_function(self.bos_token, self.embeddings[self.vocab[self.bos_token]])] + \
                           functions + \
                           [self._create_token_function(self.eos_token, self.embeddings[self.vocab[self.eos_token]])]
                
                # Handle padding
                if padding:
                    pad_length = max_len - len(ids)
                    if pad_length > 0:
                        ids.extend([self.vocab[self.pad_token]] * pad_length)
                        pad_function = self._create_token_function(self.pad_token, self.embeddings[self.vocab[self.pad_token]])
                        functions.extend([pad_function] * pad_length)
                
                ids_list.append(torch.tensor(ids, dtype=torch.long))
                functions_list.append(functions)
            
            # Convert to tensors if requested
            if return_tensors == "pt":
                ids_list = [torch.tensor(ids, device=self.device) for ids in ids_list]
            
            return {
                "input_ids": ids_list,
                "token_functions": functions_list
            }
            
        except Exception as e:
            print(f"[SubwordTokenizer] Error in batch_encode: {e}")
            raise

    def _create_token_function(self, token: str, vec: torch.Tensor) -> TokenFunction:
        """Create complex-valued periodic function for a token."""
        func = TokenFunction(self.num_basis)

        # Detach tensor and convert to numpy
        vec_np = vec.detach().cpu().numpy()

        # Use DCT to get frequency components
        dct_coeffs = dct(vec_np)

        # Set frequencies and phases
        func.frequencies = np.linspace(0.1, self.num_basis, self.num_basis)
        func.phases = np.angle(dct_coeffs[: self.num_basis])
        func.amplitudes = np.abs(dct_coeffs[: self.num_basis])

        # Create complex embedding
        x = np.linspace(0, 1, self.num_basis)
        func.embedding = _compute_basis_functions(
            x,
            func.frequencies,
            func.phases,
            func.amplitudes,
        )

        return func

    def embed(
        self,
        input_ids: torch.Tensor,
        token_functions: list[TokenFunction] | None = None,
    ) -> torch.Tensor:
        print(f"[SubwordTokenizer] embed: input_ids={input_ids}")
        if token_functions:
            funcs = [func.to_tensor() for func in token_functions]
            print(f"[SubwordTokenizer] embed: token_functions, first func tensor: {funcs[0] if funcs else 'None'}")
            return torch.stack(funcs)
        else:
            emb = self.embeddings[input_ids]
            print(f"[SubwordTokenizer] embed: embeddings for input_ids, mean={emb.abs().mean().item()}, std={emb.abs().std().item()}, min={emb.abs().min().item()}, max={emb.abs().max().item()}")
            return emb

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

    def compute_embedding_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute loss between current embeddings and reference Word2Vec embeddings."""
        if self._w2v_vectors is None:
            return torch.tensor(0.0, device=self.device)

        # Compute cosine similarity loss
        norm_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        norm_w2v = torch.nn.functional.normalize(self._w2v_vectors, p=2, dim=-1)
        
        # Compute cosine similarity matrix
        similarity = torch.matmul(norm_embeddings, norm_w2v.t())
        
        # Convert to distance
        distance = 1 - similarity
        
        # Return mean distance
        return distance.mean()

    def update_embeddings(self, learning_rate: float = 0.001):
        """Update embeddings based on Word2Vec reference model."""
        if self._w2v_vectors is None:
            return

        # Compute cosine similarity loss
        norm_embeddings = torch.nn.functional.normalize(self.embeddings, p=2, dim=-1)
        norm_w2v = torch.nn.functional.normalize(self._w2v_vectors, p=2, dim=-1)
        
        # Compute cosine similarity matrix
        similarity = torch.matmul(norm_embeddings, norm_w2v.t())
        
        # Convert to distance
        distance = 1 - similarity
        
        # Compute gradients
        grad = torch.matmul(distance, norm_w2v)
        
        # Update embeddings
        with torch.no_grad():
            self.embeddings.data -= learning_rate * grad
            
        # Update output embeddings (shared weights)
        self.output_embeddings = self.embeddings

    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, "process_pool"):
                self.process_pool.shutdown(wait=False)
            if hasattr(self, "thread_pool"):
                self.thread_pool.shutdown(wait=False)
            if hasattr(self, "_w2v_vectors"):
                del self._w2v_vectors
            gc.collect()
        except Exception as e:
            print(f"[SubwordTokenizer] Error during cleanup: {e}")


class AudioModality(nn.Module):
    """
    Optimized audio modality leveraging STFT and frequency domain processing.
    
    This modality uses:
    - Efficient STFT with proper input handling
    - Frequency domain feature extraction
    - Temporal convolution networks
    - Multivector representation for complex-valued audio features
    - Memory-efficient processing with quantization
    """
    
    def __init__(
        self,
        ops: MultivectorOps,
        input_channels: int = 1,  # Mono by default
        output_dim: int = 256,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        dropout: float = 0.3,
        activation: str = "gelu",
        use_batchnorm: bool = True,
        use_quantization: bool = True,
        quantize_bits: int = 4,
        use_cache: bool = True,
        cache_size: int = 1024,
        num_workers: int = 4,
        device: str = "cpu",
    ):
        super().__init__()
        self.ops = ops
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.use_quantization = use_quantization
        self.quantize_bits = quantize_bits
        self.device = device
        
        # Initialize optimized structures
        self._init_optimized_structures(cache_size, num_workers)
        
        # STFT frequency processing
        self.freq_processor = nn.Sequential(
            nn.Linear(n_fft // 2 + 1, n_mels),
            nn.LayerNorm(n_mels),
            getattr(nn, activation.title())() if hasattr(nn, activation.title()) else nn.GELU(),
            nn.Dropout(dropout),
        )

        # Temporal processing with 1D convolutions
        self.temporal = nn.Sequential(
            nn.Conv1d(n_mels, n_mels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_mels * 2) if use_batchnorm else nn.Identity(),
            getattr(nn, activation.title())() if hasattr(nn, activation.title()) else nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(n_mels * 2, n_mels * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_mels * 4) if use_batchnorm else nn.Identity(),
            getattr(nn, activation.title())() if hasattr(nn, activation.title()) else nn.GELU(),
            nn.Dropout(dropout),
        )

        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(n_mels * 4, output_dim),
            nn.LayerNorm(output_dim),
            getattr(nn, activation.title())() if hasattr(nn, activation.title()) else nn.GELU(),
            nn.Dropout(dropout),
        )
        
        print("[AudioModality] Initialized successfully with optimized processing")

    def _init_optimized_structures(self, cache_size: int, num_workers: int):
        """Initialize optimized data structures for memory efficiency."""
        # Audio hash cache for deduplication
        self.audio_cache = {}
        self.cache_size = cache_size
        
        # Quantizer for memory efficiency
        if self.use_quantization:
            self.quantizer = Quantizer(
                qmin=PRECISION_CONFIGS[PrecisionLevel.INT4].qmin,
                qmax=PRECISION_CONFIGS[PrecisionLevel.INT4].qmax,
                init_scale=1.0,
                mode="complex"
            )
        
        # Tensor hasher for deduplication
        self.hasher = TensorHasher(strategy="auto")
        
        # Mixed precision router
        self.precision_router = MixedPrecisionRouter(
            small=1024, low_rank=16, large=1_000_000
        )

    def _compute_audio_hash(self, x: torch.Tensor) -> int:
        """Compute hash for audio tensor for deduplication."""
        return self.hasher.hash_tensor(x)

    def _check_deduplication(self, audio_hash: int, x: torch.Tensor) -> torch.Tensor | None:
        """Check if audio has been processed before."""
        if audio_hash in self.audio_cache:
            return self.audio_cache[audio_hash]
        return None

    def _store_deduplication(self, audio_hash: int, result: torch.Tensor):
        """Store processed result for deduplication."""
        if len(self.audio_cache) >= self.cache_size:
            # Remove oldest entry (simple LRU)
            oldest_key = next(iter(self.audio_cache))
            del self.audio_cache[oldest_key]
        self.audio_cache[audio_hash] = result

    def _process_audio_stft(self, x: torch.Tensor) -> torch.Tensor:
        """Process audio through STFT with proper input handling."""
        B, C, T = x.shape
        
        # Handle different input shapes
        if C == 1:
            # Mono audio - process directly
            x_mono = x.squeeze(1)  # [B, T]
        else:
            # Multi-channel audio - convert to mono
            x_mono = x.mean(dim=1)  # [B, T]
        
        # Apply STFT to each batch element
        stft_results = []
        for i in range(B):
            # Ensure minimum length for STFT
            audio_slice = x_mono[i]
            if audio_slice.shape[0] < self.n_fft:
                # Pad with zeros if too short
                padding = self.n_fft - audio_slice.shape[0]
                audio_slice = torch.cat([audio_slice, torch.zeros(padding, device=audio_slice.device)])
            
            # Apply STFT
            stft_result = torch.stft(
                audio_slice,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True,
                window=torch.hann_window(self.n_fft, device=audio_slice.device)
            )  # [F, T]
            
            stft_results.append(stft_result)
        
        # Stack results
        x_stft = torch.stack(stft_results)  # [B, F, T]
        
        return x_stft

    async def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process audio input with optimization and caching.
        x: [B, C, T] real tensor
        returns: [B, output_dim] complex multivector
        """
        B, C, T = x.shape
        assert C == self.input_channels
        
        # Check deduplication
        audio_hash = self._compute_audio_hash(x)
        cached_result = self._check_deduplication(audio_hash, x)
        if cached_result is not None:
            return cached_result
        
        # Process STFT
        x_stft = self._process_audio_stft(x)  # [B, F, T]
        
        # Process frequency domain - transpose to [B, T, F] for Linear layer
        x_stft_transposed = x_stft.transpose(1, 2)  # [B, T, F]
        x_freq = self.freq_processor(x_stft_transposed.abs())  # [B, T, n_mels]
        
        # Transpose back for temporal processing
        x_freq_temporal = x_freq.transpose(1, 2)  # [B, n_mels, T]
        
        # Process temporal domain
        x_temp = self.temporal(x_freq_temporal)  # [B, n_mels*4, T]
        
        # Global pooling
        x_pool = x_temp.mean(dim=-1)  # [B, n_mels*4]
        
        # Project to output dimension
        x_proj = self.projection(x_pool)  # [B, output_dim]
        
        # Convert to multivector representation
        mv = self.ops.tensor_to_multivector(x_proj, n_dims=None)
        
        # Store for deduplication
        self._store_deduplication(audio_hash, mv)
        
        return mv

    def forward_sync(self, x: torch.Tensor) -> torch.Tensor:
        """Sync version for ONNX export and testing."""
        B, C, T = x.shape
        
        # Handle single sample case
        if B == 1 and C == 1:
            # Single mono audio sample
            x_mono = x.squeeze()  # [T]
            
            # Ensure minimum length
            if x_mono.shape[0] < self.n_fft:
                padding = self.n_fft - x_mono.shape[0]
                x_mono = torch.cat([x_mono, torch.zeros(padding, device=x_mono.device)])
            
            # Apply STFT
            x_stft = torch.stft(
                x_mono,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True,
                window=torch.hann_window(self.n_fft, device=x_mono.device)
            )  # [F, T]
            
            # Process frequency domain
            x_freq = self.freq_processor(x_stft.abs())  # [n_mels, T]
            
            # Process temporal domain
            x_temp = self.temporal(x_freq.unsqueeze(0))  # [1, n_mels*4, T]
            
            # Global pooling
            x_pool = x_temp.mean(dim=-1).squeeze(0)  # [n_mels*4]
            
            # Project to output dimension
            x_proj = self.projection(x_pool)  # [output_dim]
            
            # Convert to multivector
            mv = self.ops.tensor_to_multivector(x_proj, n_dims=None)
            
            return mv
        else:
            # Handle batch case
            return self.forward(x)

    async def cleanup(self):
        """Clean up resources."""
        self.audio_cache.clear()

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.audio_cache.clear()
        except:
            pass


class ModalityManager(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.modalities = nn.ModuleDict()
        # Use shared ops instance
        self.ops = get_shared_ops()
        print("[ModalityManager] I got initialized successfully! :D")
        
    def add_modality(self, name: str, modality_config: dict):
        """Add a new modality to the model."""
        if name == "text":
            self._init_text_modality(modality_config)
        elif name == "image":
            self._init_image_modality(modality_config)
        elif name == "audio":
            self._init_audio_modality(modality_config)
        else:
            raise ValueError(f"Unknown modality: {name}")
        print(f"[ModalityManager] Added {name} modality")

    def _init_text_modality(self, config):
        """Initialize text modality with Word2Vec tokenizer."""
        try:
            # Create tokenizer
            text_tokenizer = SubwordTokenizer(
                w2v_path=config.get("w2v_path"),
                unk_token=config.get("unk_token", "<UNK>"),
                pad_token=config.get("pad_token", "<PAD>"),
                bos_token=config.get("bos_token", "<bos>"),
                eos_token=config.get("eos_token", "<eos>"),
                device="cpu",  # Force CPU for better async performance
                ops=self.ops,  # Pass the shared ops instance
                embedding_dim=self.config.d_model,
                num_basis=32,
                quantize=config.get("quantize", True),
                quantize_bits=config.get("quantize_bits", 4),
                use_mmap=config.get("use_mmap", True),
                w2v_quantize_bits=config.get("w2v_quantize_bits", 8),
                w2v_cache_size=config.get("w2v_cache_size", 10000),
                w2v_mmap_threshold=config.get("w2v_mmap_threshold", 50000),
            )
            
            # Build initial vocabulary with some common words
            initial_texts = [
                "hello world",
                "this is a test",
                "the quick brown fox",
                "jumps over the lazy dog",
                "artificial intelligence",
                "machine learning",
                "deep learning",
                "neural networks",
                "computer science",
                "data science",
                "natural language processing",
                "computer vision",
                "reinforcement learning",
                "transfer learning",
                "unsupervised learning",
                "supervised learning",
                "semi supervised learning",
                "self supervised learning",
                "few shot learning",
                "zero shot learning",
                "four years old",
                "complex numbers",
                "symbolic reasoning",
                "I am Steve"
            ]
            text_tokenizer._build_vocab(initial_texts)
            
            # Create projection layer with proper dimensions
            text_proj = LinearLayer(
                text_tokenizer.num_basis,  # Input dimension matches token function basis size
                self.config.d_model,  # Output dimension matches model dimension
            )
            
            # Store components temporarily
            self.modalities["text_tokenizer"] = text_tokenizer
            self.modalities["text_proj"] = text_proj
            
            print("[TextModality] Initialized successfully with tokenizer and projection")
            
        except Exception as e:
            print(f"[TextModality] Initialization failed: {e}")
            raise ValueError(f"Failed to initialize text modality: {e}")

    def _init_image_modality(self, config):
        self.modalities["image_encoder"] = ImageModality(
            ops=self.ops,  # Pass ops to ImageModality
            input_channels=config.get("input_channels", 3),
            output_dim=self.config.d_model,
            dropout=config.get("dropout_rate", 0.1),
        )
        self.modalities["image_proj"] = LinearLayer(
            self.config.d_model,
            self.config.d_model,
        )

    def _init_audio_modality(self, config):
        self.modalities["audio_encoder"] = AudioModality(
            input_channels=config.get("input_channels", 1),
            output_dim=self.config.d_model,
            sample_rate=config.get("sample_rate", 44100),
            n_fft=config.get("n_fft", 2048),
            hop_length=config.get("hop_length", 512),
            n_mels=config.get("n_mels", 128),
            dropout=config.get("dropout_rate", 0.1),
            ops=self.ops,  # Pass ops to AudioModality
        )
        self.modalities["audio_proj"] = LinearLayer(
            self.config.d_model,
            self.config.d_model,
        )

    async def process_text(self, text):
        """Process text input through the text modality pipeline."""
        if "text_tokenizer" not in self.modalities:
            raise ValueError("Text modality not initialized!")
            
        # Get tokenizer and projection
        text_tokenizer = self.modalities["text_tokenizer"]
        text_proj = self.modalities["text_proj"]
        
        # Get token IDs and functions from encoder
        token_ids, token_functions = await text_tokenizer.encode(text)
        print(f"[ModalityManager] Token IDs shape: {token_ids.shape if hasattr(token_ids, 'shape') else len(token_ids)}")
        print(f"[ModalityManager] Number of token functions: {len(token_functions)}")
        
        # Convert token IDs to embeddings using the embed method
        text_emb = text_tokenizer.embed(token_ids, token_functions)
        print(f"[ModalityManager] Text embeddings shape: {text_emb.shape}")
        
        # Project embeddings
        projected = text_proj(text_emb)
        print(f"[ModalityManager] Projected embeddings shape: {projected.shape}")
        
        return projected

    def process_image(self, images):
        img_features = self.modalities["image_encoder"](images)
        return self.modalities["image_proj"](img_features)

    def process_audio(self, audio):
        audio_features = self.modalities["audio_encoder"](audio)
        return self.modalities["audio_proj"](audio_features)
