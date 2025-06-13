from typing import Dict, List, Optional, Set, Tuple, Union, Any
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .config import ModelConfig, Modality
from .ops import MultivectorOps, TensorCache, MixedPrecisionRouter, Quantizer
from .shared import get_shared_ops
import gensim
from gensim.models import KeyedVectors
from .core import LinearLayer, FeatureRouter
import re
import gc
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from functools import lru_cache
import json
import logging
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import aiohttp
from collections import deque
import hashlib
from threading import Lock
import array
import mmap
import struct
import ctypes
from dataclasses import dataclass
import bitarray
import numpy.typing as npt
import queue
import time
import warnings
import numba
from numba import jit, prange
from scipy.fft import dct, idct
from .complex_tokens import (
    ComplexBasisSet,
    ComplexTokenEmbedding,
    TokenFunction,
    ComplexTokenStream,
    _compute_basis_functions,
)


@jit(nopython=True, parallel=True)
def _process_tokens_numba(tokens: np.ndarray, vocab: np.ndarray) -> np.ndarray:
    """Process tokens using Numba for faster tokenization."""
    n = len(tokens)
    result = np.zeros(n, dtype=np.int64)
    for i in prange(n):
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

    def to_tokens(self) -> List[int]:
        """Convert bit-packed view to token list."""
        tokens = []
        for i in range(self.length):
            if self.data[self.start + i]:
                tokens.append(i)
        return tokens

    def slice(self, start: int, length: int) -> "TokenView":
        """Create a new view into the same data."""
        return TokenView(self.data, self.start + start, length)


class RingBuffer:
    """Lock-free ring buffer for token streaming."""

    def __init__(self, size: int, dtype=np.int32):
        self.size = size
        self.buffer = np.zeros(size, dtype=dtype)
        self.head = 0
        self.tail = 0
        self.mask = size - 1

    def push(self, item: int) -> bool:
        """Push item to buffer if space available."""
        next_head = (self.head + 1) & self.mask
        if next_head == self.tail:
            return False
        self.buffer[self.head] = item
        self.head = next_head
        return True

    def pop(self) -> Optional[int]:
        """Pop item from buffer if available."""
        if self.head == self.tail:
            return None
        item = self.buffer[self.tail]
        self.tail = (self.tail + 1) & self.mask
        return item

    def peek(self) -> Optional[int]:
        """Peek at next item without removing."""
        if self.head == self.tail:
            return None
        return self.buffer[self.tail]


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

    def get_token(self, idx: int) -> Tuple[np.ndarray, bool]:
        """Get token vector and mask."""
        return self.data[idx], self.masks[idx]

    def batch_get(self, indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get multiple tokens efficiently."""
        return self.data[indices], self.masks[indices]


class TokenStream:
    """Lock-free token stream with ring buffer and SIMD processing."""

    def __init__(self, buffer_size: int = 1024, num_workers: int = 4):
        self.buffer = RingBuffer(buffer_size)
        self.workers = ThreadPoolExecutor(max_workers=num_workers)
        self.views: List[TokenView] = []

    def push_tokens(self, tokens: List[int]):
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
    def __init__(
        self,
        ops: MultivectorOps,
        input_channels: int = 3,
        output_dim: int = 256,
        channels: List[int] = [64, 128, 256, 512],
        kernel_sizes: Optional[List[int]] = None,
        pool_type: str = "adaptive",  # "adaptive" | "max" | "none"
        pool_size: Tuple[int, int] = (4, 4),  # for adaptive
        dropout: float = 0.3,
        activation: str = "gelu",
        use_batchnorm: bool = True,
        min_resolution: int = 32,
    ):
        super().__init__()
        assert pool_type in ("adaptive", "max", "none")

        self.ops = ops
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.channels = channels
        self.kernel_sizes = kernel_sizes or [3] * len(channels)
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.use_bn = use_batchnorm
        self.min_resolution = min_resolution

        # Build conv params for each block
        self.convs = nn.ModuleList()
        in_c = input_channels
        for out_c, k in zip(channels, self.kernel_sizes):
            conv = nn.ModuleDict(
                {
                    "weight": nn.Parameter(
                        torch.randn(out_c, in_c, k, k, dtype=torch.complex64) * 0.02
                    ),
                    "bias": nn.Parameter(torch.zeros(out_c, dtype=torch.complex64)),
                    # real & imag BatchNorm
                    "bn_real": nn.BatchNorm2d(out_c) if use_batchnorm else None,
                    "bn_imag": nn.BatchNorm2d(out_c) if use_batchnorm else None,
                    "act": getattr(nn, activation.title())()
                    if hasattr(nn, activation.title())
                    else nn.GELU(),
                }
            )
            self.convs.append(conv)
            in_c = out_c

        # Pooling
        if pool_type == "adaptive":
            self.pool = nn.AdaptiveAvgPool2d(pool_size)
        elif pool_type == "max":
            self.pool = nn.MaxPool2d(2)
        else:
            self.pool = None

        # FC as weight & bias (complex)
        # flatten_dim computed lazily on first forward if pool_type=="none"
        self.fc_weight = nn.Parameter(
            torch.randn(1, output_dim, dtype=torch.complex64)
        )  # placeholder
        self.fc_bias = nn.Parameter(torch.zeros(output_dim, dtype=torch.complex64))
        self._flatten_dim = None

        self.dropout = nn.Dropout2d(dropout)

    async def forward(
        self, x: torch.Tensor, input_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """
        x: real tensor [B, C, H, W]
        returns: complex multivector [B, output_dim, mv_dim]
        """
        B, C, H, W = x.shape
        assert C == self.input_channels
        assert H >= self.min_resolution and W >= self.min_resolution

        # promote to complex
        x_c = x.to(torch.complex64)

        # conv blocks
        for conv in self.convs:
            w, b = conv["weight"], conv["bias"]
            x_c = await self.ops.conv2d(x_c, w, b)  # complex conv
            if self.use_bn:
                real = conv["bn_real"](x_c.real)
                imag = conv["bn_imag"](x_c.imag)
                x_c = torch.complex(real, imag)
            x_c = conv["act"](x_c)
            x_c = self.dropout(x_c)

        # pooling / flatten
        if self.pool:
            # apply to real & imag separately
            r = self.pool(x_c.real)
            i = self.pool(x_c.imag)
            x_flat = torch.complex(r, i).flatten(1)
        else:
            # no pooling: flatten full H*W → determine dim once
            flat = x_c.flatten(1)  # [B, C*H*W]
            if self._flatten_dim is None:
                self._flatten_dim = flat.shape[-1]
                # reinit fc weight to correct shape
                self.fc_weight.data = torch.randn(
                    self._flatten_dim, self.output_dim, dtype=torch.complex64
                )
            x_flat = flat

        # linear projection
        # [B, flatten_dim] @ [flatten_dim, output_dim] => [B, output_dim]
        out = await self.ops.matmul(x_flat, self.fc_weight)
        out = out + self.fc_bias

        # embed as multivector for SillyAI
        mv = self.ops.tensor_to_multivector(out, n_dims=None)
        return mv

    def forward_sync(self, x: torch.Tensor) -> torch.Tensor:
        """Sync path for ONNX export; mirrors async logic."""
        B, C, H, W = x.shape
        x_c = x.to(torch.complex64)
        for conv in self.convs:
            w, b = conv["weight"], conv["bias"]
            x_c = self.ops.conv2d(x_c, w, b)
            if self.use_bn:
                real = conv["bn_real"](x_c.real)
                imag = conv["bn_imag"](x_c.imag)
                x_c = torch.complex(real, imag)
            x_c = conv["act"](x_c)
            x_c = self.dropout(x_c)

        if self.pool:
            r = self.pool(x_c.real)
            i = self.pool(x_c.imag)
            x_flat = torch.complex(r, i).flatten(1)
        else:
            flat = x_c.flatten(1)
            if self._flatten_dim is None:
                self._flatten_dim = flat.shape[-1]
                self.fc_weight.data = torch.randn(
                    self._flatten_dim, self.output_dim, dtype=torch.complex64
                )
            x_flat = flat

        out = self.ops.matmul_sync(x_flat, self.fc_weight) + self.fc_bias
        mv = self.ops.tensor_to_multivector(out, n_dims=None)
        return mv


class Word2VecTokenizer(nn.Module):
    """Word2Vec-based tokenizer with complex-valued periodic functions and Numba acceleration."""

    def __init__(
        self,
        w2v_path: str,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        lowercase: bool = True,
        device: str = "cpu",
        ops: Optional[MultivectorOps] = None,
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
        num_basis: int = 32,
    ):
        super().__init__()

        # Core parameters
        self.w2v_path = w2v_path
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.lowercase = lowercase
        self.device = device
        self.ops = ops or get_shared_ops()
        self.max_vocab_size = max_vocab_size
        self.cache_size = cache_size
        self.quantize = quantize
        self.quantize_bits = quantize_bits
        self.use_mmap = use_mmap
        self.chunk_size = chunk_size
        self.subword_fallback = subword_fallback
        self.imag_scale = 0.1
        self.num_basis = num_basis
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.buffer_size = buffer_size

        # Initialize data structures
        self.vocab = {}
        self.reverse_vocab = {}
        self._model = None
        self._embedding = None
        self._embedding_chunks = {}  # Store embeddings in chunks
        self._chunk_lock = Lock()

        # Initialize components with memory-efficient settings
        self.precision_router = MixedPrecisionRouter()
        self.quantizer = Quantizer(
            qmin=-(2 ** (quantize_bits - 1)), qmax=2 ** (quantize_bits - 1) - 1
        )

        # Initialize complex-valued components with reduced memory footprint
        self.basis_set = ComplexBasisSet(num_basis)
        self.token_embedding = ComplexTokenEmbedding(
            max_vocab_size, num_basis, ops=self.ops
        )

        # Initialize lock-free data structures with optimized memory usage
        self.token_stream = ComplexTokenStream(buffer_size, num_workers, num_basis)
        self.bloom_trie = ConcurrentBloomTrie(cache_size)
        
        # Build initial vocab to get vector size
        self._build_vocab_from_header()
        
        # Initialize token storage with correct vector size
        self.token_storage = AoSoA(max_vocab_size, self.vec_size)

        # Initialize parallel processing with memory-aware settings
        self._init_parallel_processing()

        print("[Word2VecTokenizer] Initialized with complex-valued periodic functions!")

    def _build_vocab_from_header(self):
        """Build initial vocab from file header with memory-efficient approach."""
        # Read header without loading full file
        with open(self.w2v_path, "rb") as f:
            header = f.readline().decode("utf-8").strip().split()
            if len(header) == 2:
                self.vocab_size = int(header[0])
                self.vec_size = int(header[1])
            else:
                self.vocab_size = int(header[0])
                self.vec_size = int(header[1])

        # Add special tokens with memory-efficient storage
        self.vocab[self.pad_token] = 0
        self.vocab[self.unk_token] = 1
        self.reverse_vocab[0] = self.pad_token
        self.reverse_vocab[1] = self.unk_token

        # Add to bloom trie for fast lookups
        self.bloom_trie.add(self.pad_token)
        self.bloom_trie.add(self.unk_token)

        # Initialize token stream with special tokens
        self.token_stream.push_tokens([0, 1])  # Add pad and unk tokens

    def _init_parallel_processing(self):
        """Initialize parallel processing components with memory awareness."""
        # Create thread pool for tokenization with limited workers
        self.tokenizer_pool = ThreadPoolExecutor(
            max_workers=min(self.num_workers, 4),  # Limit initial workers
            thread_name_prefix="tokenizer"
        )

        # Create process pool for heavy computations with memory limits
        self.process_pool = ProcessPoolExecutor(
            max_workers=min(self.num_workers, 2),  # Start with fewer processes
            mp_context=mp.get_context("spawn")
        )

    def _load_embedding_chunk(self, chunk_idx: int) -> torch.Tensor:
        """Load a chunk of embeddings with memory-efficient processing."""
        with self._chunk_lock:
            if chunk_idx in self._embedding_chunks:
                return self._embedding_chunks[chunk_idx]

            # Calculate chunk boundaries
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, self.vocab_size)

            # Get vectors for this chunk using AoSoA
            vectors = []
            for i in range(start_idx, end_idx):
                if i < len(self.model.index_to_key):
                    word = self.model.index_to_key[i]
                    # Try to get from token storage first
                    vec, exists = self.token_storage.get_token(i)
                    if not exists:
                        vec = self.model[word]
                        # Store in token storage for future use
                        self.token_storage.set_token(i, vec, True)
                    vectors.append(vec)
                else:
                    # Use zero vector from token storage
                    vec, _ = self.token_storage.get_token(0)  # Get pad token vector
                    vectors.append(vec)

            # Convert to tensor with memory-efficient dtype
            chunk = torch.tensor(vectors, dtype=torch.float32, device=self.device)
            
            # Store in cache with size limit
            self._embedding_chunks[chunk_idx] = chunk
            
            # Implement LRU cache
            if len(self._embedding_chunks) > 10:
                oldest_chunk = min(self._embedding_chunks.keys())
                del self._embedding_chunks[oldest_chunk]

            return chunk

    def get_embedding_matrix(self) -> torch.Tensor:
        """Get the embedding matrix with memory-efficient loading."""
        if self._embedding is None:
            # Create empty tensor with memory-efficient dtype
            self._embedding = torch.zeros(
                (self.vocab_size, self.vec_size),
                dtype=torch.float32,
                device=self.device
            )
            
            # Load chunks in parallel with memory limits
            with ThreadPoolExecutor(max_workers=min(self.num_workers, 4)) as executor:
                futures = []
                num_chunks = (self.vocab_size + self.chunk_size - 1) // self.chunk_size
                
                # Process chunks in smaller batches
                batch_size = 2  # Process 2 chunks at a time
                for i in range(0, num_chunks, batch_size):
                    batch_end = min(i + batch_size, num_chunks)
                    batch_futures = []
                    
                    for j in range(i, batch_end):
                        future = executor.submit(self._load_embedding_chunk, j)
                        batch_futures.append((j, future))
                    
                    # Process batch results
                    for chunk_idx, future in batch_futures:
                        chunk = future.result()
                        start_idx = chunk_idx * self.chunk_size
                        end_idx = min(start_idx + self.chunk_size, self.vocab_size)
                        self._embedding[start_idx:end_idx] = chunk
                        
                        # Clear some memory
                        if chunk_idx in self._embedding_chunks:
                            del self._embedding_chunks[chunk_idx]
                    
                    # Force garbage collection after each batch
                    gc.collect()

        return self._embedding

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

    def tokenize(self, text: str) -> Tuple[List[str], List[TokenFunction]]:
        """Tokenize text with complex-valued periodic functions."""
        # Check bloom trie first
        if not self.bloom_trie.contains(text):
            # Process text
            if self.lowercase:
                text = text.lower()

            # Split into sentences for parallel processing
            sentences = text.split(".")

            # Process sentences in parallel
            with self.tokenizer_pool as pool:
                futures = []
                for sentence in sentences:
                    if sentence.strip():
                        future = pool.submit(self._tokenize_sentence, sentence)
                        futures.append(future)

                # Collect results
                tokens = []
                functions = []
                for future in futures:
                    sent_tokens, sent_funcs = future.result()
                    tokens.extend(sent_tokens)
                    functions.extend(sent_funcs)

            # Add to bloom trie
            self.bloom_trie.add(text)

            # Push tokens to stream
            self.token_stream.push_tokens(tokens, functions)

            return tokens, functions

        # Get from stream if in bloom trie
        return self.token_stream.process_stream()

    def _tokenize_sentence(
        self, sentence: str
    ) -> Tuple[List[str], List[TokenFunction]]:
        """Tokenize a single sentence with complex-valued periodic functions."""
        # Use regex for better tokenization
        words = re.findall(r"\b\w+(?:'\w+)?\b", sentence)

        # Process words in parallel
        with self.tokenizer_pool as pool:
            futures = []
            for word in words:
                future = pool.submit(self._process_word, word)
                futures.append(future)

            # Collect results
            tokens = []
            functions = []
            for future in futures:
                token, func = future.result()
                tokens.append(token)
                functions.append(func)

        return tokens, functions

    def _process_word(self, word: str) -> Tuple[str, TokenFunction]:
        """Process a single word with complex-valued periodic functions."""
        if word in self.vocab:
            # Get vector from model
            vec = self.model[word]
            # Create token function
            func = self._create_token_function(word, vec)
            return word, func

        # Try subword matching
        if self.subword_fallback:
            subword = self._subword(word)
            if subword in self.model:
                vec = self.model[subword]
                func = self._create_token_function(subword, vec)
                return subword, func

        # Create default function for unknown tokens
        func = TokenFunction(self.num_basis)
        return self.unk_token, func

    def _subword(self, tok: str) -> str:
        """Find best matching subword with SIMD optimizations."""
        if not self.subword_fallback:
            return self.unk_token

        # Process subwords in parallel
        with self.tokenizer_pool as pool:
            futures = []
            for i in range(len(tok)):
                subword = tok[i:]
                future = pool.submit(self._check_subword, subword)
                futures.append(future)

            # Collect results
            for future in futures:
                result = future.result()
                if result:
                    return result

        return self.unk_token

    def _check_subword(self, subword: str) -> Optional[str]:
        """Check if a subword exists in the index."""
        if subword in self.subword_index:
            return self.subword_index[subword][0]
        return None

    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        padding: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Union[Tuple[List[int], List[TokenFunction]], torch.Tensor]:
        """Encode text to token IDs and functions."""
        # Tokenize
        tokens, functions = self.tokenize(text)

        # Process tokens in parallel with Numba
        ids = _process_tokens_numba(np.array(tokens), np.array(list(self.vocab.keys())))

        # Truncate if needed
        if max_length:
            ids = ids[:max_length]
            functions = functions[:max_length]

        # Pad if needed
        if padding and max_length:
            pad_len = max_length - len(ids)
            ids = np.pad(ids, (0, pad_len), constant_values=0)
            functions.extend([TokenFunction(self.num_basis) for _ in range(pad_len)])

        # Convert to tensor if requested
        if return_tensors == "pt":
            return torch.tensor(ids, device=self.device), functions

        return ids.tolist(), functions

    def batch_encode(
        self, texts: List[str], max_length: Optional[int] = None
    ) -> Dict[str, Union[torch.Tensor, List[List[TokenFunction]]]]:
        """Encode a batch of texts with complex-valued periodic functions."""
        # Process texts in parallel
        with self.tokenizer_pool as pool:
            futures = []
            for text in texts:
                future = pool.submit(
                    self.encode, text, max_length, padding=True, return_tensors="pt"
                )
                futures.append(future)

            # Collect results
            results = [f.result() for f in futures]

        # Unzip results
        ids, functions = zip(*results)

        # Stack tensors
        return {
            "input_ids": torch.stack(ids),
            "token_functions": functions,
            "attention_mask": (torch.stack(ids) != 0).long(),
        }

    def decode(
        self,
        ids: Union[List[int], torch.Tensor],
        functions: Optional[List[TokenFunction]] = None,
    ) -> str:
        """Decode token IDs and functions to text."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()

        # Process IDs in parallel
        with self.tokenizer_pool as pool:
            futures = []
            for i, id in enumerate(ids):
                future = pool.submit(
                    self._decode_id, id, functions[i] if functions else None
                )
                futures.append(future)

            # Collect results
            return " ".join([f.result() for f in futures])

    def _decode_id(self, id: int, func: Optional[TokenFunction] = None) -> str:
        """Decode a single ID and function."""
        word = self.reverse_vocab.get(id, self.unk_token)
        if func and word != self.unk_token:
            # Update token function if provided
            self.token_stream.token_functions[id] = func
        return word

    def embed(
        self,
        input_ids: torch.Tensor,
        token_functions: Optional[List[TokenFunction]] = None,
    ) -> torch.Tensor:
        """Get embeddings for input IDs and functions."""
        if token_functions:
            # Use complex-valued periodic functions
            funcs = [func.to_tensor() for func in token_functions]
            return torch.stack(funcs)
        else:
            # Fallback to regular embeddings
            return self.get_embedding_matrix()[input_ids]

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

    def __del__(self):
        """Enhanced cleanup with memory management."""
        # Cleanup parallel processing
        if hasattr(self, 'tokenizer_pool'):
            self.tokenizer_pool.shutdown()
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown()

        # Cleanup token stream
        if hasattr(self, 'token_stream'):
            self.token_stream.close()

        # Cleanup model and embeddings
        if self._model is not None:
            del self._model
        if self._embedding is not None:
            del self._embedding
        self._embedding_chunks.clear()

        # Cleanup token storage
        if hasattr(self, 'token_storage'):
            del self.token_storage

        # Clear vocab dictionaries
        self.vocab.clear()
        self.reverse_vocab.clear()

        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


class AudioModality(nn.Module):
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
    ):
        super().__init__()
        self.ops = ops
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # STFT parameters
        self.stft = nn.Sequential(
            nn.Linear(n_fft // 2 + 1, n_mels),
            nn.LayerNorm(n_mels),
            getattr(nn, activation.title())()
            if hasattr(nn, activation.title())
            else nn.GELU(),
            nn.Dropout(dropout),
        )

        # Temporal processing
        self.temporal = nn.Sequential(
            nn.Conv1d(n_mels, n_mels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_mels * 2) if use_batchnorm else nn.Identity(),
            getattr(nn, activation.title())()
            if hasattr(nn, activation.title())
            else nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(n_mels * 2, n_mels * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_mels * 4) if use_batchnorm else nn.Identity(),
            getattr(nn, activation.title())()
            if hasattr(nn, activation.title())
            else nn.GELU(),
            nn.Dropout(dropout),
        )

        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(n_mels * 4, output_dim),
            nn.LayerNorm(output_dim),
            getattr(nn, activation.title())()
            if hasattr(nn, activation.title())
            else nn.GELU(),
            nn.Dropout(dropout),
        )

    async def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process audio input.
        x: [B, C, T] real tensor
        returns: [B, output_dim, mv_dim] complex multivector
        """
        B, C, T = x.shape
        assert C == self.input_channels

        # Convert to complex via STFT
        x_stft = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True
        )  # [B, C, F, T]

        # Process frequency domain
        x_freq = self.stft(x_stft.abs())  # [B, C, n_mels, T]

        # Process temporal domain
        x_temp = self.temporal(x_freq)  # [B, C, n_mels*4, T]

        # Global pooling
        x_pool = x_temp.mean(dim=-1)  # [B, C, n_mels*4]

        # Project to output dimension
        x_proj = self.projection(x_pool)  # [B, C, output_dim]

        # Convert to multivector
        mv = self.ops.tensor_to_multivector(x_proj, n_dims=None)
        return mv

    def forward_sync(self, x: torch.Tensor) -> torch.Tensor:
        """Sync version for ONNX export."""
        B, C, T = x.shape
        x_stft = torch.stft(
            x, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True
        )
        x_freq = self.stft(x_stft.abs())
        x_temp = self.temporal(x_freq)
        x_pool = x_temp.mean(dim=-1)
        x_proj = self.projection(x_pool)
        mv = self.ops.tensor_to_multivector(x_proj, n_dims=None)
        return mv


class ModalityManager(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.modalities = nn.ModuleDict()
        self.ops = MultivectorOps().compile()  # Create single ops instance
        print("[ModalityManager] I got initialized successfully! :D")

    def add_modality(self, name: str, modality_config: dict):
        if name == "text":
            self._init_text_modality(modality_config)
        elif name == "image":
            self._init_image_modality(modality_config)
        elif name == "audio":
            self._init_audio_modality(modality_config)

    def _init_text_modality(self, config):
        self.modalities["text_tokenizer"] = Word2VecTokenizer(
            w2v_path=config["w2v_path"],
            unk_token=config.get("unk_token", "<UNK>"),
            pad_token=config.get("pad_token", "<PAD>"),
            lowercase=config.get("lowercase", True),
            device=self.config.device,
            ops=self.ops,  # Use shared ops instance
            max_vocab_size=config.get(
                "max_vocab_size", 50000
            ),  # Limit vocab size for speed
            cache_size=config.get(
                "cache_size", 10000
            ),  # Size of LRU cache for tokenization
            quantize=config.get("quantize", True),  # Whether to quantize embeddings
            quantize_bits=config.get(
                "quantize_bits", 8
            ),  # Number of bits for quantization
            use_mmap=config.get(
                "use_mmap", True
            ),  # Whether to use memory mapping for large files
            chunk_size=config.get(
                "chunk_size", 100000
            ),  # Size of chunks for processing large files
            subword_fallback=config.get("subword_fallback", True),
        )
        self.modalities["text_proj"] = LinearLayer(
            self.modalities["text_tokenizer"].get_embedding_matrix().shape[1],
            self.config.d_model,
            ops=self.ops,  # Pass ops to LinearLayer
        )

    def _init_image_modality(self, config):
        self.modalities["image_encoder"] = ImageModality(
            input_channels=config.get("input_channels", 3),
            output_dim=self.config.d_model,
            input_size=config.get("input_size", (224, 224)),
            dropout_rate=config.get("dropout_rate", 0.1),
            ops=self.ops,  # Pass ops to ImageModality
        )
        self.modalities["image_proj"] = LinearLayer(
            self.config.d_model,
            self.config.d_model,
            ops=self.ops,  # Pass ops to LinearLayer
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
            ops=self.ops,  # Pass ops to LinearLayer
        )

    def process_text(self, text):
        if isinstance(text, str):
            text = [text]
        encoded = self.modalities["text_tokenizer"].batch_encode(text)
        text_emb = self.modalities["text_tokenizer"].embed(encoded["input_ids"])
        return self.modalities["text_proj"](text_emb)

    def process_image(self, images):
        img_features = self.modalities["image_encoder"](images)
        return self.modalities["image_proj"](img_features)

    def process_audio(self, audio):
        audio_features = self.modalities["audio_encoder"](audio)
        return self.modalities["audio_proj"](audio_features)
