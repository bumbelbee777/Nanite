import asyncio
import hashlib
import io
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any
import time

import numpy as np
import torch
import torch.nn.functional as F
import xxhash
from torch import linalg as LA
from torch import nn
from bitarray import bitarray

from .config import ModelConfig, PrecisionLevel
import nanite.backend.BitplaneEngine as BitplaneEngine


# Quantizer with real/cartesian/polar, sync+async entrypoints, JIT/compile-ready
class Quantizer(nn.Module):
    """Quantizes real or complex tensors for efficient storage and computation."""
    def __init__(self, qmin, qmax, init_scale=1.0, mode="real"):
        super().__init__()
        self.qmin, self.qmax, self.mode = qmin, qmax, mode
        self.scale = nn.Parameter(torch.tensor(init_scale))
        self.zero_point = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        # Unified logic for real/complex
        if isinstance(x, torch.Tensor) and torch.is_complex(x):
            r, i = x.real, x.imag
            qr = self._qs(r)[0]
            qi = self._qs(i)[0]
            # Ensure qr and qi are real-valued tensors
            if torch.is_complex(qr):
                qr = qr.real
            if torch.is_complex(qi):
                qi = qi.real
            qr = qr.reshape(r.shape)
            qi = qi.reshape(i.shape)
            return torch.complex(qr, qi)
        else:
            q = self._qs(x)[0]
            if isinstance(x, torch.Tensor):
                q = q.reshape(x.shape)
            return q

    @torch.jit.export
    def dequantize(self, q):
        zp, sc = self.zero_point, self.scale
        if isinstance(q, torch.Tensor) and torch.is_complex(q):
            r, i = q.real, q.imag
            dr = (r - zp) * sc
            di = (i - zp) * sc
            return torch.complex(dr, di)
        else:
            return (q.float() - zp) * sc

    def _qs(self, x):
        # Ensure x is real-valued before quantization
        x = x.real if isinstance(x, torch.Tensor) and torch.is_complex(x) else x
        q = MultivectorOps.clamp(
            torch.round(x / self.scale + self.zero_point),
            self.qmin,
            self.qmax,
        )
        # Ensure q is a tensor and has the same shape as x
        if not isinstance(q, torch.Tensor):
            q = torch.tensor(q, dtype=x.dtype if isinstance(x, torch.Tensor) else torch.float32)
        if isinstance(x, torch.Tensor):
            q = q.reshape(x.shape)
        return (q,)


@dataclass
class PrecisionConfig:
    bits: int
    qmin: float
    qmax: float


PRECISION_CONFIGS = {
    PrecisionLevel.TERNARY: PrecisionConfig(2, -1, 1),
    PrecisionLevel.INT4: PrecisionConfig(4, -8, 7),
    PrecisionLevel.FP4: PrecisionConfig(4, -7, 7),
    PrecisionLevel.FP8: PrecisionConfig(8, -128, 127),
    PrecisionLevel.FP16: PrecisionConfig(16, -65504, 65504),
}


class MixedPrecisionRouter:
    def __init__(self, small=1024, low_rank=16, large=1_000_000):
        self.s, self.lr, self.l = small, low_rank, large

    def get_precision(self, t):
        n = t.numel()
        if n <= self.s:
            return PrecisionLevel.TERNARY
        if t.ndim >= 2:
            try:
                r = int(LA.matrix_rank(t.reshape(-1, t.shape[-1])).item())
            except:
                r = min(*t.shape[-2:])
            if r <= self.lr:
                return PrecisionLevel.INT4
        return PrecisionLevel.FP8 if n >= self.l else PrecisionLevel.FP4


class TensorCache:
    """Python wrapper for the C++ TensorCache (BitplaneEngine). Maintains async interface for compatibility, now with batch/zero-copy/lazy caching."""
    def __init__(self, max_bytes, *args, **kwargs):
        self._cache = BitplaneEngine.TensorCache(max_bytes)
        self._cache.StartWorker()
        self._pending_batch = []
        self._batch_size = 16

    async def put(self, tag, fused, tensor_hash):
        # Lazy batch put: accumulate, then flush in batch
        self._pending_batch.append((tag, fused, tensor_hash))
        if len(self._pending_batch) >= self._batch_size:
            for tag, fused, tensor_hash in self._pending_batch:
                # Zero-copy: pass numpy arrays directly to C++
                if hasattr(fused, 'numpy'):
                    arr = np.asarray(fused)
        else:
            arr = fused
            self._cache.Put(tag, arr, tensor_hash)
            self._pending_batch.clear()

    async def flush(self):
        # Flush any remaining pending puts
        for tag, fused, tensor_hash in self._pending_batch:
            if hasattr(fused, 'numpy'):
                arr = np.asarray(fused)
        else:
            arr = fused
            self._cache.Put(tag, arr, tensor_hash)
            self._pending_batch.clear()

    async def get(self, tag, *args, mvops=None):
        return self._cache.Get(tag)

    def clear(self):
        self._cache.Clear()

    def size(self):
        return self._cache.Size()

    def start_worker(self):
        self._cache.StartWorker()

    def stop_worker(self):
        self._cache.StopWorker()


class TensorHasher:
    """
    Fast, lock-free tensor hashing utility using the C++ BitplaneEngine.HashTensor implementation.
    """
    def __init__(self, strategy: str = "auto", cache_size: int = 1024):
        self.strategy = strategy
        # No Python-side cache or dedup table needed; all logic is in C++

    def hash_tensor(self, tensor: torch.Tensor) -> int:
        arr = tensor.detach().cpu().numpy()
        return int(BitplaneEngine.HashTensor(arr))

    def warmup(self):
        # Optionally call HashTensor on a few dummy tensors to warm up C++ side
        t = torch.randn(8, 8, dtype=torch.float32)
        self.hash_tensor(t)


class MultivectorOps(nn.Module):
    # Clifford/geometric algebra ops are direct BitplaneEngine C++ calls, but quantization, mixed precision, and SVD remain in Python.
    def __init__(self, cmax: int = 1 << 26, cache_bytes: int = 1 << 28) -> None:
        super().__init__()
        self.cmax = cmax
        self._tensor_hasher = TensorHasher()
        self._tensor_hasher.warmup()  # Warm up hasher for fast dedup/caching
        self.basis_vectors = np.eye(256, dtype=np.complex128)
        self.cache = TensorCache(cache_bytes)
        self.quantizer = Quantizer(qmin=-8, qmax=7, init_scale=1.0, mode="real")
        self.mixed_precision_router = MixedPrecisionRouter()
        self._prefetch_queue = asyncio.Queue()
        self._recent_keys = set()
        self._last_cache_use = time.time()
        self._cache_enabled_for_medium = True
        self._cache_disable_threshold = 10.0  # seconds
        self._cache_hits_medium = 0
        self._cache_misses_medium = 0
        self._cache_window = []  # List of 1 (hit) or 0 (miss)
        self._cache_window_size = 100
        self._cache_hit_rate_threshold = 0.2  # 20%

    def _update_cache_window(self, hit):
        self._cache_window.append(1 if hit else 0)
        if len(self._cache_window) > self._cache_window_size:
            self._cache_window.pop(0)

    def _current_cache_hit_rate(self):
        if not self._cache_window:
            return 0.0
        return sum(self._cache_window) / len(self._cache_window)

    def _svd_decompose(self, tensor, decompose_thresh=1_000_000):
        if tensor.numel() > decompose_thresh and tensor.ndim >= 2:
            try:
                u, s, v = torch.linalg.svd(tensor, full_matrices=False)
                rank = min(32, s.numel())
                tensor = (u[:, :rank] @ torch.diag(s[:rank]) @ v[:rank, :]).to(tensor.dtype)
            except Exception:
                tensor = tensor.flatten()[:decompose_thresh].reshape(-1)
        return tensor

    def quantize_tensor(self, x: torch.Tensor, bits: int) -> torch.Tensor:
        return self.quantizer._qs(x)[0]

    def dequantize_tensor(self, q: torch.Tensor) -> torch.Tensor:
        return self.quantizer.dequantize(q)

    def get_precision(self, t: torch.Tensor):
        return self.mixed_precision_router.get_precision(t)

    def _clifford_dot_product(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_np = a.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()
        result = BitplaneEngine.CliffordDotProduct(a_np, b_np)
        return torch.tensor(result, dtype=torch.complex64, device=a.device)

    def _clifford_matrix_multiply(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_np = a.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()
        result = BitplaneEngine.CliffordMatrixMultiply(a_np, b_np)
        return torch.tensor(result, dtype=torch.complex64, device=a.device)

    def _clifford_matrix_vector_multiply(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_np = a.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()
        result = BitplaneEngine.CliffordMatrixVectorMultiply(a_np, b_np)
        return torch.tensor(result, dtype=torch.complex64, device=a.device)

    def _decompose_complex_to_bitplanes(self, a: torch.Tensor) -> np.ndarray:
        if not a.is_complex():
            a = torch.complex(a, torch.zeros_like(a))
        a_np = a.detach().cpu().numpy()
        return BitplaneEngine.DecomposeComplexBitplanes(a_np)

    def _recompose_complex_from_bitplanes(self, bp: np.ndarray, original_tensor: torch.Tensor) -> torch.Tensor:
        recomposed_np = BitplaneEngine.RecomposeComplexBitplanes(bp)
        return torch.from_numpy(recomposed_np).to(original_tensor.device)

    def fft(self, a: torch.Tensor) -> torch.Tensor:
        bp = self._decompose_complex_to_bitplanes(a)
        fft_bp = BitplaneEngine.BitplaneFFT(bp, inverse=False)
        return self._recompose_complex_from_bitplanes(fft_bp, a)

    def ifft(self, a: torch.Tensor) -> torch.Tensor:
        bp = self._decompose_complex_to_bitplanes(a)
        ifft_bp = BitplaneEngine.BitplaneFFT(bp, inverse=True)
        return self._recompose_complex_from_bitplanes(ifft_bp, a)

    def _fftn(self, a: torch.Tensor, dims: tuple) -> torch.Tensor:
        res = a
        for dim in dims:
            res = torch.transpose(res, dim, -1)
            res = self.fft(res)
            res = torch.transpose(res, dim, -1)
        return res

    def _ifftn(self, a: torch.Tensor, dims: tuple) -> torch.Tensor:
        res = a
        for dim in dims:
            res = torch.transpose(res, dim, -1)
            res = self.ifft(res)
            res = torch.transpose(res, dim, -1)
        return res

    def conv(self, a: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        if a.shape != k.shape:
            raise ValueError("Input and kernel tensors must have the same shape for bitplane convolution.")
        bp_a = self._decompose_complex_to_bitplanes(a)
        bp_k = self._decompose_complex_to_bitplanes(k)
        conv_bp = BitplaneEngine.BitplaneConvolution(bp_a, bp_k)
        return self._recompose_complex_from_bitplanes(conv_bp, a)

    def _convnd_fft(self, input_tensor: torch.Tensor, kernel: torch.Tensor, dims: int, padding: int):
        if not all(input_tensor.shape[i] >= kernel.shape[i] for i in range(2, dims + 2)):
            raise ValueError("Input tensor dimensions must be greater than or equal to kernel dimensions.")

        input_padded = F.pad(input_tensor, (padding,) * dims * 2)
        
        fft_shape = [
            input_padded.shape[i] + kernel.shape[i] - 1 for i in range(2, dims + 2)
        ]

        padded_input = torch.zeros(
            input_padded.shape[:2] + tuple(fft_shape),
            dtype=input_padded.dtype,
            device=input_padded.device
        )
        slicing_for_input = [slice(None)]*2 + [slice(0, s) for s in input_padded.shape[2:]]
        padded_input[slicing_for_input] = input_padded

        padded_kernel = torch.zeros(
            kernel.shape[:2] + tuple(fft_shape),
            dtype=kernel.dtype,
            device=kernel.device,
        )
        slicing_for_kernel = [slice(None)]*2 + [slice(0, s) for s in kernel.shape[2:]]
        padded_kernel[slicing_for_kernel] = kernel

        fft_dims = tuple(range(-dims, 0))
        input_fft = self._fftn(padded_input, fft_dims)
        kernel_fft = self._fftn(padded_kernel, fft_dims)

        result_fft = input_fft.unsqueeze(1) * kernel_fft.unsqueeze(0)
        result_fft = torch.sum(result_fft, dim=2)

        result_padded = self._ifftn(result_fft, fft_dims)

        output_shape = [
            input_padded.shape[i] - kernel.shape[i] + 1 for i in range(2, dims + 2)
        ]

        slicing = [slice(None)] * 2 + [slice(0, os) for os in output_shape]
        return result_padded[slicing]

    def conv2d(self, input_tensor: torch.Tensor, kernel: torch.Tensor, padding: int = 0) -> torch.Tensor:
        return self._convnd_fft(input_tensor, kernel, 2, padding)
        
    def conv3d(self, input_tensor: torch.Tensor, kernel: torch.Tensor, padding: int = 0) -> torch.Tensor:
        return self._convnd_fft(input_tensor, kernel, 3, padding)

    def matmul_geometric(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Optimized geometric product: use torch ops if possible, else fallback to C++ backend
        if a.is_cuda or b.is_cuda:
            # If on GPU, use torch.matmul directly
            return torch.matmul(a, b)
        # Try to use torch for small/medium tensors
        if a.numel() < 10000 and b.numel() < 10000:
            return torch.matmul(a, b)
        # For large tensors, use C++ backend with zero-copy if possible
        a_np = np.asarray(a.cpu()) if not isinstance(a, np.ndarray) else a
        b_np = np.asarray(b.cpu()) if not isinstance(b, np.ndarray) else b
        # TODO: Use DLPack or shared memory for true zero-copy
        result = BitplaneEngine.GeometricProduct(a_np, b_np)
        # Preallocate output and use in-place copy if possible
        out = torch.empty((a.shape[0], b.shape[1]), dtype=torch.complex64, device=a.device)
        out.copy_(torch.tensor(result, dtype=torch.complex64))
        return out

    def matmul_bitplane(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_np = a.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()
        a_bitplanes = BitplaneEngine.DecomposeBitplanes(a_np)
        b_bitplanes = BitplaneEngine.DecomposeBitplanes(b_np)
        result_bitplanes = BitplaneEngine.FastBitplaneMatmul(a_bitplanes, b_bitplanes)
        result = BitplaneEngine.RecomposeBitplanes(result_bitplanes)
        return torch.tensor(result, dtype=torch.complex64, device=a.device)

    def matmul_quantized(self, a: torch.Tensor, b: torch.Tensor, bits: int = 4) -> torch.Tensor:
        scale = (2 ** (bits - 1)) - 1
        a_quantized = torch.round(a * scale).to(torch.int32)
        b_quantized = torch.round(b * scale).to(torch.int32)
        a_np = a_quantized.detach().cpu().numpy().astype(np.int64)
        b_np = b_quantized.detach().cpu().numpy().astype(np.int64)
        a_bitplanes = BitplaneEngine.DecomposeBitplanes(a_np)
        b_bitplanes = BitplaneEngine.DecomposeBitplanes(b_np)
        result_bitplanes = BitplaneEngine.FastBitplaneIntegerMatmul(a_bitplanes, b_bitplanes)
        result = BitplaneEngine.RecomposeBitplanes(result_bitplanes)
        return torch.tensor(result, dtype=torch.int32)

    def fused_bitplane_ops(self, a: torch.Tensor, b: torch.Tensor, heuristic: str = "auto") -> torch.Tensor:
        """Fused SIMD+heuristic bitplane op using C++ backend."""
        a_np = a.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()
        a_bitplanes = BitplaneEngine.DecomposeBitplanes(a_np)
        b_bitplanes = BitplaneEngine.DecomposeBitplanes(b_np)
        result_bitplanes = BitplaneEngine.FusedBitplaneOps(a_bitplanes, b_bitplanes, heuristic)
        result = BitplaneEngine.RecomposeBitplanes(result_bitplanes)
        return torch.tensor(result, dtype=torch.complex64, device=a.device)

    def rmt_bitplane_matmul(self, a: torch.Tensor, b: torch.Tensor, heuristic: str = "auto") -> torch.Tensor:
        a_np = a.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()
        a_bitplanes = BitplaneEngine.DecomposeBitplanes(a_np)
        b_bitplanes = BitplaneEngine.DecomposeBitplanes(b_np)
        result_bitplanes = BitplaneEngine.RMTBitplaneMatmul(a_bitplanes, b_bitplanes, heuristic)
        result = BitplaneEngine.RecomposeBitplanes(result_bitplanes)
        return torch.tensor(result, dtype=torch.complex64, device=a.device)

    def _decompose_nd_to_2d_batches(self, a: torch.Tensor, b: torch.Tensor):
        """
        Given ND tensors a and b, broadcast and flatten batch dims to produce a list of (a_2d, b_2d) pairs for 2D matmul.
        Returns: list of (a_2d, b_2d), batch_shape, M, N
        """
        # Promote 1D to 2D
        if a.dim() == 1:
            a = a.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(1)
        # Broadcast batch dims
        batch_shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
        a_exp = a.expand(*batch_shape, *a.shape[-2:])
        b_exp = b.expand(*batch_shape, *b.shape[-2:])
        B = int(np.prod(batch_shape)) if batch_shape else 1
        M, K = a.shape[-2:]
        K2, N = b.shape[-2:]
        assert K == K2, f"matmul shapes do not align: {K} vs {K2}"
        a_2d = a_exp.reshape(B, M, K)
        b_2d = b_exp.reshape(B, K, N)
        return [(a_2d[i], b_2d[i]) for i in range(B)], batch_shape, M, N

    def matmul_adaptive(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Smarter, concise heuristic for dispatching
        device = a.device
        dtype = a.dtype
        # Handle ND tensors by using C++ BatchedBitplaneMatmul if possible
        if a.dim() > 2 or b.dim() > 2:
            try:
                import nanite.backend.BitplaneEngine as BitplaneEngine
                a_np = a.detach().cpu().numpy()
                b_np = b.detach().cpu().numpy()
                result_bp = BitplaneEngine.BatchedBitplaneMatmul(a_np, b_np)
                # Output is ND uint64_t array, need to recompose to complex
                from numpy import uint64
                # If result is empty, fallback
                if result_bp.size == 0:
                    raise RuntimeError("Empty result from BatchedBitplaneMatmul")
                # Recompose to complex
                result = BitplaneEngine.RecomposeBitplanes(result_bp)
                # Convert to torch tensor and move to device
                return torch.tensor(result, dtype=torch.complex64, device=device)
            except Exception as e:
                # Fallback to manual batching if C++ kernel fails
                pairs, batch_shape, M, N = self._decompose_nd_to_2d_batches(a, b)
                results = [self.matmul_bitplane(a2d, b2d) for a2d, b2d in pairs]
                out = torch.stack(results, dim=0)
                return out.reshape(*batch_shape, M, N)
        # Fallback to original logic for 2D
        numel = a.numel() * b.numel()
        is_gpu = a.is_cuda or b.is_cuda
        is_small = numel < 1_000_000
        is_medium = 1_000_000 <= numel < 10_000_000
        is_large = numel >= 10_000_000
        is_2d = a.dim() == 2 and b.dim() == 2
        is_complex = dtype.is_complex or b.dtype.is_complex
        if is_gpu or is_small:
            return torch.matmul(a, b)
        if is_2d and is_large:
            return self.rmt_bitplane_matmul(a, b, heuristic="auto")
        if is_2d and is_medium:
            return self.fused_bitplane_ops(a, b, heuristic="auto")
        if is_2d and is_complex:
            return self.matmul_geometric(a, b)
        return self.rmt_bitplane_matmul(a, b)

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.matmul_adaptive(a, b)

    async def geometric_product(self, A: torch.Tensor, B: torch.Tensor) -> np.ndarray:
        return BitplaneEngine.GeometricProduct(A, B)

    async def inner_product(self, A: torch.Tensor, B: torch.Tensor) -> np.ndarray:
        A = A.detach().cpu().numpy()
        B = B.detach().cpu().numpy()
        dim = int(np.log2(A.shape[0]))
        return BitplaneEngine.FastCliffordTransform(A, B, dim)

    async def outer_product(self, A: torch.Tensor, B: torch.Tensor) -> np.ndarray:
        A = A.detach().cpu().numpy()
        B = B.detach().cpu().numpy()
        dim = int(np.log2(A.shape[0]))
        return BitplaneEngine.FastOuterProductBitwise(A, B, dim)
    
    @staticmethod
    def clamp(x: torch.Tensor, min: float = None, max: float = None) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        if not (torch.is_floating_point(x) or torch.is_complex(x)):
            x = x.to(torch.float32)
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        real = x.real
        imag = x.imag
        if min is not None or max is not None:
            real = torch.clamp(real, min, max)
            imag = torch.clamp(imag, min, max)
        result = torch.complex(real, imag)
        # Always return a tensor of the same shape as input
        if result.shape != x.shape:
            result = result.reshape(x.shape)
        return result

    async def _prefetch_worker(self):
        while True:
            batch = []
            while not self._prefetch_queue.empty() and len(batch) < 16:
                batch.append(await self._prefetch_queue.get())
            if batch:
                for key, tensor_args in batch:
                    # Zero-copy: pass numpy arrays directly to C++
                    arr = np.asarray(tensor_args) if hasattr(tensor_args, 'numpy') else tensor_args
                    await self.cache.put(key, arr, self._tensor_hasher.hash_tensor(arr))
                self._recent_keys.add(key)
                if len(self._recent_keys) > 1000:
                    self._recent_keys.pop()
            await asyncio.sleep(0.001)

    def _dynamic_tensor_path(self, tensor):
        n = tensor.numel()
        if n < 1024:
            return 'small'
        elif n < 100000:
            return 'medium'
        else:
            return 'large'

    async def put_tensor(self, tag, tensor):
        path = self._dynamic_tensor_path(tensor)
        now = time.time()
        if path == 'small':
            # Direct, no caching/compression
            return
        elif path == 'medium':
            # Adaptive cache: enable only if hit rate is high enough
            if not self._cache_enabled_for_medium:
                if self._current_cache_hit_rate() > self._cache_hit_rate_threshold:
                    self._cache_enabled_for_medium = True
                else:
                    return
            await self.cache.put(tag, tensor, self._tensor_hasher.hash_tensor(tensor))
            self._last_cache_use = now
        else:
            # Full pipeline: decompose, hash, quantize, compress, cache
            arr = tensor.detach().cpu().numpy()
            decomp = BitplaneEngine.DecomposeBitplanes(arr)
            quant = self.quantizer._qs(tensor)[0]
            comp = BitplaneEngine.LZ4Compress(decomp.tobytes())
            await self.cache.put(tag, comp, self._tensor_hasher.hash_tensor(tensor))
            self._last_cache_use = now

    async def get_tensor(self, tag, original_shape=None):
        val = await self.cache.get(tag)
        now = time.time()
        # Only track hit/miss for medium tensors
        if val is not None:
            self._last_cache_use = now
            self._cache_enabled_for_medium = True
            self._update_cache_window(True)
        else:
            self._update_cache_window(False)
            if self._current_cache_hit_rate() < self._cache_hit_rate_threshold:
                self._cache_enabled_for_medium = False
        if isinstance(val, bytes) and original_shape is not None:
            decomp_bytes = BitplaneEngine.LZ4Decompress(val)
            arr = np.frombuffer(decomp_bytes, dtype=np.uint64).reshape(original_shape)
            return BitplaneEngine.RecomposeBitplanes(arr)
        return val

    def complex_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a complex-valued activation (ComplexPReLU) to the input tensor."""
        if not hasattr(self, '_complex_prelu'):
            # Lazy init, infer dim from last dimension
            dim = x.shape[-1] if x.ndim > 0 else 1
            self._complex_prelu = ComplexPReLU(dim)
        return self._complex_prelu(x)

    def complex_linear(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
        """Apply a complex-valued linear transformation: y = xW^T + b, supporting complex weights and bias."""
        # Ensure all are complex
        x = x if torch.is_complex(x) else torch.complex(x, torch.zeros_like(x))
        weight = weight if torch.is_complex(weight) else torch.complex(weight, torch.zeros_like(weight))
        y = torch.matmul(x, weight.transpose(-2, -1).conj())
        if bias is not None:
            bias = bias if torch.is_complex(bias) else torch.complex(bias, torch.zeros_like(bias))
            y = y + bias
        return y

    def sanitize_tensor(self, x: torch.Tensor, eps: float = 1e-8, clamp_mag: float = 1e6) -> torch.Tensor:
        """Sanitize a (possibly complex) tensor: replace NaN/Inf, clamp magnitude, avoid denormals."""
        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        real = x.real
        imag = x.imag
        # Replace NaN/Inf with eps
        real = torch.where(torch.isnan(real) | torch.isinf(real), torch.full_like(real, eps), real)
        imag = torch.where(torch.isnan(imag) | torch.isinf(imag), torch.full_like(imag, eps), imag)
        # Clamp magnitude
        mag = torch.sqrt(real ** 2 + imag ** 2)
        too_big = mag > clamp_mag
        too_small = mag < eps
        scale = torch.ones_like(mag)
        scale = torch.where(too_big, clamp_mag / (mag + eps), scale)
        scale = torch.where(too_small, eps / (mag + eps), scale)
        real = real * scale
        imag = imag * scale
        return torch.complex(real, imag)


def complex_checkpoint(function, *args):
    """Checkpointing wrapper that handles complex tensors with enhanced numerical stability."""
    # Convert complex inputs to real/imag pairs with sanitization
    real_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor) and torch.is_complex(arg):
            # Sanitize complex tensor before splitting
            real_part = torch.where(
                torch.isnan(arg.real),
                torch.tensor(1e-8, device=arg.device),
                arg.real,
            )
            imag_part = torch.where(
                torch.isnan(arg.imag),
                torch.tensor(1e-8, device=arg.device),
                arg.imag,
            )
            real_args.extend([real_part, imag_part])
        else:
            real_args.append(arg)

    # Define a wrapper that reconstructs complex tensors with stability checks
    def wrapper(*real_args_flat):
        reconstructed_args = []
        i = 0
        for arg in args:
            if isinstance(arg, torch.Tensor) and torch.is_complex(arg):
                real_part = real_args_flat[i]
                imag_part = real_args_flat[i + 1]

                # Sanitize reconstructed parts
                real_part = torch.where(
                    torch.isnan(real_part),
                    torch.tensor(1e-8, device=real_part.device),
                    real_part,
                )
                imag_part = torch.where(
                    torch.isnan(imag_part),
                    torch.tensor(1e-8, device=imag_part.device),
                    imag_part,
                )

                reconstructed_args.append(torch.complex(real_part, imag_part))
                i += 2
            else:
                reconstructed_args.append(real_args_flat[i])
                i += 1

        # Apply function and sanitize output
        result = function(*reconstructed_args)
        if isinstance(result, torch.Tensor) and torch.is_complex(result):
            result = torch.where(
                torch.isnan(result),
                torch.tensor(1e-8, device=result.device),
                result,
            )
        return result

    # Apply checkpointing to the wrapper
    return torch.utils.checkpoint.checkpoint(wrapper, *real_args, use_reentrant=False)


class ComplexPReLU(nn.Module):
    """Complex-valued PReLU activation with learnable parameters."""

    def __init__(self, config_or_dim):
        super().__init__()
        if isinstance(config_or_dim, ModelConfig):
            dim = config_or_dim.d_model
        else:
            dim = config_or_dim

        # Initialize learnable parameters for each dimension
        self.alpha = nn.Parameter(torch.ones(dim, dtype=torch.float32) * 0.25)
        self.beta = nn.Parameter(torch.ones(dim, dtype=torch.float32) * 0.25)
        self.gamma = nn.Parameter(torch.ones(dim, dtype=torch.float32) * 0.25)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Get magnitude and phase
        mag = torch.abs(z)
        phase = torch.angle(z)

        # Apply activation to magnitude
        mag_activated = torch.tanh(self.alpha * mag)

        # Apply phase shift
        phase_shifted = phase + self.beta * torch.sin(phase)

        # Combine magnitude and phase
        return mag_activated * torch.exp(1j * phase_shifted)


class ComplexLoss(nn.Module):
    """Complex-valued loss function with concept graph integration."""

    def __init__(self, concept_graph=None, alpha=0.1, beta=0.01):
        super().__init__()
        self.concept_graph = concept_graph
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-8
        print("[ComplexLoss] I got initialized successfully! :D")

    def _sanitize_tensor(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """Ensure tensor values are finite and handle numerical stability."""
        if torch.isnan(x).any():
            print(f"WARNING: NaN values detected in {name}")
            x = torch.where(torch.isnan(x), torch.tensor(self.eps, device=x.device), x)
        if torch.isinf(x).any():
            print(f"WARNING: Inf values detected in {name}")
            x = torch.where(torch.isinf(x), torch.tensor(self.eps, device=x.device), x)
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        print("[ComplexLoss] forward called")
        # Ensure inputs are complex
        pred = self._ensure_complex(pred)
        target = self._ensure_complex(target)
        print(f"[ComplexLoss] forward: pred.shape={pred.shape}, target.shape={target.shape}")
        # Sanitize inputs
        pred = self._sanitize_tensor(pred, "prediction")
        target = self._sanitize_tensor(target, "target")
        # Ensure shapes are compatible
        if pred.shape[0] != target.shape[0]:
            min_batch = min(pred.shape[0], target.shape[0])
            pred = pred[:min_batch]
            target = target[:min_batch]
        # Compute magnitude loss
        pred_mag = torch.abs(pred)
        target_mag = torch.abs(target)
        # Assert and align shapes
        assert pred_mag.shape == target_mag.shape, (
            f"Shape mismatch in loss: pred_mag {pred_mag.shape}, target_mag {target_mag.shape}")
        if pred_mag.shape != target_mag.shape:
            # Try to align shapes by slicing to the minimum shape
            min_shape = tuple(min(a, b) for a, b in zip(pred_mag.shape, target_mag.shape))
            pred_mag = pred_mag[(...,) + tuple(slice(0, s) for s in min_shape[-2:])]
            target_mag = target_mag[(...,) + tuple(slice(0, s) for s in min_shape[-2:])]
            assert pred_mag.shape == target_mag.shape, (
                f"Failed to align shapes: pred_mag {pred_mag.shape}, target_mag {target_mag.shape}")
        magnitude_loss = F.mse_loss(pred_mag, target_mag)
        # Compute phase loss
        phase_loss = self._compute_phase_loss(pred, target)
        # Compute concept loss if concept graph is available
        concept_loss = torch.tensor(0.0, device=pred.device)
        if self.concept_graph is not None:
            try:
                concept_loss = self._compute_concept_loss(pred)
            except Exception as e:
                print(f"Warning: Concept loss computation failed: {e}")
                concept_loss = torch.tensor(0.0, device=pred.device)
        # Combine losses
        total_loss = magnitude_loss + self.alpha * phase_loss + self.beta * concept_loss
        print(f"[ComplexLoss] total_loss={total_loss.item()}, magnitude_loss={magnitude_loss.item()}, phase_loss={phase_loss.item()}, concept_loss={concept_loss.item()}")
        return total_loss

    def _compute_phase_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        print("[ComplexLoss] _compute_phase_loss called")
        pred_phase = torch.angle(pred)
        target_phase = torch.angle(target)
        # Compute phase difference
        phase_diff = torch.abs(pred_phase - target_phase)
        # Normalize phase difference to [0, 1]
        phase_diff = phase_diff / (2 * math.pi)
        # Avoid mean of empty slice
        if phase_diff.numel() == 0:
            print("[ComplexLoss] _compute_phase_loss: phase_diff is empty, returning 0.0")
            return torch.tensor(0.0, device=pred.device)
        return torch.mean(phase_diff)

    def _compute_concept_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Compute concept-based loss using concept graph."""
        if self.concept_graph is None:
            print("[ComplexLoss] No concept graph provided.")
            return torch.tensor(0.0, device=pred.device)
        try:
            # Get top concepts from prediction
            pred_concepts = self.concept_graph.get_top_concepts()
            print(f"[ComplexLoss] pred.shape: {pred.shape if isinstance(pred, torch.Tensor) else type(pred)}")
            if not pred_concepts:
                print("[ComplexLoss] No top concepts found.")
                return torch.tensor(0.0, device=pred.device)
            # Compute concept alignment loss
            concept_loss = torch.tensor(0.0, device=pred.device)
            valid_count = 0
            for concept, weight in pred_concepts:
                # Get concept embedding
                concept_emb = self.concept_graph.get_concept(concept).embedding
                print(f"[ComplexLoss] Concept: {concept}, weight: {weight}")
                if not isinstance(concept_emb, torch.Tensor):
                    print(f"[ComplexLoss] Skipping {concept}: embedding is not a tensor.")
                    continue
                if pred.numel() == 0 or concept_emb.numel() == 0:
                    print(f"[ComplexLoss] Skipping {concept}: pred or concept_emb is empty. pred.numel={pred.numel()}, concept_emb.numel={concept_emb.numel()}")
                    continue
                min_len = min(pred.numel(), concept_emb.numel())
                print(f"[ComplexLoss] Concept: {concept}, concept_emb.shape: {concept_emb.shape}, min_len: {min_len}")
                if min_len == 0:
                    print(f"[ComplexLoss] Skipping {concept}: min_len is 0.")
                    continue
                similarity = torch.abs(torch.dot(pred.flatten()[:min_len], concept_emb.flatten()[:min_len]))
                print(f"[ComplexLoss] Concept: {concept}, similarity: {similarity.item()} (weight: {weight})")
                concept_loss = concept_loss + (1.0 - similarity) * float(weight)
                valid_count += 1
            # Avoid division by zero
            if valid_count == 0:
                print("[ComplexLoss] No valid concepts for loss calculation. Returning 0.0.")
                return torch.tensor(0.0, device=pred.device)
            print(f"[ComplexLoss] Final concept_loss: {concept_loss.item()}, valid_count: {valid_count}")
            return concept_loss / valid_count
        except Exception as e:
            print(f"Warning: Concept loss computation failed: {e}")
            return torch.tensor(0.0, device=pred.device)

    def _ensure_complex(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is complex."""
        if not isinstance(x, torch.Tensor) or not torch.is_complex(x):
            return torch.complex(x, torch.zeros_like(x))
        return x
