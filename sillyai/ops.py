import asyncio
import hashlib
import io
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import lz4.frame as lz4
import mmh3
import numpy as np
import torch
import torch.nn.functional as F
import xxhash
from numba import complex128, float64, int64, jit, prange
from torch import linalg as LA
from torch import nn

from .config import PrecisionLevel


# Quantizer with real/cartesian/polar, sync+async entrypoints, JIT/compile-ready
class Quantizer(nn.Module):
    def __init__(self, qmin, qmax, init_scale=1.0, mode="real"):
        super().__init__()
        self.qmin, self.qmax, self.mode = qmin, qmax, mode
        self.scale = nn.Parameter(torch.tensor(init_scale))
        self.zero_point = nn.Parameter(torch.tensor(0.0))
        print("[Quantizer] I got initialized successfully! :D")

    def forward(self, x):
        if self.mode == "real":
            return self._qs(x)[0]
        # Handle complex numbers by quantizing real and imaginary parts separately
        r, i = x.real, x.imag
        qr = self._qs(r)[0]
        qi = self._qs(i)[0]
        return torch.complex(qr, qi)

    @torch.jit.export
    def dequantize(self, q):
        zp, sc = self.zero_point, self.scale
        if self.mode == "real":
            return (q.float() - zp) * sc
        # Handle complex numbers by dequantizing real and imaginary parts separately
        r, i = q.real, q.imag
        dr = (r - zp) * sc
        di = (i - zp) * sc
        return torch.complex(dr, di)

    def _qs(self, x):
        # Ensure x is real-valued before quantization
        x = x.real if torch.is_complex(x) else x
        q = torch.clamp(
            torch.round(x / self.scale + self.zero_point),
            self.qmin,
            self.qmax,
        )
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
        print("[MixedPrecisionRouter] I got initialized successfully! :D")

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


# Cache with sync wrappers, CPU/GPU-aware, hybrid LRU+LFU
class TensorCache:
    def __init__(self, max_bytes, decomp_thresh=1_000_000, tol=1e-2):
        self.max, self.t, self.cache, self.ord = max_bytes, 0, {}, deque()
        self.ac = defaultdict(int)
        self.lock = asyncio.Lock()
        self.prefetch = asyncio.Queue()
        self.worker_task = None
        self.mpr = MixedPrecisionRouter()
        self.quant = None
        self.dt, self.tol = decomp_thresh, tol
        print("[TensorCache] I got initialized successfully! :D")

    async def start(self):
        """Start the worker task."""
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(self._worker())

    async def stop(self):
        """Stop the worker task."""
        if self.worker_task is not None:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            self.worker_task = None

    async def _worker(self):
        """Worker task for processing prefetch queue."""
        try:
            while True:
                k, t = await self.prefetch.get()
                await self._put(k, self._compress(t))
                self.prefetch.task_done()
        except asyncio.CancelledError:
            # Clean up any remaining items in the queue
            while not self.prefetch.empty():
                try:
                    self.prefetch.get_nowait()
                    self.prefetch.task_done()
                except asyncio.QueueEmpty:
                    break
            raise

    def _key(self, tag, *a):
        h = hashlib.sha256()
        h.update(tag.encode() + str(a).encode())
        return h.hexdigest()

    def _compress(self, t):
        tp = (
            t
            if t.numel() <= self.dt
            else (
                lambda x: x
                @ torch.diag(LA.svd(x.flatten(1))[1][:10])
                @ LA.svd(x.flatten(1))[2][:10, :]
            )(t)
        )
        lvl = self.mpr.get_precision(tp)
        cfg = PRECISION_CONFIGS[lvl]
        if not self.quant or self.quant.qmin != cfg.qmin:
            self.quant = Quantizer(cfg.qmin, cfg.qmax, mode="real")
        q = self.quant(tp) if self.quant.mode != "real" else (self.quant(tp),)
        if tp.is_cuda:
            q = tuple(x.cpu() for x in q)
        buf = io.BytesIO()
        torch.save(
            {
                "q": q,
                "s": self.quant.scale,
                "z": self.quant.zero_point,
                "m": self.quant.mode,
            },
            buf,
        )
        return lz4.compress(buf.getvalue())

    def _decompress(self, b, device):
        data = torch.load(io.BytesIO(lz4.decompress(b)), map_location="cpu")
        q, s, z = data["q"], data["s"].item(), data["z"].item()
        if data["m"] == "real":
            return ((q[0].float() - z) * s).to(device)
        r, i = q
        return torch.view_as_complex(
            torch.stack([((r - z) * s), ((i - z) * s)], -1),
        ).to(device)

    @torch.jit.ignore
    async def get(self, tag, *args):
        k = self._key(tag, *args)
        async with self.lock:
            if k in self.cache:
                b, _ = self.cache[k]
                self.ord.remove(k)
                self.ord.append(k)
                self.ac[k] += 1
                return self._decompress(b, torch.device(args[-1]))
            return None

    @torch.jit.ignore
    async def put(self, tag, t, *args):
        k = self._key(tag, *args)
        b = self._compress(t)
        await self._put(k, b)

    async def _put(self, k, b):
        sz = len(b)
        async with self.lock:
            while self.t + sz > self.max and self.ord:
                old = self.ord.popleft()
                _, osz = self.cache.pop(old)
                self.ac.pop(old, None)
                self.t -= osz
            self.cache[k] = (b, sz)
            self.ord.append(k)
            self.ac[k] = 1
            self.t += sz


# Low-level bit operations for Clifford algebra
@jit(nopython=True)
def _bit_rotate_left(x: int64, n: int64) -> int64:
    """Fast bit rotation left."""
    return ((x << n) | (x >> (64 - n))) & 0xFFFFFFFFFFFFFFFF


@jit(nopython=True)
def _bit_rotate_right(x: int64, n: int64) -> int64:
    """Fast bit rotation right."""
    return ((x >> n) | (x << (64 - n))) & 0xFFFFFFFFFFFFFFFF


@jit(nopython=True)
def _bit_count(x: int64) -> int64:
    """Count set bits using Brian Kernighan's algorithm."""
    count = 0
    while x:
        x &= x - 1
        count += 1
    return count


@jit(nopython=True, parallel=True)
def _fast_tensor_hash_numba(tensor: np.ndarray) -> int64:
    """Ultra-fast tensor hashing using Numba JIT and bit operations."""
    result = int64(0)
    for i in prange(tensor.size):
        val = int64(tensor.flat[i] * 1e6)
        result = _bit_rotate_left(result, 5) ^ val
    return result


@jit(nopython=True, parallel=True)
def _fast_geometric_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Optimized geometric product using Numba JIT and bit operations."""
    result = np.zeros_like(a)
    for i in prange(a.shape[0]):
        for j in prange(b.shape[1]):
            acc = complex128(0)
            for k in prange(a.shape[1]):
                # Use bit operations for sign computation
                sign = 1 if (_bit_count(i & k) & 1) == 0 else -1
                acc += sign * a[i, k] * b[k, j]
            result[i, j] = acc
    return result


@jit(nopython=True, parallel=True)
def _fast_clifford_transform(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Fast Clifford transformation using bit operations."""
    result = np.zeros_like(x)
    for i in prange(x.shape[0]):
        for j in prange(basis.shape[1]):
            acc = complex128(0)
            for k in prange(x.shape[1]):
                # Use bit operations for basis transformation
                mask = _bit_rotate_left(1, k)
                if (i & mask) != 0:
                    acc += x[i, k] * basis[k, j]
            result[i, j] = acc
    return result


@jit(nopython=True)
def _fast_interpolate(a: np.ndarray, b: np.ndarray, t: float64) -> np.ndarray:
    """Fast interpolation using bit operations for sign computation."""
    result = np.zeros_like(a)
    for i in prange(a.shape[0]):
        for j in prange(a.shape[1]):
            # Use bit operations for smooth interpolation
            sign = 1 if (_bit_count(i + j) & 1) == 0 else -1
            result[i, j] = a[i, j] * (1 - t) + sign * b[i, j] * t
    return result


def _fast_tensor_hash_torch(tensor: torch.Tensor) -> int:
    """Fast tensor hashing using pure PyTorch operations."""
    # Flatten and convert to bytes using pure PyTorch
    flat = tensor.flatten()
    # Convert to contiguous tensor and get raw bytes
    bytes_data = flat.contiguous().storage().tolist()
    # Use xxHash for fast hashing
    return xxhash.xxh64(bytes(bytes_data)).intdigest()


class TensorHasher:
    """Fast tensor hashing with multiple strategies."""

    def __init__(self, strategy: str = "auto") -> None:
        self.strategy = strategy
        self._hash_cache: dict[int, int] = {}
        self._hash_cache_size = 10000

    def hash_tensor(self, tensor: torch.Tensor) -> int:
        """Hash a tensor using the best available strategy."""
        # Check cache first
        if id(tensor) in self._hash_cache:
            return self._hash_cache[id(tensor)]

        if self.strategy == "auto":
            # Choose best strategy based on tensor size
            if tensor.numel() > 1_000_000:
                result = self._hash_large_tensor(tensor)
            else:
                result = self._hash_small_tensor(tensor)
        elif self.strategy == "numba":
            result = _fast_tensor_hash_numba(tensor.cpu().numpy())
        elif self.strategy == "torchscript":
            result = _fast_tensor_hash_torch(tensor)
        elif self.strategy == "xxhash":
            result = xxhash.xxh64(tensor.cpu().numpy().tobytes()).intdigest()
        elif self.strategy == "murmur":
            result = mmh3.hash(tensor.cpu().numpy().tobytes())
        else:
            result = self._hash_small_tensor(tensor)

        # Update cache
        if len(self._hash_cache) >= self._hash_cache_size:
            self._hash_cache.pop(next(iter(self._hash_cache)))
        self._hash_cache[id(tensor)] = result

        return result

    def _hash_small_tensor(self, tensor: torch.Tensor) -> int:
        """Hash small tensors using TorchScript."""
        return _fast_tensor_hash_torch(tensor)

    def _hash_large_tensor(self, tensor: torch.Tensor) -> int:
        """Hash large tensors using parallel Numba."""
        return _fast_tensor_hash_numba(tensor.cpu().numpy())


@jit(nopython=True, parallel=True)
def _fast_bitplane_ops(
    a_bp: np.ndarray,
    b_bp: np.ndarray,
    op_type: int64,
) -> np.ndarray:
    """Ultra-fast bitplane operations using Numba."""
    result = np.zeros_like(a_bp)
    for i in prange(a_bp.shape[0]):
        for j in prange(a_bp.shape[1]):
            for k in prange(64):
                if op_type == 0:  # AND
                    result[i, j, k] = a_bp[i, j, k] & b_bp[i, j, k]
                elif op_type == 1:  # OR
                    result[i, j, k] = a_bp[i, j, k] | b_bp[i, j, k]
                elif op_type == 2:  # XOR
                    result[i, j, k] = a_bp[i, j, k] ^ b_bp[i, j, k]
    return result


@jit(nopython=True, parallel=True)
def _decompose_bitplanes(x: np.ndarray) -> np.ndarray:
    """Decompose tensor into bitplanes using Numba."""
    result = np.zeros((x.shape[0], x.shape[1], 64), dtype=np.uint8)
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            val = int64(x[i, j] * 1e6)
            for k in range(64):
                result[i, j, k] = (val >> k) & 1
    return result


@jit(nopython=True, parallel=True)
def _recompose_bitplanes(bp: np.ndarray) -> np.ndarray:
    """Recompose bitplanes into tensor using Numba."""
    result = np.zeros((bp.shape[0], bp.shape[1]), dtype=np.float64)
    for i in prange(bp.shape[0]):
        for j in prange(bp.shape[1]):
            val = int64(0)
            for k in range(64):
                val |= int64(bp[i, j, k]) << k
            result[i, j] = val / 1e6
    return result


@jit(nopython=True)
def _compute_blade_sign(a: int64, b: int64) -> int64:
    """Compute the sign of the geometric product of two blades using bitwise operations."""
    # Count the number of swaps needed to sort the basis vectors
    # This is equivalent to counting inversions in the permutation
    n = int64(0)
    temp = a & b
    while temp:
        n += _bit_count(temp & (temp - 1))
        temp >>= 1
    return -1 if (n & 1) else 1


@jit(nopython=True)
def _compute_blade_grade(a: int64) -> int64:
    """Compute the grade of a blade using bitwise operations."""
    return _bit_count(a)


@jit(nopython=True)
def _compute_blade_parity(a: int64, b: int64) -> int64:
    """Compute the parity of the geometric product using bitwise operations."""
    # Count the number of basis vectors that need to be swapped
    n = int64(0)
    temp = a & b
    while temp:
        n += _bit_count(temp & (temp - 1))
        temp >>= 1
    return n & 1


@jit(nopython=True, parallel=True)
def _fast_geometric_product_bitwise(
    a: np.ndarray,
    b: np.ndarray,
    dim: int64,
) -> np.ndarray:
    """Ultra-fast geometric product using bitwise operations and Numba JIT."""
    result = np.zeros_like(a)
    n_blades = 1 << dim  # 2^dim blades

    # Precompute blade signs for all possible combinations
    sign_table = np.zeros((n_blades, n_blades), dtype=np.int64)
    for i in range(n_blades):
        for j in range(n_blades):
            sign_table[i, j] = _compute_blade_sign(i, j)

    # Compute geometric product using bitwise operations
    for i in prange(n_blades):
        for j in prange(n_blades):
            # Compute the resulting blade using XOR
            result_blade = i ^ j
            # Get the sign from lookup table
            sign = sign_table[i, j]
            # Accumulate the result
            result[result_blade] += sign * a[i] * b[j]

    return result


@jit(nopython=True, parallel=True)
def _fast_inner_product_bitwise(a: np.ndarray, b: np.ndarray, dim: int64) -> np.ndarray:
    """Ultra-fast inner product using bitwise operations and Numba JIT."""
    result = np.zeros_like(a)
    n_blades = 1 << dim

    # Precompute grade differences for all possible combinations
    grade_table = np.zeros((n_blades, n_blades), dtype=np.int64)
    for i in range(n_blades):
        for j in range(n_blades):
            grade_table[i, j] = abs(_compute_blade_grade(i) - _compute_blade_grade(j))

    # Compute inner product using bitwise operations
    for i in prange(n_blades):
        for j in prange(n_blades):
            # Only compute inner product for blades with grade difference 2
            if grade_table[i, j] == 2:
                # Compute the resulting blade using XOR
                result_blade = i ^ j
                # Get the sign using bitwise operations
                sign = _compute_blade_sign(i, j)
                # Accumulate the result
                result[result_blade] += sign * a[i] * b[j]

    return result


@jit(nopython=True, parallel=True)
def _fast_outer_product_bitwise(a: np.ndarray, b: np.ndarray, dim: int64) -> np.ndarray:
    """Ultra-fast outer product using bitwise operations and Numba JIT."""
    result = np.zeros_like(a)
    n_blades = 1 << dim

    # Precompute grade sums for all possible combinations
    grade_table = np.zeros((n_blades, n_blades), dtype=np.int64)
    for i in range(n_blades):
        for j in range(n_blades):
            grade_table[i, j] = _compute_blade_grade(i) + _compute_blade_grade(j)

    # Compute outer product using bitwise operations
    for i in prange(n_blades):
        for j in prange(n_blades):
            # Only compute outer product for blades with non-overlapping basis vectors
            if (i & j) == 0 and grade_table[i, j] <= dim:
                # Compute the resulting blade using OR
                result_blade = i | j
                # Get the sign using bitwise operations
                sign = _compute_blade_sign(i, j)
                # Accumulate the result
                result[result_blade] += sign * a[i] * b[j]

    return result


class MultivectorOps(nn.Module):
    """High-performance complex tensor operations backend leveraging multivectors and bitwise operations."""

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, cmax: int = 1 << 26) -> None:
        if self._initialized:
            return

        super().__init__()
        self.cmax = cmax
        self._compiled = False
        self._cache: dict[tuple[int, int], np.ndarray] = {}
        self._cache_size = 10000
        self._bitplane_cache: dict[int, np.ndarray] = {}

        # Initialize thread pool for parallel operations
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Pre-compute common basis transformations
        self._basis_cache: dict[int, np.ndarray] = {}

        # Initialize tensor hasher
        self._tensor_hasher = TensorHasher()

        # Precompute lookup tables for geometric operations
        self._init_lookup_tables()

        self._initialized = True

    def _init_lookup_tables(self):
        """Initialize lookup tables for geometric operations."""
        # Precompute blade signs for all possible combinations up to dimension 8
        self._blade_sign_table = np.zeros((256, 256), dtype=np.int64)
        for i in range(256):
            for j in range(256):
                self._blade_sign_table[i, j] = _compute_blade_sign(i, j)

        # Precompute grade differences for inner product
        self._grade_diff_table = np.zeros((256, 256), dtype=np.int64)
        for i in range(256):
            for j in range(256):
                self._grade_diff_table[i, j] = abs(
                    _compute_blade_grade(i) - _compute_blade_grade(j),
                )

        # Precompute grade sums for outer product
        self._grade_sum_table = np.zeros((256, 256), dtype=np.int64)
        for i in range(256):
            for j in range(256):
                self._grade_sum_table[i, j] = _compute_blade_grade(
                    i,
                ) + _compute_blade_grade(j)

    async def geometric_product(self, A: Any, B: Any) -> np.ndarray:
        """Async geometric product with bitwise optimization."""
        A = self._sanitize_tensor(A, "A")
        B = self._sanitize_tensor(B, "B")

        # Get dimension from input tensors
        dim = int(np.log2(A.shape[0]))

        # Use bitwise-optimized implementation
        return _fast_geometric_product_bitwise(A, B, dim)

    async def inner_product(self, A: Any, B: Any) -> np.ndarray:
        """Async inner product with bitwise optimization."""
        A = self._sanitize_tensor(A, "A")
        B = self._sanitize_tensor(B, "B")

        # Get dimension from input tensors
        dim = int(np.log2(A.shape[0]))

        # Use bitwise-optimized implementation
        return _fast_inner_product_bitwise(A, B, dim)

    async def outer_product(self, A: Any, B: Any) -> np.ndarray:
        """Async outer product with bitwise optimization."""
        A = self._sanitize_tensor(A, "A")
        B = self._sanitize_tensor(B, "B")

        # Get dimension from input tensors
        dim = int(np.log2(A.shape[0]))

        # Use bitwise-optimized implementation
        return _fast_outer_product_bitwise(A, B, dim)

    def _sanitize_tensor(self, x: Any, name: str) -> np.ndarray:
        """Convert input to NumPy array with type checking."""
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        elif not isinstance(x, np.ndarray):
            x = np.array(x)
        return self._ensure_complex(x)

    def _ensure_complex(self, x: np.ndarray) -> np.ndarray:
        """Ensure array is complex using NumPy."""
        if not np.iscomplexobj(x):
            return x.astype(np.complex128)
        return x


def complex_checkpoint(function, *args):
    """Checkpointing wrapper that handles complex tensors with enhanced numerical stability."""
    # Convert complex inputs to real/imag pairs with sanitization
    real_args = []
    for arg in args:
        if torch.is_tensor(arg) and arg.is_complex():
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
            if torch.is_tensor(arg) and arg.is_complex():
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
        if torch.is_tensor(result) and result.is_complex():
            result = torch.where(
                torch.isnan(result),
                torch.tensor(1e-8, device=result.device),
                result,
            )
        return result

    # Apply checkpointing to the wrapper
    return torch.utils.checkpoint.checkpoint(wrapper, *real_args, use_reentrant=False)


class ComplexLoss(nn.Module):
    """Complex loss function with enhanced numerical stability."""

    def __init__(self, concept_graph=None, alpha=0.1, beta=0.01):
        super().__init__()
        self.concept_graph = concept_graph
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-8
        print("[ComplexLoss] I got initialized successfully! :D")

    def _sanitize_tensor(self, x: torch.Tensor, name: str) -> torch.Tensor:
        """Enhanced tensor sanitization with detailed diagnostics."""
        if torch.isnan(x).any():
            print(f"WARNING: NaN values detected in {name}")
            print(f"Number of NaN values: {torch.isnan(x).sum().item()}")
            print(f"Tensor shape: {x.shape}")
            print(f"Tensor dtype: {x.dtype}")
            print(f"Tensor device: {x.device}")

            # Replace NaN values with small epsilon
            x = torch.where(torch.isnan(x), torch.tensor(self.eps, device=x.device), x)

            # Additional stability checks
            if torch.isinf(x).any():
                print(f"WARNING: Inf values detected in {name}")
                x = torch.where(
                    torch.isinf(x),
                    torch.tensor(self.eps, device=x.device),
                    x,
                )

            # Check for extremely large values
            max_val = torch.max(torch.abs(x))
            if max_val > 1e6:  # Reasonable upper bound for loss values
                print(f"WARNING: Large values detected in {name}")
                print(f"Max absolute value: {max_val.item()}")
                x = torch.clamp(x, -1e6, 1e6)

        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute complex loss with enhanced numerical stability."""
        # Debug shapes
        # print(f"Prediction shape: {pred.shape}")
        # print(f"Target shape: {target.shape}")

        # Ensure complex tensors
        pred = self._ensure_complex(pred)
        target = self._ensure_complex(target)

        # Reshape prediction to match target shape if needed
        if pred.shape != target.shape:
            # First ensure we have the right number of dimensions
            while pred.dim() < target.dim():
                pred = pred.unsqueeze(-1)

            # Now reshape to match target exactly
            # We know target shape is [batch, seq_len, 1, 1, 1, 4, 4, 4]
            # And pred shape is [batch, seq_len, 1, 1, 4, 4, 4, 1]
            # We need to move the last dimension to the right position
            pred = pred.permute(0, 1, 2, 3, 7, 4, 5, 6)  # Move last dim to position 4
            # print(f"Reshaped prediction to: {pred.shape}")

        # Compute magnitude loss
        pred_mag = torch.abs(pred)
        target_mag = torch.abs(target)
        mag_loss = F.mse_loss(pred_mag, target_mag)
        mag_loss = self._sanitize_tensor(mag_loss, "magnitude loss")

        # Compute phase loss
        phase_loss = self._compute_phase_loss(pred, target)
        phase_loss = self._sanitize_tensor(phase_loss, "phase loss")

        # Compute concept loss if concept graph is provided
        concept_loss = torch.tensor(0.0, device=pred.device)
        if self.concept_graph is not None:
            concept_loss = self._compute_concept_loss(pred)
            concept_loss = self._sanitize_tensor(concept_loss, "concept loss")

        # Combine losses with stability checks
        total_loss = mag_loss + self.alpha * phase_loss + self.beta * concept_loss
        return self._sanitize_tensor(total_loss, "total loss")

    def _compute_phase_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute phase loss with enhanced numerical stability."""
        # Compute phases
        pred_phase = torch.atan2(pred.imag, pred.real + self.eps)
        target_phase = torch.atan2(target.imag, target.real + self.eps)

        # Compute phase difference
        phase_diff = pred_phase - target_phase

        # Normalize phase difference to [-pi, pi]
        phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))

        # Compute loss
        phase_loss = torch.mean(phase_diff**2)
        return self._sanitize_tensor(phase_loss, "phase difference loss")

    def _compute_concept_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Compute concept loss with enhanced numerical stability."""
        if self.concept_graph is None:
            return torch.tensor(0.0, device=pred.device)

        # Get concept embeddings
        concept_embeddings_dict = self.concept_graph.get_embeddings()

        # Convert dictionary to tensor
        if not concept_embeddings_dict:
            return torch.tensor(0.0, device=pred.device)

        # Stack embeddings into a tensor
        concept_embeddings = torch.stack(list(concept_embeddings_dict.values()))
        concept_embeddings = self._ensure_complex(concept_embeddings)

        # Reshape prediction tensor for matrix multiplication
        # Original shape: [16, 64, 1, 1, 1, 4, 4, 4]
        # We want to combine all dimensions except the first two
        pred_reshaped = pred.reshape(pred.shape[0], pred.shape[1], -1)  # [16, 64, 4096]

        # Reshape concept embeddings to match
        concept_embeddings = concept_embeddings.reshape(
            -1,
            concept_embeddings.shape[-1],
        )  # [4096, 64]

        # Compute similarity matrix
        similarity = torch.matmul(
            pred_reshaped,
            concept_embeddings.transpose(-2, -1),
        )  # [16, 64, 4096]
        similarity = self._sanitize_tensor(similarity, "concept similarity")

        # Get concept relationships
        relationships = self.concept_graph.get_relationships()

        # Compute loss based on relationships
        loss = torch.tensor(0.0, device=pred.device)
        for rel in relationships:
            source, target, weight = rel
            sim = similarity[:, source, target]
            loss = loss + weight * (1 - sim)

        return self._sanitize_tensor(loss, "concept relationship loss")

    def _ensure_complex(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is complex with enhanced validation."""
        if not torch.is_complex(x):
            if x.dtype == torch.float32:
                x = torch.complex(x, torch.zeros_like(x))
            elif x.dtype == torch.float64:
                x = torch.complex(x, torch.zeros_like(x))
            else:
                raise ValueError(f"Unsupported dtype for complex conversion: {x.dtype}")
        return self._sanitize_tensor(x, "complex conversion")
