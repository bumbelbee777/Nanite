import torch
import asyncio
import hashlib
import io
import logging
from enum import Enum
from collections import deque, defaultdict, Counter
from dataclasses import dataclass
from typing import Any, Optional, Tuple, List

import lz4.frame as lz4
import torch.nn.functional as F
from torch import nn, linalg as LA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PrecisionLevel(Enum):
    TERNARY = 1
    INT4    = 2
    FP4     = 3
    FP8     = 4
    FP16    = 5

@dataclass
class PrecisionConfig:
    bits: int
    qmin: float
    qmax: float
    scale: float

PRECISION_CONFIGS = {
    PrecisionLevel.TERNARY: PrecisionConfig(bits=2,  qmin=-1,     qmax=1,       scale=0.5),
    PrecisionLevel.INT4:    PrecisionConfig(bits=4,  qmin=-8,     qmax=7,       scale=1.0),
    PrecisionLevel.FP4:     PrecisionConfig(bits=4,  qmin=-7.0,   qmax=7.0,     scale=1.0),
    PrecisionLevel.FP8:     PrecisionConfig(bits=8,  qmin=-128,   qmax=127,     scale=1.0),
    PrecisionLevel.FP16:    PrecisionConfig(bits=16, qmin=-65504, qmax=65504,   scale=1.0),
}

class MixedPrecisionRouter:
    """
    Chooses a PrecisionLevel for a tensor based on its shape, rank,
    and overall size. Heuristics here are examples — tune as needed.
    """
    def __init__(
        self,
        small_tensor_thresh: int = 1024,     # numel below → ternary
        low_rank_thresh: int = 16,           # rank below → int4
        large_tensor_thresh: int = 1_000_000 # numel above → fp8
    ):
        self.small_tensor_thresh = small_tensor_thresh
        self.low_rank_thresh   = low_rank_thresh
        self.large_tensor_thresh = large_tensor_thresh

    def get_precision(self, t: torch.Tensor) -> PrecisionLevel:
        numel = t.numel()

        # 1) Very small tensors → extreme quantization
        if numel <= self.small_tensor_thresh:
            return PrecisionLevel.TERNARY

        # 2) If it's 2D (matrix) and low rank, we can use INT4
        if t.dim() >= 2:
            # collapse to 2D for rank estimation
            mat = t.flatten(0, t.dim() - 2)  # [..., M, N] → [*, M, N]
            # pick the first slice for an estimate
            M, N = mat.shape[-2:]
            slice0 = mat.reshape(-1, M, N)[0]
            try:
                rank = int(LA.matrix_rank(slice0).item())
            except Exception:
                rank = min(M, N)
            if rank <= self.low_rank_thresh:
                return PrecisionLevel.INT4

        # 3) Very large tensors → moderate quantization
        if numel >= self.large_tensor_thresh:
            return PrecisionLevel.FP8

        # 4) Medium-sized → a bit higher fidelity
        return PrecisionLevel.FP4

class AsyncLRUTensorCache:
    def __init__(self,
                 max_size_bytes: int,
                 quant_precision: PrecisionLevel = PrecisionLevel.INT4,
                 decomp_threshold: int = 1_000_000,
                 decomp_gain_ratio: float = 0.5, **kwargs):
        self.max_size = max_size_bytes
        self.size_bytes = 0
        self.cache: dict[str, Tuple[bytes, int]] = {}
        self.order = deque()
        self.access_count = defaultdict(int)
        self.lock = asyncio.Lock()
        self.hits = self.misses = 0

        # for lookahead: track transitions between keys
        self._last_key: Optional[str] = None
        self._transitions: defaultdict[str, Counter] = defaultdict(Counter)

        # for background prefetch
        self.prefetch_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        loop.create_task(self._prefetch_worker())

        self.mpp_router = MixedPrecisionRouter(
            small_tensor_thresh=kwargs.get("small_tensor_thresh", 1024),
            low_rank_thresh=kwargs.get("low_rank_thresh", 16),
            large_tensor_thresh=kwargs.get("large_tensor_thresh", 1_000_000),
        )

        self.quant_precision = quant_precision
        self.decomp_threshold = decomp_threshold
        self.decomp_gain_ratio = decomp_gain_ratio

    def _make_key(self, tag: str, *args: Any) -> str:
        h = hashlib.sha256()
        h.update(tag.encode())
        h.update(str(args).encode())
        return h.hexdigest()

    async def _prefetch_worker(self):
        while True:
            key, tensor = await self.prefetch_queue.get()
            # use same put logic but with known key
            await self._put_serialized(key, self._compress(tensor))
            self.prefetch_queue.task_done()

    def _hash_tensor(self, t: torch.Tensor) -> str:
        # content-based hash for deduplication
        h = hashlib.sha256()
        h.update(t.detach().cpu().numpy().tobytes())
        h.update(str(t.shape).encode())
        h.update(str(t.dtype).encode())
        return h.hexdigest()

    def _quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        1) Choose a precision level via MixedPrecisionRouter
        2) Apply uniform quantization: round(tensor/scale), clamp to [qmin,qmax]
        3) Cast to the smallest integer dtype that holds it
        """
        # 1) route to level
        level = self.mpp_router.get_precision(tensor)
        cfg   = PRECISION_CONFIGS[level]

        # 2) scale, round, clamp
        q = (tensor / cfg.scale).round().clamp(cfg.qmin, cfg.qmax)

        # 3) cast to integer dtype
        #    Use torch.int8 for ≤8 bits, torch.int16 for ≤16 bits.
        if cfg.bits <= 8:
            q = q.to(torch.int8)
        else:
            q = q.to(torch.int16)

        return q

    def _compress(self, tensor: torch.Tensor) -> bytes:
        t_proc = tensor
        # optionally decompose when tensor large
        if tensor.numel() > self.decomp_threshold:
            flat = tensor.flatten(start_dim=1)
            u, s, v = LA.svd(flat)
            k = min(10, s.size(0))
            approx = (u[:, :k] @ torch.diag(s[:k]) @ v[:k, :]).reshape(tensor.shape)
            # check gain
            raw_size = self._tensor_size(tensor)
            approx_size = self._tensor_size(approx)
            if approx_size / raw_size < self.decomp_gain_ratio:
                t_proc = approx
                logger.debug(f"Applied low-rank decomposition rank={k}")
        # quantize before serialization
        t_quant = self._quantize_tensor(t_proc)
        buf = io.BytesIO()
        torch.save(t_quant.cpu(), buf)
        return lz4.compress(buf.getvalue())

    def _decompress(self, compressed: bytes) -> torch.Tensor:
        raw = lz4.decompress(compressed)
        buf = io.BytesIO(raw)
        t_quant = torch.load(buf)
        # de-quantize: cast back to float and rescale
        cfg = PRECISION_CONFIGS[self.quant_precision]
        t_float = t_quant.to(torch.float32) * cfg.scale
        return t_float

    async def get(self, tag: str, tensor_args: Tuple[Any, ...]) -> Optional[torch.Tensor]:
        key = self._make_key(tag, *tensor_args)
        async with self.lock:
            if key in self.cache:
                compressed, _ = self.cache[key]
                # update LRU/LFU
                self.order.remove(key)
                self.order.append(key)
                self.access_count[key] += 1
                self.hits += 1

                # record transition for lookahead
                if self._last_key and self._last_key != key:
                    self._transitions[self._last_key][key] += 1
                self._last_key = key

                # schedule lookahead prefetch of most likely next key
                next_counts = self._transitions[key]
                if next_counts:
                    next_key, _ = next_counts.most_common(1)[0]
                    # no await: fire-and-forget
                    self.prefetch_queue.put_nowait((next_key, torch.empty(0)))  
                    # note: caller must ensure tensor is available in some form

                return self._decompress(compressed)

            self.misses += 1
            self._last_key = key
            return None

    async def put(self, tag: str, tensor: torch.Tensor, tensor_args: Tuple[Any, ...]):
        """
        Public API: schedules tensor to be compressed & cached.
        """
        key = self._make_key(tag, *tensor_args)
        # Inline if small, else background
        if tensor.numel() < self.decomp_threshold:
            await self._put_serialized(key, self._compress(tensor))
        else:
            await self.prefetch_queue.put((key, tensor))

    async def _put_serialized(self, key: str, serialized: bytes):
        entry_size = len(serialized)
        async with self.lock:
            # evict until fits
            while self.size_bytes + entry_size > self.max_size and self.cache:
                candidates = list(self.order)[:max(1, len(self.order)//2)]
                lfu = min(candidates, key=lambda k: self.access_count[k])
                _, old_sz = self.cache.pop(lfu)
                self.order.remove(lfu)
                self.size_bytes -= old_sz
                self.access_count.pop(lfu, None)
            # insert
            self.cache[key] = (serialized, entry_size)
            self.order.append(key)
            self.access_count[key] = 1
            self.size_bytes += entry_size

    async def prefetch(self, tensor: torch.Tensor, *key_args: Any):
        """Enqueue a tensor to be cached in background."""
        key = hashlib.sha256(str(key_args).encode()).hexdigest()
        await self.prefetch_queue.put((key, tensor))

def tensor_to_multivector(t: torch.Tensor, n_dims: int) -> torch.Tensor:
    """
    Expand [N, D] tensor to [N, 2^n_dims] multivector representation.
    For n_dims=2 → 4 channels: [1, e1, e2, e12]
    Assumes input has D ≤ n_dims (e.g. x, y for 2D).
    """
    N, D = t.shape
    mv_dim = 2 ** n_dims

    out = torch.zeros(N, mv_dim, dtype=t.dtype, device=t.device)

    # Assign scalar part
    out[:, 0] = 1.0  # optional: constant scalar basis

    # Assign e1, e2, ..., eD
    for i in range(min(D, n_dims)):
        out[:, 1 << i] = t[:, i]  # e1=1, e2=2, e3=4, etc.

    return out

def multivector_to_tensor(mv: torch.Tensor) -> torch.Tensor:
    """
    Collapse multivector back to a “vector” by taking the scalar part.
    """
    return mv[..., 0]

class MultivectorOps(nn.Module):
    def __init__(self, cache_max_bytes: int = 1 << 26):
        super().__init__()
        self.cache = AsyncLRUTensorCache(cache_max_bytes)
        self._gp_table: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._gp_dim: Optional[int] = None

    def _make_key(self, tag: str, *args: Any) -> Tuple[str, Tuple[Any, ...]]:
        return tag, args

    def _ensure_gp_table(self, mv_dim: int):
        if self._gp_table is not None and self._gp_dim == mv_dim:
            return
        # Build multiplication table for basis blades (0..mv_dim-1)
        # Here we assume canonical bitwise representation of blades.
        signs = torch.zeros(mv_dim, mv_dim, dtype=torch.int8)
        idx_map = torch.zeros(mv_dim, mv_dim, dtype=torch.long)
        for i in range(mv_dim):
            for j in range(mv_dim):
                # The grade-blade product: XOR of bitmasks
                out_idx = i ^ j
                # sign: determined by how many swaps needed
                # count bitwise overlap parity:
                common = i & j
                # number of 1-bits in common*(common-1)/2 gives sign
                swaps = (common.bit_count() * (common.bit_count() - 1) // 2) & 1
                s = -1 if swaps else 1
                idx_map[i, j] = out_idx
                signs[i, j] = s
        self._gp_table = (signs, idx_map)
        self._gp_dim = mv_dim

    async def geometric_product(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # A, B: [..., mv_dim]
        mv_dim = A.shape[-1]
        self._ensure_gp_table(mv_dim)
        signs, idx_map = self._gp_table
        # Compute out[..., k] = sum_{i,j | idx_map[i,j]=k} signs[i,j]*A[...,i]*B[...,j]
        out = torch.zeros_like(A)
        for i in range(mv_dim):
            for j in range(mv_dim):
                out[..., idx_map[i,j]] += signs[i,j] * A[..., i] * B[..., j]
        return out

    async def dot(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # inner product over the mv channels
        return torch.sum(A.conj() * B, dim=-1)

    # Helper to decompose large tensors
    def _decompose(self, t: torch.Tensor) -> torch.Tensor:
        if t.numel() > self.cache.decomp_threshold:
            flat = t.flatten(start_dim=1)
            u, s, v = LA.svd(flat)
            k = min(10, s.size(0))
            return (u[:, :k] @ torch.diag(s[:k]) @ v[:k, :]).reshape(t.shape)
        return t

    async def _generic_op(self,
                          tag: str,
                          args: Tuple[Any, ...],
                          compute_fn: callable) -> torch.Tensor:
        out = await self.cache.get(tag, args)
        if out is None:
            raw = compute_fn()
            processed = self._decompose(raw)
            await self.cache.put(tag, processed, args)
            out = processed
        return out

    async def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        tag, args = self._make_key("mv_matmul", a.shape, b.shape, str(a.dtype))
        return await self._generic_op(tag, args, lambda: torch.matmul(a, b))

    async def conv1d(self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor]=None) -> torch.Tensor:
        tag, args = self._make_key("mv_conv1d", x.shape, w.shape, None if b is None else b.shape, str(x.dtype))
        return await self._generic_op(tag, args, lambda: F.conv1d(x, w, b))

    async def conv2d(self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor]=None) -> torch.Tensor:
        tag, args = self._make_key("mv_conv2d", x.shape, w.shape, None if b is None else b.shape, str(x.dtype))
        return await self._generic_op(tag, args, lambda: F.conv2d(x, w, b))

    async def conv3d(self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor]=None) -> torch.Tensor:
        tag, args = self._make_key("mv_conv3d", x.shape, w.shape, None if b is None else b.shape, str(x.dtype))
        return await self._generic_op(tag, args, lambda: F.conv3d(x, w, b))

    async def batch_norm(self, x: torch.Tensor, w: Optional[torch.Tensor]=None, b: Optional[torch.Tensor]=None) -> torch.Tensor:
        tag, args = self._make_key("mv_batch_norm", x.shape, None if w is None else w.shape, None if b is None else b.shape, str(x.dtype))
        return await self._generic_op(tag, args, lambda: F.batch_norm(x, w, b))

    async def fft(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        tag, args = self._make_key("mv_fft", x.shape, str(x.dtype), dim)
        return await self._generic_op(tag, args, lambda: torch.fft.fft(x, dim=dim))

    async def ifft(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        tag, args = self._make_key("mv_ifft", x.shape, str(x.dtype), dim)
        return await self._generic_op(tag, args, lambda: torch.fft.ifft(x, dim=dim))
