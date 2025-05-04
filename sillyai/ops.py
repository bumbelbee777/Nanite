import torch
import asyncio
import hashlib
import io
import logging
from enum import Enum
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Tuple, List

import lz4.frame as lz4
from torch import nn
from torch import linalg as LA
import torch.nn.functional as F

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
    PrecisionLevel.TERNARY: PrecisionConfig(bits=2, qmin=-1, qmax=1, scale=0.5),
    PrecisionLevel.INT4:    PrecisionConfig(bits=4, qmin=-8, qmax=7, scale=1.0),
    PrecisionLevel.FP4:     PrecisionConfig(bits=4, qmin=-7.0, qmax=7.0, scale=1.0),
    PrecisionLevel.FP8:     PrecisionConfig(bits=8, qmin=-128, qmax=127, scale=1.0),
    PrecisionLevel.FP16:    PrecisionConfig(bits=16, qmin=-65504, qmax=65504, scale=1.0),
}

# ------------------------------
# Async LRU Cache with Advanced Eviction & Quantization
# ------------------------------
class AsyncLRUTensorCache:
    def __init__(self,
                 max_size_bytes: int,
                 quant_precision: PrecisionLevel = PrecisionLevel.INT4,
                 decomp_threshold: int = 1_000_000,
                 decomp_gain_ratio: float = 0.5):
        self.max_size = max_size_bytes
        self.size_bytes = 0
        self.cache: dict[str, Tuple[bytes, int]] = {}
        self.order = deque()
        self.access_count = defaultdict(int)
        self.lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0
        # New parameters
        self.quant_precision = quant_precision
        self.decomp_threshold = decomp_threshold
        self.decomp_gain_ratio = decomp_gain_ratio

    def _tensor_size(self, t: torch.Tensor) -> int:
        return t.element_size() * t.nelement()

    def _quantize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        # apply uniform quantization according to chosen precision
        cfg = PRECISION_CONFIGS[self.quant_precision]
        # scale and cast
        t_scaled = tensor / cfg.scale
        if self.quant_precision == PrecisionLevel.INT4:
            return t_scaled.to(torch.int8)
        else:
            return t_scaled.to(torch.int32)

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

    async def get(self, *key_args: Any) -> Optional[torch.Tensor]:
        key = hashlib.sha256(str(key_args).encode()).hexdigest()
        async with self.lock:
            if key in self.cache:
                compressed, _ = self.cache[key]
                self.order.remove(key)
                self.order.append(key)
                self.access_count[key] += 1
                self.hits += 1
                logger.debug(f"Cache hit for {key}")
                return self._decompress(compressed)
            self.misses += 1
            logger.debug(f"Cache miss for {key}")
            return None

    async def put(self, tensor: torch.Tensor, *key_args: Any):
        serialized = self._compress(tensor)
        entry_size = len(serialized)
        key = hashlib.sha256(str(key_args).encode()).hexdigest()
        async with self.lock:
            while self.size_bytes + entry_size > self.max_size and self.cache:
                candidates = list(self.order)[:max(1, len(self.order)//2)]
                lfu_key = min(candidates, key=lambda k: self.access_count[k])
                _, old_sz = self.cache.pop(lfu_key)
                self.order.remove(lfu_key)
                self.size_bytes -= old_sz
                self.access_count.pop(lfu_key, None)
                logger.debug(f"Evicted {lfu_key} via hybrid LRU+LFU")
            self.cache[key] = (serialized, entry_size)
            self.order.append(key)
            self.access_count[key] = 1
            self.size_bytes += entry_size

    def adjust_cache_size(self):
        total = self.hits + self.misses
        if total == 0:
            return
        hit_rate = self.hits / total
        if hit_rate < 0.3:
            self.max_size = max(self.max_size // 2, 1 << 20)
        elif hit_rate > 0.7:
            self.max_size = min(self.max_size * 2, 1 << 30)  # Cap to a max size
        logger.info(f"Cache resized to {self.max_size} bytes (hit rate {hit_rate:.2f})")

    def save_to_disk(self, path: str):
        with open(path, 'wb') as f:
            torch.save({'cache': self.cache, 'order': list(self.order)}, f)
        logger.info(f"Cache serialized to {path}")

    def load_from_disk(self, path: str):
        data = torch.load(path)
        self.cache = data['cache']
        self.order = deque(data['order'])
        self.size_bytes = sum(sz for _, sz in self.cache.values())
        logger.info(f"Cache loaded from {path}")

# ------------------------------
# Optimized Ops with Per-Op Decomposition
# ------------------------------
class OptimizedComplexOps(nn.Module):
    def __init__(self, cache_max_bytes: int = 1 << 26):
        super().__init__()
        self.cache = AsyncLRUTensorCache(cache_max_bytes)
        self.batch_queue: List[Tuple[Any, ...]] = []

    async def _flush_batch(self):
        if len(self.batch_queue) >= 256:  # Or time-based flush
            tasks = [self.cache.put(*args) for args in self.batch_queue]
            await asyncio.gather(*tasks)
            self.batch_queue.clear()

    def _decompose_matmul(self, out: torch.Tensor) -> torch.Tensor:
        # apply only if large out tensor
        if out.numel() > self.cache.decomp_threshold:
            u, s, v = LA.svd(out)
            k = min(out.size(0)//2, out.size(1)//2, 20)
            return (u[:, :k] @ torch.diag(s[:k]) @ v[:k, :])
        return out

    def _decompose_conv(self, out: torch.Tensor) -> torch.Tensor:
        if out.numel() > self.cache.decomp_threshold:
            B, C, *rest = out.shape
            flat = out.view(B * C, -1)
            u, s, v = LA.svd(flat)
            k = min(10, s.size(0))
            return (u[:, :k] @ torch.diag(s[:k]) @ v[:k, :]).view(out.shape)
        return out

    async def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.dim() == b.dim() == 2, "matmul expects 2D tensors"
        key = ("matmul", a.shape, b.shape, str(a.dtype), str(b.dtype))
        out = await self.cache.get(*key)
        if out is None:
            raw = torch.matmul(a, b)
            comp = self._decompose_matmul(raw)
            self.batch_queue.append((comp, *key))
            out = comp
        await self._flush_batch()
        return out

    async def conv1d(self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert x.dim() == 3 and w.dim() == 3, "conv1d expects 3D inputs"
        key = ("conv1d", x.shape, w.shape, None if b is None else b.shape, str(x.dtype))
        out = await self.cache.get(*key)
        if out is None:
            raw = F.conv1d(x, w, b)
            out = self._decompose_conv(raw)
            self.batch_queue.append((out, *key))
        await self._flush_batch()
        return out

    async def conv2d(self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert x.dim() == 4 and w.dim() == 4, "conv2d expects 4D inputs"
        key = ("conv2d", x.shape, w.shape, None if b is None else b.shape, str(x.dtype))
        out = await self.cache.get(*key)
        if out is None:
            raw = F.conv2d(x, w, b)
            out = self._decompose_conv(raw)
            self.batch_queue.append((out, *key))
        await self._flush_batch()
        return out

    async def conv3d(self, x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert x.dim() == 5 and w.dim() == 5, "conv3d expects 5D inputs"
        key = ("conv3d", x.shape, w.shape, None if b is None else b.shape, str(x.dtype))
        out = await self.cache.get(*key)
        if out is None:
            raw = F.conv3d(x, w, b)
            out = self._decompose_conv(raw)
            self.batch_queue.append((out, *key))
        await self._flush_batch()
        return out

    async def batch_norm(self, x: torch.Tensor, w: Optional[torch.Tensor] = None, b: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert x.dim() == 4, "batch_norm expects 4D inputs"
        key = ("batch_norm", x.shape, None if w is None else w.shape, None if b is None else b.shape, str(x.dtype))
        out = await self.cache.get(*key)
        if out is None:
            raw = F.batch_norm(x, w, b)
            # no heavy decomposition for batch_norm
            out = raw
            self.batch_queue.append((out, *key))
        await self._flush_batch()
        return out

    async def fft(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert x.is_complex(), "fft expects complex tensor"
        assert x.dim() >= 1, "fft expects at least 1D"
        key = ("fft", x.shape, str(x.dtype), dim)
        out = await self.cache.get(*key)
        if out is None:
            raw = torch.fft.fft(x, dim=dim)
            # frequency domain typically dense; skip decomposition
            out = raw
            self.batch_queue.append((out, *key))
        await self._flush_batch()
        return out

    async def ifft(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        assert x.is_complex(), "ifft expects complex tensor"
        assert x.dim() >= 1, "ifft expects at least 1D"
        key = ("ifft", x.shape, str(x.dtype), dim)
        out = await self.cache.get(*key)
        if out is None:
            raw = torch.fft.ifft(x, dim=dim)
            # time-domain dense; skip decomposition
            out = raw
            self.batch_queue.append((out, *key))
        await self._flush_batch()
        return out
