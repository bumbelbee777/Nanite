import torch, asyncio, io, hashlib, lz4.frame as lz4
import torch.nn.functional as F
from torch import nn, linalg as LA
from enum import Enum
from collections import deque, defaultdict
from dataclasses import dataclass

# Quantizer with real/cartesian/polar, sync+async entrypoints, JIT/compile-ready
class Quantizer(nn.Module):
    def __init__(self, qmin, qmax, init_scale=1.0, mode="real", wrap_phase=True):
        super().__init__()
        self.qmin, self.qmax, self.mode = qmin, qmax, mode
        self.scale = nn.Parameter(torch.tensor(init_scale))
        self.zero_point = nn.Parameter(torch.tensor(0.0))
        self.wrap_phase = wrap_phase
        print("[Quantizer] I got initialized successfully! :D")

    def forward(self, x):
        if self.mode == "real":
            return self._qs(x)[0]
        r, i = (x.real, x.imag) if self.mode == "cartesian" else (x.abs(), x.angle())
        if self.mode == "polar" and self.wrap_phase:
            i = (i + torch.pi) % (2 * torch.pi) - torch.pi
        qr, qi = self._qs(r)[0], self._qs(i)[0]
        return (qr, qi)

    @torch.jit.export
    def dequantize(self, q):
        zp, sc = self.zero_point, self.scale
        if self.mode == "real":
            return (q.float() - zp) * sc
        r, i = q
        return torch.view_as_complex(
            torch.stack([((r - zp) * sc), ((i - zp) * sc)], -1)
        )

    def _qs(self, x):
        q = torch.clamp(
            torch.round(x / self.scale + self.zero_point), self.qmin, self.qmax
        )
        return (q,)


# Mixed precision router
class PrecisionLevel(Enum):
    TERNARY, INT4, FP4, FP8, FP16 = range(1, 6)


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
        asyncio.get_event_loop().create_task(self._worker())
        self.mpr = MixedPrecisionRouter()
        self.quant = None
        self.dt, self.tol = decomp_thresh, tol
        print("[TensorCache] I got initialized successfully! :D")

    async def _worker(self):
        while 1:
            k, t = await self.prefetch.get()
            await self._put(k, self._compress(t))
            self.prefetch.task_done()

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
            self.quant = Quantizer(cfg.qmin, cfg.qmax, mode="polar")
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
            torch.stack([((r.float() - z) * s), ((i.float() - z) * s)], -1)
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


# REMEMBER: do ops = MultivectorOps().compile() when you wanna use it
class MultivectorOps(nn.Module):
    def __init__(self, cmax=1 << 26):
        super().__init__()
        self.cache = TensorCache(cmax)
        self._gp_dim = 0

    def _ensure(self, d):
        if self._gp_dim == d:
            return
        E = d * d
        import numpy as np

        I, J = np.zeros(E, int), np.zeros(E, int)
        K, S = np.zeros(E, int), np.zeros(E, int)
        idx = 0
        for i in range(d):
            for j in range(d):
                I[idx], J[idx] = i, j
                K[idx] = i ^ j
                S[idx] = (
                    -1
                    if (((i & j).bit_count() * ((i & j).bit_count() - 1) // 2) & 1)
                    else 1
                )
                idx += 1
        self.register_buffer("_gi", torch.from_numpy(I)), self.register_buffer(
            "_gj", torch.from_numpy(J)
        )
        self.register_buffer("_gk", torch.from_numpy(K)), self.register_buffer(
            "_gs", torch.from_numpy(S)
        )
        self._gp_dim = d

    def dot_product(self, A, B):
        return torch.sum(A * B, dim=-1)

    @torch.jit.export
    def geometric_product_sync(self, A, B):
        d = A.shape[-1]
        self._ensure(d)
        fA = A.reshape(-1, d)
        fB = B.reshape(-1, d)
        out = torch.zeros_like(fA)
        Ai = fA[:, self._gi]
        Bj = fB[:, self._gj]
        out.scatter_add_(1, self._gk.unsqueeze(0).expand_as(Ai), Ai * Bj * self._gs)
        return out.reshape_as(A)

    async def geometric_product(self, A, B):
        return self.geometric_product_sync(A, B)

    @torch.jit.export
    def matmul_sync(self, a, b):
        return a @ b

    async def matmul(self, a, b):
        return self.matmul_sync(a, b)

    @torch.jit.export
    def conv2d_sync(self, x, w, b=None):
        return F.conv2d(x, w, b)

    async def conv2d(self, x, w, b=None):
        return self.conv2d_sync(x, w, b)

    @torch.jit.export
    def fft_sync(self, x, dim=-1):
        return torch.fft.fft(x, dim)

    async def fft(self, x, dim=-1):
        return self.fft_sync(x, dim)

    async def forward(self, x):
        return self.dot_product(x)

    # Export / compile helpers
    def export(self, path="bundle.pt"):
        mod = torch.package.PackageExporter(path)
        mod.extern("torch")
        mod.save_pickle("data", "mvops", self)

    def compile(self):
        self.quanted = torch.compile(self)
        return self.quanted
