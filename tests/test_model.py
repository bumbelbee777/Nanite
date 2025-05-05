# tests/test_sillyai.py

import asyncio

import pytest
import torch
import torch.nn.functional as F
import torch.testing

from ops import AsyncLRUTensorCache, MultivectorOps, PrecisionLevel
from config import ModelConfig
from model import SillyAI


@pytest.mark.asyncio
async def test_cache_put_get_roundtrip():
    # small cache, INT4 quantization
    cache = AsyncLRUTensorCache(
        max_size_bytes=1024 * 1024,
        quant_precision=PrecisionLevel.INT4,
        decomp_threshold=10_000,      # low threshold so no SVD
        decomp_gain_ratio=0.1
    )

    t = torch.randn(16, 16)  # real tensor
    # store under key ("foo", 123)
    await cache.put(t, "foo", 123)
    out = await cache.get("foo", 123)
    assert out is not None
    # because we quantize to INT4 then dequantize,
    # we allow some tolerance
    torch.testing.assert_allclose(out, t, atol=1e-1, rtol=1e-1)

    # missing key
    miss = await cache.get("nope", 0)
    assert miss is None


@pytest.mark.asyncio
async def test_cache_eviction_lru_lfu():
    # capacity for only two small tensors
    cache = AsyncLRUTensorCache(
        max_size_bytes=500,  # very small
        quant_precision=PrecisionLevel.INT4
    )

    a = torch.ones(10, 10) * 1
    b = torch.ones(10, 10) * 2
    c = torch.ones(10, 10) * 3

    await cache.put(a, "a")
    await cache.put(b, "b")
    # both fit
    assert await cache.get("a") is not None
    assert await cache.get("b") is not None

    await cache.put(c, "c")  # should evict either 'a' or 'b'
    remaining = [key for key, _ in cache.cache.items()]
    assert len(remaining) == 2
    assert "c" in remaining


@pytest.mark.asyncio
async def test_optimized_complex_ops_matmul_and_fft():
    ops = MultivectorOps(cache_max_bytes=1 << 20)

    # test matmul
    a = torch.randn(4, 4, dtype=torch.complex64)
    b = torch.randn(4, 4, dtype=torch.complex64)
    out = await ops.matmul(a, b)
    torch.testing.assert_allclose(out, torch.matmul(a, b), atol=1e-6)

    # test fft / ifft roundtrip
    x = torch.randn(16, dtype=torch.complex64)
    X = await ops.fft(x)
    x_back = await ops.ifft(X)
    torch.testing.assert_allclose(x_back, x, atol=1e-6)


@pytest.mark.asyncio
async def test_optimized_complex_ops_conv1d():
    ops = MultivectorOps(cache_max_bytes=1 << 20)

    # use real dtype for conv1d, since F.conv1d doesn't support complex by default
    x = torch.randn(2, 3, 8)
    w = torch.randn(4, 3, 3)
    out = await ops.conv1d(x, w)
    expected = F.conv1d(x, w)
    assert out.shape == expected.shape
    torch.testing.assert_allclose(out, expected, atol=1e-6)


def test_modelconfig_defaults_and_assignment():
    cfg = ModelConfig(
        dim=32,
        num_heads=4,
        mlp_dim=64,
        num_layers=2
    )
    # defaults
    assert cfg.dropout == 0.1
    assert isinstance(cfg.precision, PrecisionLevel)
    assert cfg.cache_max_bytes == 1 << 26
    # override
    cfg2 = ModelConfig(
        dim=16,
        num_heads=2,
        mlp_dim=32,
        num_layers=1,
        dropout=0.2,
        precision=PrecisionLevel.FP8,
        cache_max_bytes=1 << 20,
        decomp_threshold=500_000,
        decomp_gain_ratio=0.3,
        device="cpu",
        dtype="complex64"
    )
    assert cfg2.dim == 16
    assert cfg2.precision is PrecisionLevel.FP8
    assert cfg2.device == "cpu"
    assert cfg2.dtype == "complex64"


def test_forward_and_train_step(tmp_path):
    # synchronous tests
    cfg = ModelConfig(
        dim=16,
        num_heads=2,
        mlp_dim=32,
        num_layers=1,
        device=None,
        dtype="complex64"
    )
    model = SillyAI(cfg, num_classes=5)

    # forward
    x = torch.randn(2, 4, cfg.dim, dtype=torch.complex64)
    y = model.forward(x)
    assert y.shape == (2, 4, 5)
    # loss should be finite
    dummy_labels = torch.randint(0, 5, (2, 4))
    loss = model.train_step(x, dummy_labels)
    assert isinstance(loss, float)
    assert loss >= 0
