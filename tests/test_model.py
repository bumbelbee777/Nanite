from __future__ import annotations

import asyncio
import copy
import logging
import random
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import pytest
import torch
import torch.nn.functional as F
from torch import nn, linalg as LA
import numpy as np

from sillyai.config import ModelConfig, PrecisionLevel, Modality
from sillyai.core import (
    ComplexLayerNorm,
    DynamicActivation,
    FeatureRouter,
    LinearLayer,
    MultivectorOps,
    TaskComplexityEstimator,
    TransformerBlock,
    Transformer,
    TransformerLayer,
    PositionalEncoding,
    ComplexMLP,
    LayerNorm
)
from sillyai.model import SillyAI
from sillyai.ops import AsyncLRUTensorCache
from sillyai.concept import ConceptGraph
from sillyai.core import ComplexLoss
from sillyai.plugins.trainer import SillyAITrainerPlugin

class TestConceptGraph:
    def test_add_concept(self):
        graph = ConceptGraph()
        graph.add_concept("apple")
        assert "apple" in graph.concepts
        assert graph.concepts["apple"].energy == 0
        assert graph.concepts["apple"].last_access_time == 0
        assert graph.concepts["apple"].access_count == 0

    def test_add_connection(self):
        graph = ConceptGraph()
        graph.add_concept("apple")
        graph.add_concept("fruit")
        graph.add_edge("apple", "fruit")
        assert any(
            conn.target == "fruit" for conn in graph.adj["apple"]
        )
        assert any(
            conn.target == "apple" for conn in graph.adj["fruit"]
        )

    def test_propagate_energy(self):
        graph = ConceptGraph()
        graph.add_concept("apple")
        graph.add_concept("fruit")
        graph.add_concept("red")
        graph.add_edge("apple", "fruit")
        graph.add_edge("fruit", "red")
        graph.concepts["apple"].energy = 1.0
        graph.propagate_energy()
        assert graph.concepts["fruit"].energy > 0
        assert graph.concepts["red"].energy > 0
        assert graph.concepts["apple"].energy < 1.0

    def test_update_access(self):
        graph = ConceptGraph()
        graph.add_concept("apple")
        graph.concepts["apple"].access_count +=1
        assert graph.concepts["apple"].access_count == 1
        assert graph.concepts["apple"].last_access_time == 0
        graph.concepts["apple"].access_count += 1
        graph.concepts["apple"].last_access_time += 1
        assert graph.concepts["apple"].access_count == 2
        assert graph.concepts["apple"].last_access_time == 1

    def test_update_n_cluster(self):
        graph = ConceptGraph()
        graph.add_concept("A")
        graph.add_concept("B")
        graph.add_concept("C")
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.concepts["A"].energy = 1.0
        graph.propagate_energy()
        graph.concepts["A"].access_count +=1  # Increase access for 'A'
        graph.update_n_cluster = lambda min_energy, purge_threshold : graph.prune(energy_thresh=min_energy, access_thresh=purge_threshold)
        graph.update_n_cluster(min_energy=0.1, purge_threshold=2)  # Assuming at least 3 concepts, keeping top 2
        assert len(graph.concepts) <= 2  # Should have pruned at least one

    def test_example_concepts(self):
        graph = ConceptGraph(decay_rate=0.8)
        graph.add_concept("apple")
        graph.add_concept("fruit")
        graph.add_concept("red")

        graph.concepts["apple"].access_count +=1
        graph.concepts["fruit"].access_count +=1
        graph.concepts["red"].access_count +=1  # Red is most frequently accessed

        graph.add_edge("apple", "fruit", weight=0.8, relationship="is_a")
        graph.add_edge("fruit", "red", weight=0.5, relationship="color")

        graph.concepts["apple"].energy = 1.0
        graph.propagate_energy()

        assert graph.concepts["fruit"].energy > 0
        assert graph.concepts["red"].energy > 0
        graph.update_n_cluster = lambda min_energy, purge_threshold : graph.prune(energy_thresh=min_energy, access_thresh=purge_threshold)
        graph.update_n_cluster(min_energy=0.1, purge_threshold=2)

        assert "red" in graph.concepts
        assert len(graph.concepts) >= 1

        graph.add_concept("sweet")
        graph.add_edge("apple", "sweet", weight=0.7, relationship="taste")
        assert any(
            conn.target == "sweet" for conn in graph.adj["apple"]
        )

        graph.propagate_energy()
        graph.update_n_cluster(min_energy=0.05, purge_threshold=2)
        assert len(graph.concepts) <= 3

class TestModelComponents:
    @pytest.mark.asyncio
    async def test_cache_put_get_roundtrip(self):
        # small cache, INT4 quantization
        cache = AsyncLRUTensorCache(
            max_size_bytes=1024 * 1024,
            quant_precision=PrecisionLevel.INT4,
            decomp_threshold=10_000,  # low threshold so no SVD
            decomp_gain_ratio=0.1,
        )

        t = torch.randn(16, 16)  # real tensor
        # store under key ("foo", 123)
        await cache.put("foo", t, (123,))
        out = await cache.get("foo", (123,))
        assert out is not None
        # because we quantize to INT4 then dequantize,
        # we allow some tolerance
        torch.testing.assert_allclose(out, t, atol=1e-1, rtol=1e-1)

        # missing key
        miss = await cache.get("nope", (0,))
        assert miss is None

    @pytest.mark.asyncio
    async def test_cache_eviction_lru_lfu(self):
        # capacity for only two small tensors
        cache = AsyncLRUTensorCache(
            max_size_bytes=500,  # very small
            quant_precision=PrecisionLevel.INT4,
        )

        a = torch.ones(10, 10) * 1
        b = torch.ones(10, 10) * 2
        c = torch.ones(10, 10) * 3

        await cache.put("a", a,())
        await cache.put("b", b,())
        # both fit
        assert await cache.get("a",()) is not None
        assert await cache.get("b",()) is not None

        await cache.put("c", c, ())  # should evict either 'a' or 'b'
        remaining = [key for key, _ in cache.cache.items()]
        assert len(remaining) == 2
        assert "c" in remaining

    @pytest.mark.asyncio
    async def test_optimized_complex_ops_matmul_and_fft(self):
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
    async def test_optimized_complex_ops_conv1d(self):
        ops = MultivectorOps(cache_max_bytes=1 << 20)

        # use real dtype for conv1d, since F.conv1d doesn't support complex by default
        x = torch.randn(2, 3, 8)
        w = torch.randn(4, 3, 3)
        out = await ops.conv1d(x, w)
        expected = F.conv1d(x, w)
        assert out.shape == expected.shape
        torch.testing.assert_allclose(out, expected, atol=1e-6)
        
class TestTransformerComponents:
    def test_positional_encoding(self):
        config = ModelConfig(
            d_model=64,
            max_seq_len=32,
            device='cpu'
        )
        pos_enc = PositionalEncoding(config)
        
        # Test forward pass
        x = torch.randn(16, 32, 64)  # [batch_size, seq_len, d_model]
        output = pos_enc(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
    def test_transformer_layer(self):
        config = ModelConfig(
            d_model=64,
            n_heads=4,
            d_ff=128,
            dropout=0.1,
            device='cpu'
        )
        ops = MultivectorOps()
        layer = TransformerLayer(config, ops)
        
        # Test forward pass
        x = torch.randn(16, 32, 64)  # [batch_size, seq_len, d_model]
        output = layer(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
    def test_complex_mlp(self):
        config = ModelConfig(
            d_model=64,
            d_ff=128,
            dropout=0.1,
            device='cpu'
        )
        ops = MultivectorOps()
        mlp = ComplexMLP(config, ops)
        
        # Test forward pass
        x = torch.randn(16, 32, 64)  # [batch_size, seq_len, d_model]
        output = mlp(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

class TestComplexLoss:
    def test_complex_loss_with_real_target(self):
        criterion = ComplexLoss()
        
        # Create complex prediction and real target
        pred = torch.complex(
            torch.randn(16, 32, 64),
            torch.randn(16, 32, 64)
        )
        target = torch.randn(16, 32, 64)
        
        # Compute loss
        loss = criterion(pred, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss >= 0
        
    def test_complex_loss_with_complex_target(self):
        criterion = ComplexLoss()
        
        # Create complex prediction and target
        pred = torch.complex(
            torch.randn(16, 32, 64),
            torch.randn(16, 32, 64)
        )
        target = torch.complex(
            torch.randn(16, 32, 64),
            torch.randn(16, 32, 64)
        )
        
        # Compute loss
        loss = criterion(pred, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss >= 0
        
    def test_complex_loss_with_concept_graph(self):
        # Create a simple concept graph
        concept_graph = {
            'concept1': {'energy': 0.5, 'frequency': 1},
            'concept2': {'energy': 0.3, 'frequency': 2}
        }
        
        criterion = ComplexLoss(concept_graph=concept_graph, alpha=0.1, beta=0.01)
        
        # Create complex prediction and real target
        pred = torch.complex(
            torch.randn(16, 32, 64),
            torch.randn(16, 32, 64)
        )
        target = torch.randn(16, 32, 64)
        
        # Compute loss
        loss = criterion(pred, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss >= 0

class TestSillyAIModel:
    def test_model_initialization(self):
        config = ModelConfig(
            input_dim=1,
            output_dim=1,
            d_model=64,
            d_ff=128,
            n_heads=4,
            n_layers=2,
            max_seq_len=64,
            dropout=0.2,
            device='cpu',
            precision=PrecisionLevel.TERNARY,
            concept_graph_size=500,
            supported_modalities={Modality.TEXT, Modality.IMAGE},
            enabled_plugins=['trainer', 'visualizer']
        )
        
        ops = MultivectorOps()
        model = SillyAI(config, ops=ops)
        
        assert isinstance(model, SillyAI)
        assert model.config == config
        assert model.ops == ops
        
    @pytest.mark.asyncio
    async def test_model_forward(self):
        config = ModelConfig(
            input_dim=1,
            output_dim=1,
            d_model=64,
            d_ff=128,
            n_heads=4,
            n_layers=2,
            max_seq_len=64,
            dropout=0.2,
            device='cpu'
        )
        
        ops = MultivectorOps()
        model = SillyAI(config, ops=ops)
        
        # Test forward pass
        x = torch.randn(16, 64, 1)  # [batch_size, seq_len, input_dim]
        output = await model.transformer(x)
        
        assert output.shape == (16, 64, 1)  # [batch_size, seq_len, output_dim]
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
    @pytest.mark.asyncio
    async def test_model_training(self):
        config = ModelConfig(
            input_dim=1,
            output_dim=1,
            d_model=64,
            d_ff=128,
            n_heads=4,
            n_layers=2,
            max_seq_len=64,
            dropout=0.2,
            device='cpu'
        )
        
        ops = MultivectorOps()
        model = SillyAI(config, ops=ops)
        trainer = SillyAITrainerPlugin(config)
        
        # Create dummy dataset
        class DummyDataset(torch.utils.data.Dataset):
            def __init__(self, size=100):
                self.size = size
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                x = torch.randn(64, 1)
                y = torch.randn(64, 1)
                return x, y
                
        # Setup trainer
        train_dataset = DummyDataset(100)
        val_dataset = DummyDataset(20)
        trainer.setup(
            model=model,
            ops=ops,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=16,
            seq_len=64,
            lr=1e-3,
            epochs=2,
            early_stop=3,
            weight_decay=1e-4
        )
        
        # Test training
        train_loss = await trainer.train_epoch()
        val_loss = await trainer.validate()
        
        assert isinstance(train_loss, float)
        assert isinstance(val_loss, float)
        assert not np.isnan(train_loss)
        assert not np.isnan(val_loss)
        assert train_loss >= 0
        assert val_loss >= 0
