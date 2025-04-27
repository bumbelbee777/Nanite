import pytest
import torch
import cmath
import math
import os
import numpy as np
import torch.nn as nn
from torch import serialization
from sillyai.api import SillyAI, ModelConfig
from sillyai.graph.concept import ConceptGraph
from sillyai.plugins.plugin import SillyPlugin
from sillyai.core.transformer import DynamicActivation
from sillyai.core.complex.norms import ComplexLayerNorm

# Add both ModelConfig and ConceptGraph to safe globals at the start
serialization.add_safe_globals([ModelConfig, ConceptGraph])

def complex_abs(z):
    return abs(z)

def complex_sin(z):
    return cmath.sin(z)

def generate_complex_inputs(num_samples, real_range=(-1.0, 1.0), imag_range=(-1.0, 1.0)):
    real_part = torch.rand(num_samples) * (real_range[1] - real_range[0]) + real_range[0]
    imag_part = torch.rand(num_samples) * (imag_range[1] - imag_range[0]) + imag_range[0]
    return torch.stack([real_part, imag_part], dim=-1)

def generate_complex_data(n_samples=1000):
    """Generate complex-valued test data points"""
    # Generate points in a grid on the complex plane
    real = np.linspace(-2, 2, int(np.sqrt(n_samples)))
    imag = np.linspace(-2, 2, int(np.sqrt(n_samples)))
    x, y = np.meshgrid(real, imag)
    z = x + 1j * y
    z = z.flatten()
    
    return torch.tensor(np.stack([z.real, z.imag], axis=-1), dtype=torch.float32)

@pytest.fixture
def base_config():
    return ModelConfig(
        # Core model dimensions
        input_dim=1,
        d_model=32,
        num_layers=2,
        nhead=2,
        dim_ff=64,
        output_dim=1,
        concept_dim=16,
        
        # Optimization parameters
        factorized_linear=False,
        kronecker_rank=4,
        toeplitz_complex_method='fft',
        weight_init='xavier',
        
        # InfiniToeplitz parameters
        infini_local_window=512,
        infini_mem_size=1024,
        infini_compress_ratio=4,
        
        # Training parameters
        mixed_precision=True,
        snapshot_dir="./test_snapshots",
        keep_best_only=True,
        
        # Plugin configuration
        plugin_dir="./test_plugins",
        
        # Optional parameters with defaults
        optim_args={
            'use_toeplitz': True,
            'factorized_linear': False,
            'mixed_precision': True
        }
    )

def test_sillyai_initialization(base_config):
    config = base_config

    # Test with different num_layers
    config.num_layers = 4
    model_layers = SillyAI(config)
    assert len(model_layers.layers) == 4

def test_complex_function_approximation():
    """Test SillyAI's ability to approximate complex functions sin(z) and abs(z)"""
    
    # Create model config
    config = ModelConfig(
        input_dim=2,          # Real and imaginary components
        output_dim=2,         # Real and imaginary output
        concept_dim=32,       # Dimension for concept embeddings
        d_model=64,           # Model dimension
        num_layers=3,         # Number of transformer layers
        nhead=4,              # Number of attention heads
        dim_ff=128,          # Feed-forward dimension
        factorized_linear=True,
        kronecker_rank=4
    )
    
    # Initialize model
    model = SillyAI(config)
    
    # Generate training data
    z = generate_complex_data(1000)
    # Reshape to [batch, seq_len=1, input_dim, 2]
    z = z.unsqueeze(1)
    
    # Compute target values for sin(z)
    z_complex = torch.complex(z[..., 0], z[..., 1])
    sin_z = torch.sin(z_complex)
    y_sin = torch.stack([sin_z.real, sin_z.imag], dim=-1)
    
    # Compute target values for abs(z)
    abs_z = torch.abs(z_complex)
    y_abs = torch.stack([abs_z, torch.zeros_like(abs_z)], dim=-1)
    
    # Train model on sin(z)
    losses_sin = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(100):
        optimizer.zero_grad()
        out_sin = model(z)
        loss = nn.MSELoss()(out_sin, y_sin)
        loss.backward()
        optimizer.step()
        losses_sin.append(loss.item())
    
    # Verify sin(z) approximation
    with torch.no_grad():
        pred_sin = model(z)
        error_sin = torch.mean(torch.abs(pred_sin - y_sin))
        assert error_sin < 0.1, f"Sin(z) approximation error too high: {error_sin}"
    
    # Reset model and train on abs(z)
    model = SillyAI(config)
    losses_abs = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(100):
        optimizer.zero_grad()
        out_abs = model(z)
        loss = nn.MSELoss()(out_abs, y_abs)
        loss.backward()
        optimizer.step()
        losses_abs.append(loss.item())
    
    # Verify abs(z) approximation
    with torch.no_grad():
        pred_abs = model(z)
        error_abs = torch.mean(torch.abs(pred_abs - y_abs))
        assert error_abs < 0.1, f"Abs(z) approximation error too high: {error_abs}"
    
    # Check convergence
    assert losses_sin[-1] < losses_sin[0], "Loss did not decrease for sin(z)"
    assert losses_abs[-1] < losses_abs[0], "Loss did not decrease for abs(z)"
    
    # Test model properties
    assert model.config.input_dim == 2
    assert model.config.output_dim == 2
    assert isinstance(model(z), torch.Tensor)
    assert model(z).shape == (1000, 1, 2, 2)  # [batch, seq_len, output_dim, 2]

def test_concept_graph_integration(base_config):
    model = SillyAI(base_config)
    
    # Test concept graph initialization
    assert hasattr(model, 'concept_graph')
    assert isinstance(model.concept_graph, ConceptGraph)
    
    # Test concept addition and energy propagation
    model.concept_graph.add_concept("test_concept1")
    model.concept_graph.add_concept("test_concept2")
    model.concept_graph.add_connection("test_concept1", "test_concept2", weight=0.8)
    
    model.concept_graph.concepts["test_concept1"].energy = 1.0
    model.concept_graph.propagate_energy()
    
    assert model.concept_graph.concepts["test_concept2"].energy > 0

def test_plugin_system(base_config):
    class TestPlugin(SillyPlugin):
        def name(self) -> str:
            return "test_plugin"
            
        def before_forward(self, model, x):
            return x * 2  # Multiply input by 2
    
    # Create test plugin directory and add plugin
    os.makedirs("test_plugins", exist_ok=True)
    
    config = base_config
    config.plugin_dir = "test_plugins"
    model = SillyAI(config)
    
    # Manually add plugin for testing
    plugin = TestPlugin()
    model.plugin_manager.plugins.append(plugin)
    
    # Test plugin effect
    x = torch.ones(1, 1, model.config.d_model * 2)
    modified_x = model.plugin_manager.apply_before_forward(model, x)
    assert torch.all(modified_x == 2)

def test_vm_integration(base_config):
    model = SillyAI(base_config)
    
    # Test VM initialization
    assert hasattr(model, 'vm')
    
    # Test proof verification with simple bytecode
    proof_code = """
    .TestProof() {
        %x: r0<int> = 1
        %y: r1<int> = 1
        %result: r2<bool>
        IFF x, y, result  // Compare x and y
        ASSERT result     // Assert they are equal
        HLT
    }
    TestProof()
    """
    
    assert model._verify_proof_code(proof_code)

def test_snapshot_management(base_config):
    model = SillyAI(base_config)
    
    # Test snapshot saving with weights_only=False for backward compatibility
    test_loss = 0.5
    os.makedirs(base_config.snapshot_dir, exist_ok=True)
    snapshot_path = os.path.join(base_config.snapshot_dir, f"{model.__class__.__name__}_best.pt.gz")
    model.save_snapshot(test_loss)
    
    # Test loading snapshot
    loaded_model = SillyAI(base_config)
    loaded_model.load_snapshot(snapshot_path)
    assert loaded_model.best_loss == test_loss

    # Cleanup
    if os.path.exists(snapshot_path):
        os.remove(snapshot_path)
    os.rmdir(base_config.snapshot_dir)

def test_infini_attention(base_config):
    model = SillyAI(base_config)
    
    # Test with infini_local_window
    assert hasattr(model.layers[0].attn, 'chunk_size')
    assert model.layers[0].attn.chunk_size == base_config.infini_local_window
    assert hasattr(model.layers[0].attn, 'mem_k')
    assert model.layers[0].attn.mem_k.size(0) == 1
    
    # Test forward pass with complex inputs
    batch_size = 2
    seq_len = 6
    d_model = base_config.d_model

    x = torch.randn(batch_size, seq_len, d_model, 2)
    
    # Forward pass 
    output = model(x)
    expected_output_dim = base_config.output_dim
    assert output.shape == (batch_size, seq_len, expected_output_dim, 2)

    # Test memory mechanism
    # Do another forward pass to check memory updates
    new_output = model(x)
    assert model.layers[0].attn.mem_k.size(0) == 1  # Memory is maintained
    assert model.layers[0].attn.mem_v.size(0) == 1

    # Test compression by doing multiple passes
    for _ in range(5):
        model(x)
    # Memory should stay within bounds
    assert model.layers[0].attn.mem_k.size(0) <= base_config.infini_mem_size

def test_hybrid_loss_and_concept_alignment(base_config):
    model = SillyAI(base_config)
    
    # Setup test data
    batch_size = 2
    seq_len = 4
    d_model = base_config.d_model
    
    # Create complex inputs and targets
    x = torch.randn(batch_size, seq_len, d_model, 2)  # [B, L, D, 2]
    targets = torch.randn(batch_size, seq_len, base_config.output_dim, 2)
    
    # Add some concepts and set energies
    model.concept_graph.add_concept("test1")
    model.concept_graph.add_concept("test2")
    model.concept_graph.concepts["test1"].energy = 0.8
    model.concept_graph.concepts["test2"].energy = 0.5
    
    # Test hybrid loss computation
    loss = model.hybrid_loss(x, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # Scalar loss
    assert loss.item() >= 0
    
    # Test concept alignment loss separately
    align_loss = model.concept_alignment_loss(x)
    assert isinstance(align_loss, torch.Tensor)
    assert align_loss.ndim == 0
    
    # Verify concept projector is working
    proj = model.concept_projector(x)
    assert proj.shape == (batch_size * seq_len, base_config.concept_dim, 2)

def test_complex_components(base_config):
    model = SillyAI(base_config)
    
    # Test input shapes
    batch_size = 2
    seq_len = 4
    d_model = base_config.d_model
    
    # Create complex input: [B, L, D, 2]
    x = torch.randn(batch_size, seq_len, d_model, 2)
    
    # Test DynamicActivation in transformer blocks
    for layer in model.layers:
        # DynamicActivation is used in feed-forward when not in real mode
        assert isinstance(layer.ff.activation, DynamicActivation)
        
        # Test activation forward pass
        ff_out = layer.ff(x)
        assert ff_out.shape == x.shape
        
        # ComplexLayerNorm should be used
        assert isinstance(layer.norm1, ComplexLayerNorm)
        assert isinstance(layer.norm2, ComplexLayerNorm)
        
        # Test layer norm with separate real and imaginary parts
        norm_out = layer.norm1(x)
        assert norm_out.shape == x.shape
        real_mean = norm_out[..., 0].mean(dim=-1)
        real_std = norm_out[..., 0].std(dim=-1)
        imag_mean = norm_out[..., 1].mean(dim=-1)
        imag_std = norm_out[..., 1].std(dim=-1)
        
        # Verify normalization properties for both real and imaginary parts
        assert torch.allclose(real_mean, torch.zeros_like(real_mean), atol=1e-6)
        assert torch.allclose(imag_mean, torch.zeros_like(imag_mean), atol=1e-6)
        assert torch.allclose(real_std, torch.ones_like(real_std), atol=1e-6)
        assert torch.allclose(imag_std, torch.ones_like(imag_std), atol=1e-6)

def test_feature_routing(base_config):
    model = SillyAI(base_config)
    
    # Test input dimensions
    batch_size = 2
    seq_len = 4
    d_model = base_config.d_model
    
    # Test with complex input
    x = torch.randn(batch_size, seq_len, d_model, 2)
    
    # Get positional encoding from model
    pos_enc = model.pos_enc(x)
    
    # Pass pos_enc module to FeatureRouter
    encoded = model.pos_enc_router(x, model.pos_enc)
    assert encoded.shape == (batch_size, seq_len, d_model, 2)
    
    # Test TaskComplexityEstimator
    complexity_score = model.complexity_estimator(x)
    assert complexity_score.shape == (batch_size,)
    assert torch.all((complexity_score >= 0) & (complexity_score <= 1))
    
    # Test complexity estimation with different input patterns
    x_complex = torch.randn(batch_size, seq_len, d_model, 2) * 2.0
    x_simple = torch.ones(batch_size, seq_len, d_model, 2) * 0.1
    
    score_complex = model.complexity_estimator(x_complex)
    score_simple = model.complexity_estimator(x_simple)
    
    # Verify that more varied input has higher complexity score
    assert torch.mean(score_complex) > torch.mean(score_simple)
