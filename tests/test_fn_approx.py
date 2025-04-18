import pytest
import torch
import cmath
import math
from sillyai.ai import SillyAI
from sillyai.concept import ConceptGraph

def complex_abs(z):
    return abs(z)

def complex_sin(z):
    return cmath.sin(z)

def generate_complex_inputs(num_samples, real_range=(-1.0, 1.0), imag_range=(-1.0, 1.0)):
    real_part = torch.rand(num_samples) * (real_range[1] - real_range[0]) + real_range[0]
    imag_part = torch.rand(num_samples) * (imag_range[1] - imag_range[0]) + imag_range[0]
    return torch.stack([real_part, imag_part], dim=-1)

def generate_real_inputs(num_samples, real_range=(0.0, 1.0)):
    return torch.rand(num_samples) * (real_range[1] - real_range[0]) + real_range[0]

def test_sillyai_initialization():
    # Test with real mode
    model_real = SillyAI(input_dim=1, d_model=32, num_layers=2, nhead=2, dim_ff=64, real_mode=True, output_dim=1)
    assert model_real.real_mode == True

    # Test with complex mode
    model_complex = SillyAI(input_dim=1, d_model=32, num_layers=2, nhead=2, dim_ff=64, real_mode=False, output_dim=1)
    assert model_complex.real_mode == False

    # Test with different num_layers
    model_layers = SillyAI(input_dim=1, d_model=32, num_layers=4, nhead=2, dim_ff=64, real_mode=False, output_dim=1)
    assert len(model_layers.layers) == 4

def test_complex_function_approximation():
    # Initialize model for complex approximation
    model = SillyAI(input_dim=1, d_model=16, num_layers=1, nhead=1, dim_ff=32, real_mode=False, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_samples = 100

    # Test abs(z)
    complex_inputs = generate_complex_inputs(num_samples)
    complex_targets_abs = torch.tensor([complex_abs(complex(z[0].item(), z[1].item())) for z in complex_inputs]).unsqueeze(-1)
    
    # Train
    for _ in range(10):
        loss = model.train_step((complex_inputs, complex_targets_abs), optimizer)
        assert loss >= 0

    preds = model(complex_inputs)
    complex_preds = torch.norm(preds, dim=-1)
    diff = torch.abs(complex_preds.squeeze() - complex_targets_abs.squeeze())
    assert torch.mean(diff) < 0.1

    # Test sin(z)
    complex_inputs = generate_complex_inputs(num_samples)
    complex_targets_sin = torch.tensor([[complex_sin(complex(z[0].item(), z[1].item())).real,
                                           complex_sin(complex(z[0].item(), z[1].item())).imag] for z in complex_inputs])
    
    #Train
    for _ in range(10):
        loss = model.train_step((complex_inputs, complex_targets_sin), optimizer)
        assert loss >= 0

    preds = model(complex_inputs)
    diff = torch.abs(preds - complex_targets_sin)
    assert torch.mean(diff) < 0.1

def test_real_function_approximation():
    # Initialize model for real approximation
    model = SillyAI(input_dim=1, d_model=16, num_layers=1, nhead=1, dim_ff=32, real_mode=True, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_samples = 100

    # Test x^2
    real_inputs = generate_real_inputs(num_samples, real_range=(0.0, 1.0)).unsqueeze(-1)
    real_targets_sq = real_inputs ** 2

    # Train
    for _ in range(10):
        loss = model.train_step((real_inputs, real_targets_sq), optimizer)
        assert loss >= 0

    preds = model(real_inputs)
    diff = torch.abs(preds.squeeze() - real_targets_sq.squeeze())
    assert torch.mean(diff) < 0.1

    # Test sqrt(x)
    real_inputs = generate_real_inputs(num_samples, real_range=(0.0, 1.0)).unsqueeze(-1)
    real_targets_sqrt = torch.sqrt(real_inputs)
    
    # Train
    for _ in range(10):
        loss = model.train_step((real_inputs, real_targets_sqrt), optimizer)
        assert loss >= 0

    preds = model(real_inputs)
    diff = torch.abs(preds.squeeze() - real_targets_sqrt.squeeze())
    assert torch.mean(diff) < 0.1

def test_hybrid_loss():
    # Test hybrid loss with concepts
    model = SillyAI(input_dim=1, d_model=16, num_layers=1, nhead=1, dim_ff=32, real_mode=False, output_dim=1)
    model.concept_graph.add_concept("test_concept")
    model.concept_graph.concepts["test_concept"].embedding = torch.randn(1, 2)  # Assign embedding
    complex_inputs = generate_complex_inputs(10)
    complex_targets = generate_complex_inputs(10)
    loss_with_concepts = model.hybrid_loss(model(complex_inputs), complex_targets)

    # Test hybrid loss without concepts
    model.concept_graph.concepts.clear()
    loss_without_concepts = model.hybrid_loss(model(complex_inputs), complex_targets)

    assert loss_with_concepts >= 0
    assert loss_without_concepts >= 0
    assert loss_with_concepts > loss_without_concepts

def test_concept_alignment_loss():
    # Test concept alignment loss with concepts
    model = SillyAI(input_dim=1, d_model=16, num_layers=1, nhead=1, dim_ff=32, real_mode=False, output_dim=1)
    model.concept_graph.add_concept("test_concept")
    model.concept_graph.concepts["test_concept"].embedding = torch.randn(1, 2)  # Assign embedding
    complex_inputs = generate_complex_inputs(10)
    concept_loss = model.concept_alignment_loss(model(complex_inputs))
    
    assert concept_loss >= 0
