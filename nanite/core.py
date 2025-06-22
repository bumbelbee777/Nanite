import logging
import math
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .concept import ConceptGraph
from .config import ModelConfig, PrecisionLevel
from .ops import MultivectorOps, MixedPrecisionRouter, Quantizer, ComplexPReLU, PRECISION_CONFIGS
from .shared import get_shared_ops
from .tgu import ResponseGenerator
from .vm import SillyVM, Opcode,TypedValue, OpcodeMapper
from .utils import DebugLogger


class TaskComplexityEstimator(nn.Module):
    """Estimates task complexity using spectral analysis."""

    def __init__(
        self,
        threshold: float = 1e-3,
        eps: float = 1e-6,
        feature_dim: int = 32,
        use_fft: bool = True,
    ):
        super().__init__()
        self.threshold = threshold
        self.eps = eps
        self.use_fft = use_fft
        self.feature_dim = feature_dim

        # Use simple linear layers instead of ComplexMLP to avoid recursion
        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
        )

        print("[TaskComplexityEstimator] I got initialized successfully! :D")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate task complexity from input features."""
        # Accept both 2D and 3D tensors
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
        elif x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, d_model]
        batch_size, seq_len, d_model = x.shape
        
        # Reshape for feature extraction
        x_reshaped = x.reshape(-1, d_model)  # [batch_size * seq_len, d_model]
        
        # Extract features
        if self.use_fft:
            # Use FFT for feature extraction
            features = torch.fft.fft(x_reshaped, dim=-1)
            features = torch.abs(features)  # Get magnitude spectrum
        else:
            # Use direct feature extraction
            features = x_reshaped
        
        # Project to feature dimension
        features = features[:, :self.feature_dim]
        
        # Compute complexity score
        complexity = torch.norm(features, dim=-1)  # [batch_size * seq_len]
        complexity = complexity.reshape(batch_size, seq_len)  # [batch_size, seq_len]
        
        # Apply threshold
        complexity = torch.where(
            complexity > self.threshold,
            complexity,
            torch.tensor(self.eps, device=complexity.device)
        )
        
        # Return mean complexity as a scalar tensor
        return complexity.mean()


class LinearLayer(nn.Module):
    """Complex-valued linear layer with MultivectorOps."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ops = get_shared_ops()
        if self.ops is None:
            raise ValueError("MultivectorOps initialization failed - ops is None")
        # Weight shape: [out_dim, in_dim] for F.linear
        self.weight = nn.Parameter(
            torch.randn(out_dim, in_dim, dtype=torch.complex64) * 0.02
        )
        self.bias = nn.Parameter(torch.zeros(out_dim, dtype=torch.complex64))
        self.router = MixedPrecisionRouter(small=1024, low_rank=16, large=1_000_000)
        self.quantizer = Quantizer(
            qmin=PRECISION_CONFIGS[PrecisionLevel.INT4].qmin,
            qmax=PRECISION_CONFIGS[PrecisionLevel.INT4].qmax,
            init_scale=1.0,
            mode="complex"
        )
        self.eps = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"[LinearLayer] input x shape: {x.shape}, dtype: {x.dtype}")
        x = self._ensure_complex(x)
        x = self._sanitize_tensor(x)
        precision = self.router.get_precision(x)
        if precision == PrecisionLevel.INT4:
            weight = self.quantizer(self.weight)
        else:
            weight = self.weight
        weight = self._sanitize_tensor(weight)
        print(f"[LinearLayer] weight shape: {weight.shape}, dtype: {weight.dtype}")
        was_0d = x.dim() == 0
        was_1d = x.dim() == 1
        if was_0d:
            x = x.unsqueeze(0).unsqueeze(0)
        elif was_1d:
            x = x.unsqueeze(0)
        # Failsafe: ensure x and weight are at least 2D
        if x.dim() < 2:
            print(f"[LinearLayer][WARN] x is {x.dim()}D before F.linear, unsqueezing")
            x = x.unsqueeze(0)
        if weight.dim() < 2:
            print(f"[LinearLayer][WARN] weight is {weight.dim()}D before F.linear, unsqueezing")
            weight = weight.unsqueeze(0)
        out = F.linear(x, weight, self.bias)
        out = self._sanitize_tensor(out)
        if was_0d:
            out = out.squeeze(0).squeeze(0)
        elif was_1d:
            out = out.squeeze(0)
        print(f"[LinearLayer] output shape: {out.shape}, dtype: {out.dtype}")
        return out

    def _ensure_complex(self, x):
        """Ensure tensor is complex."""
        if not x.is_complex():
            return torch.complex(x, torch.zeros_like(x))
        return x

    def _sanitize_tensor(self, x):
        """Ensure tensor values are finite and handle numerical stability."""
        # Replace NaN and Inf values
        x = torch.where(torch.isnan(x), torch.tensor(self.eps, device=x.device), x)
        x = torch.where(torch.isinf(x), torch.tensor(self.eps, device=x.device), x)
        
        # Use MultivectorOps clamp for complex tensors
        x = self.ops.clamp(x, -1e6, 1e6)
        
        return x


class FeatureRouter(nn.Module):
    """Routes features based on complexity estimation."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int | None = None,
        threshold: float = 0.5,
        estimator: TaskComplexityEstimator | None = None,
        ops: MultivectorOps | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim or d_model * 2
        self.threshold = threshold
        self.estimator = estimator or TaskComplexityEstimator()
        self.ops = ops or get_shared_ops()  # Use shared instance

        # Simple routing network without ComplexMLP
        self.route_net = nn.Sequential(
            nn.Linear(d_model, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, d_model),
            nn.Sigmoid(),
        )
        print("[FeatureRouter] I got initialized successfully! :D")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.route_net(x)


class ComplexDropout(nn.Module):
    """Complex-valued dropout layer."""

    def __init__(self, p=0.1, seed=42):
        super().__init__()
        self.p = p
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        print("[ComplexDropout] I got initialized successfully! :D")

    def forward(self, x):
        if self.training:
            if not torch.is_complex(x):
                # Standard dropout for real tensors
                return F.dropout(x, self.p, training=True)
            # For complex tensors, apply dropout to real and imag parts separately
            real = torch.bernoulli(torch.ones_like(x.real) * (1 - self.p), generator=self.generator)
            imag = torch.bernoulli(torch.ones_like(x.imag) * (1 - self.p), generator=self.generator)
            x_real = x.real * real / (1 - self.p)
            x_imag = x.imag * imag / (1 - self.p)
            return torch.complex(x_real, x_imag)
        return x


class LayerNorm(nn.Module):
    """Complex-valued layer normalization."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim, dtype=torch.complex64))
        self.beta = nn.Parameter(torch.zeros(dim, dtype=torch.complex64))
        print("[LayerNorm] I got initialized successfully! :D")

    def forward(self, x: torch.Tensor):
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))

        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected input dimension {self.dim}, but got {x.shape[-1]}",
            )

        # Clamp and sanitize input using MultivectorOps for complex tensors
        x = torch.where(torch.isnan(x), torch.tensor(1e-8, device=x.device), x)
        x = torch.where(torch.isinf(x), torch.tensor(1e-8, device=x.device), x)
        
        # Use MultivectorOps clamp for complex tensors
        ops = get_shared_ops()
        x = ops.clamp(x, -1e6, 1e6)

        mean = x.mean(dim=-1, keepdim=True) if x.numel() > 0 else torch.zeros_like(x)
        var = x.var(dim=-1, keepdim=True, unbiased=False) if x.numel() > 0 else torch.ones_like(x)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = torch.where(torch.isnan(x), torch.tensor(1e-8, device=x.device), x)
        x = torch.where(torch.isinf(x), torch.tensor(1e-8, device=x.device), x)
        x = ops.clamp(x, -1e6, 1e6)
        return self.gamma * x + self.beta


class ComplexInputProjection(nn.Module):
    """Complex-valued input projection with proper shape handling."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.projection = ComplexLinear(config.d_model, config.d_model)
        self.eps = 1e-8
        # Use shared debug logger if available, else create one
        try:
            from .shared import get_shared_debug_logger
            self.debug = get_shared_debug_logger() or DebugLogger()
        except Exception:
            self.debug = DebugLogger()
        print("[ComplexInputProjection] I got initialized successfully! :D")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.debug.log_tensor("[InputProjection] Input", x)
        # Store original shape
        original_shape = x.shape
        print(f"[ComplexInputProjection] Input shape: {x.shape}")

        # Handle 2D input by adding batch dimension if needed
        if len(original_shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension
            print(f"[ComplexInputProjection] After adding batch dim: {x.shape}")
            self.debug.log_tensor("[InputProjection] After batch dim", x)

        # Ensure we have a 3D tensor [batch_size, seq_len, d_model]
        if len(x.shape) != 3:
            raise ValueError(f"Expected 3D tensor, got shape {x.shape}")

        # Apply projection
        x = self.projection(x)
        print(f"[ComplexInputProjection] After projection: {x.shape}")
        self.debug.log_tensor("[InputProjection] After projection", x)

        # Normalize
        x = x / (torch.norm(x, dim=-1, keepdim=True) + self.eps)
        self.debug.log_tensor("[InputProjection] After normalization", x)
        print(f"[ComplexInputProjection] After normalization: {x.shape}")

        return x


class PositionalEncoding(nn.Module):
    """Complex positional encoding."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len

        # Generate frequencies for the full d_model dimension
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model),
        )
        self.register_buffer("frequencies", div_term)

        print("[PositionalEncoding] I got initialized successfully! :D")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get the actual sequence length from the input
        seq_len = x.shape[0] if x.dim() == 2 else x.shape[1]
        
        # Generate positional encoding for the actual sequence length
        position = torch.arange(seq_len, device=x.device).unsqueeze(1)
        pos_enc = torch.zeros(seq_len, self.d_model, dtype=torch.complex64, device=x.device)
        pos_enc[:, 0::2] = torch.sin(position * self.frequencies)
        pos_enc[:, 1::2] = torch.cos(position * self.frequencies)
        
        # Add positional encoding to each sequence position
        if x.dim() == 2:
            # Input shape: [seq_len, d_model]
            return x + pos_enc
        else:
            # Input shape: [batch, seq_len, d_model]
            return x + pos_enc.unsqueeze(0)  # Add batch dimension


class DynamicActivation(nn.Module):
    """Dynamic activation."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.alpha = nn.Parameter(torch.ones(dim, dtype=torch.complex64))
        self.beta = nn.Parameter(torch.ones(dim, dtype=torch.complex64))
        self.gamma = nn.Parameter(torch.ones(dim, dtype=torch.complex64))
        print("[DynamicActivation] I got initialized successfully! :D")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply complex activation
        mag = torch.abs(x)
        phase = torch.angle(x)
        
        # Dynamic magnitude scaling
        mag = torch.tanh(self.alpha * mag)
        
        # Dynamic phase shift
        phase = phase + self.beta * torch.sin(phase)
        
        # Combine magnitude and phase
        return mag * torch.exp(1j * phase)


class MatrixTypeSelector(nn.Module):
    """Automatically selects the best structured matrix type based on input characteristics."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Initialize parameters for matrix type selection
        self.type_weights = nn.Parameter(
            torch.ones(4),
        )  # Weights for different matrix types
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)  # Temperature for softmax

        # Initialize feature extractors
        self.feature_net = ComplexMLP(
            ModelConfig(
                input_dim=4,
                mlp_dim=32,
                output_dim=4,
                n_heads=1,
                n_layers=1,
                dropout=0.0,
            )
        )

        # Matrix types: 0=Toeplitz, 1=BlockToeplitz, 2=Circulant, 3=Hankel
        self.matrix_types = ["toeplitz", "block_toeplitz", "circulant", "hankel"]


class InfiniToeplitz(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.eps = 1e-6
        self.ops = get_shared_ops()  # Add the ops attribute

        # Initialize complex parameters with proper scaling
        self.query = nn.Parameter(
            torch.randn(self.d_model, self.d_model, dtype=torch.complex64)
            / np.sqrt(self.d_model),
        )
        self.key = nn.Parameter(
            torch.randn(self.d_model, self.d_model, dtype=torch.complex64)
            / np.sqrt(self.d_model),
        )
        self.value = nn.Parameter(
            torch.randn(self.d_model, self.d_model, dtype=torch.complex64)
            / np.sqrt(self.d_model),
        )

        # Initialize matrix type selection weights
        self.type_weights = nn.Parameter(
            torch.randn(4, 4) / np.sqrt(4),
        )  # 4 features, 4 matrix types

        # Initialize temperature parameter for softmax
        self.temperature = nn.Parameter(torch.tensor(0.1))

        print("[InfiniToeplitz] I got initialized successfully! :D")

    def forward(self, x):
        """Forward pass through the attention layer."""
        # Get input shape
        batch_size, seq_len, d_model = x.shape
        
        # Normalize input
        x = x / (torch.norm(x, dim=-1, keepdim=True) + self.eps)

        # Project queries, keys, and values with proper scaling
        q = self.ops.matmul(x, self.query)  # [batch_size, seq_len, d_model]
        k = self.ops.matmul(x, self.key)    # [batch_size, seq_len, d_model]
        v = self.ops.matmul(x, self.value)  # [batch_size, seq_len, d_model]

        # Normalize projections
        q = q / (torch.norm(q, dim=-1, keepdim=True) + self.eps)
        k = k / (torch.norm(k, dim=-1, keepdim=True) + self.eps)
        v = v / (torch.norm(v, dim=-1, keepdim=True) + self.eps)

        # Reshape for geometric product
        q_reshaped = q.reshape(-1, d_model)  # [batch_size * seq_len, d_model]
        k_reshaped = k.reshape(-1, d_model)  # [batch_size * seq_len, d_model]
        v_reshaped = v.reshape(-1, d_model)  # [batch_size * seq_len, d_model]

        # Compute attention scores using geometric product
        scores = self.ops.matmul(q_reshaped, k_reshaped.T)  # [batch_size * seq_len, batch_size * seq_len]
        scores = scores.reshape(batch_size, seq_len, batch_size, seq_len)
        scores = scores.mean(dim=2)  # Average over batch dimension
        scores = scores / np.sqrt(d_model)
        
        # Use softmax on the real part only, but keep complex
        attn_weights = F.softmax(scores.real, dim=-1)  # [batch_size, seq_len, seq_len]
        attn_weights = torch.complex(attn_weights, torch.zeros_like(attn_weights))  # Convert to complex
        
        # Apply attention weights (keep batch dimension)
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, seq_len, d_model]

        # Final normalization
        attn_output = attn_output / (torch.norm(attn_output, dim=-1, keepdim=True) + self.eps)

        return attn_output


class ComplexMLP(nn.Module):
    """Complex-valued MLP with feature routing."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_ff = config.d_ff

        # Create layers
        self.lin1a = LinearLayer(self.d_model, self.d_ff)
        self.lin1b = LinearLayer(self.d_model, self.d_ff)
        self.lin2 = LinearLayer(self.d_ff, self.d_model)

        print("[ComplexMLP] I got initialized successfully! :D")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape if needed
        original_shape = x.shape
        if len(original_shape) > 2:
            x = x.reshape(-1, original_shape[-1])

        # Apply MLP
        xa = self.lin1a(x)
        xb = self.lin1b(x)
        x = xa * xb
        x = self.lin2(x)

        # Reshape back if needed
        if len(original_shape) > 2:
            x = x.reshape(*original_shape[:-1], self.d_model)

        return x


class TransformerLayer(nn.Module):
    """Complex transformer layer with attention and MLP."""

    def __init__(self, config: ModelConfig, ops: MultivectorOps, concept_graph=None):
        super().__init__()
        self.config = config
        self.ops = ops
        self.attention = InfiniToeplitz(config)
        self.norm1 = LayerNorm(config.d_model)
        self.norm2 = LayerNorm(config.d_model)
        self.mlp = ComplexMLP(config)
        self.dropout = ComplexDropout(config.dropout)
        self.activation = DynamicActivation(config.d_model)
        self.eps = 1e-8
        print("[TransformerLayer] I got initialized successfully! :D")

    def _sanitize_tensor(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if torch.isnan(x).any():
            print(f"WARNING: NaN values detected in {name}")
            print(f"Number of NaN values: {torch.isnan(x).sum().item()}")
            x = torch.where(torch.isnan(x), torch.tensor(self.eps, device=x.device), x)
        return x

    def forward(
        self,
        x: torch.Tensor,
        ops: MultivectorOps | None = None,
    ) -> torch.Tensor:
        x = self._sanitize_tensor(x, "transformer layer input")
        x = x + self.eps

        attn_output = self.attention(x)
        attn_output = self._sanitize_tensor(attn_output, "attention output")
        x = self.norm1(x + self.dropout(attn_output))
        x = self._sanitize_tensor(x, "after attention norm")

        ff_output = self.mlp(x)
        ff_output = self._sanitize_tensor(ff_output, "feed-forward output")
        x = self.norm2(x + self.dropout(ff_output))
        x = self._sanitize_tensor(x, "after feed-forward norm")

        x = self.activation(x)
        x = self._sanitize_tensor(x, "after activation")

        return x


class Transformer(nn.Module):
    """Main transformer model with complex operations and concept graph integration."""

    def __init__(
        self,
        config: ModelConfig,
        ops: MultivectorOps | None = None,
        concept_graph=None,
        debug_logger: DebugLogger | None = None,
    ):
        super().__init__()
        self.config = config
        self.ops = ops or get_shared_ops()  # Use shared ops instance
        self.concept_graph = concept_graph or ConceptGraph(
            max_size=config.concept_graph_size,
            num_basis=config.d_model,
        )
        self.debug = debug_logger or DebugLogger(print_to_stdout=False)
        
        # Initialize opcode mapper
        self.opcode_mapper = OpcodeMapper()
        
        # Bytecode execution monitoring
        self.execution_history = []
        self.last_execution_result = None
        self.execution_confidence = 0.0

        # Initialize components for the pipeline
        self.input_proj = ComplexInputProjection(config)
        self.pos_encoding = PositionalEncoding(config)

        # InfiniToeplitz attention with structured matrices
        self.attention = InfiniToeplitz(config)

        # Multivector encoding layers
        self.layers = nn.ModuleList(
            [
                TransformerLayer(config, self.ops, self.concept_graph)
                for _ in range(config.n_layers)
            ],
        )

        # Output projection
        self.output_proj = ComplexLinear(config.d_model, config.d_model)

        # Task complexity estimation
        self.complexity_estimator = TaskComplexityEstimator(
            threshold=1e-3,
            eps=1e-6,
            feature_dim=32,
            use_fft=True,
        )

        # Response generation components
        self.response_generator = ResponseGenerator(
            d_model=config.d_model,
            n_tgus=3,
            max_candidates=5,
            min_score=15.0,
            max_retries=3,
        )

        # VM components
        self.vm = SillyVM()  # Will be initialized during execution
        self.current_bytecode = None
        self.execution_state = {
            "active": False,
            "current_instruction": 0,
            "last_result": None,
            "execution_count": 0,
        }

        # Initialize weights
        self.apply(self._init_weights)

        print("[Transformer] I got initialized successfully! :D")

    def _init_weights(self, module):
        if isinstance(module, (ComplexLinear, LinearLayer)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def _update_concept_graph(self, x: torch.Tensor, layer_idx: int | None = None):
        """Update concept graph with new relationships from transformer states."""
        try:
            # Convert tensor to concept embeddings
            embeddings = x.detach().cpu().numpy()

            # Limit the number of concepts to prevent memory issues
            max_concepts_per_layer = 10
            concept_count = 0

            # Update concept graph with layer-specific information
            for i in range(min(embeddings.shape[0], 5)):  # Limit to first 5 rows
                for j in range(min(embeddings.shape[1], 5)):  # Limit to first 5 columns
                    if concept_count >= max_concepts_per_layer:
                        break
                        
                    if torch.norm(x[i, j]) > 0.1:  # Threshold for significant concepts
                        concept_name = f"C_layer_{layer_idx}_concept_{i}_{j}" if layer_idx is not None else f"C_concept_{i}_{j}"
                        # Fix: handle scalar embedding
                        emb_val = embeddings[i, j]
                        if np.isscalar(emb_val):
                            emb_tensor = torch.tensor(emb_val)
                        else:
                            emb_tensor = torch.from_numpy(emb_val)
                        self.concept_graph.add_concept(
                            name=concept_name,
                            embedding=emb_tensor,
                            basis_weights=x[i, j],
                        )
                        concept_count += 1

                        # Add relationships between concepts (limit to prevent memory issues)
                        if j > 0 and concept_count < max_concepts_per_layer:
                            prev_concept = f"C_layer_{layer_idx}_concept_{i}_{j-1}" if layer_idx is not None else f"C_concept_{i}_{j-1}"
                            self.concept_graph.add_concept(
                                name=prev_concept,
                                relationships={concept_name: 0.8},  # Strong relationship
                            )
                            concept_count += 1
                            
                if concept_count >= max_concepts_per_layer:
                    break
                    
        except Exception as e:
            print(f"[Transformer] Error in _update_concept_graph: {e}")
            # Continue without updating concept graph if there's an error
            pass

    def _generate_bytecode(self, x: torch.Tensor) -> list[tuple[Opcode, list[str]]]:
        """Generate bytecode using the canonical ConceptGraph API."""
        # Use the built-in traversal and bytecode generation logic
        return self.concept_graph.to_bytecode()

    async def _execute_bytecode(
        self,
        bytecode: list[tuple[Opcode, list[str]]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Execute bytecode instructions and monitor results (fully async)."""
        # Load bytecode into VM
        self.vm.load_program(bytecode)
        # Execute bytecode (now async)
        result = await self.vm.run()
        # Convert result to tensor
        if isinstance(result, (int, float)):
            result = torch.tensor(result, dtype=torch.complex64, device=x.device)
        elif isinstance(result, TypedValue):
            result = torch.tensor(result.value, dtype=torch.complex64, device=x.device)
        # Store execution result and compute confidence
        self.last_execution_result = result
        self.execution_confidence = self._compute_execution_confidence(result, x)
        # Log execution details
        self.execution_history.append({
            'bytecode': bytecode,
            'result': result,
            'confidence': self.execution_confidence,
            'vm_state': self.vm.get_execution_state()
        })
        return result.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions

    def _compute_execution_confidence(self, result: torch.Tensor, original: torch.Tensor) -> float:
        """Compute confidence in bytecode execution result."""
        if result is None or original is None:
            return 0.0
            
        # Normalize tensors
        result_norm = torch.norm(result)
        original_norm = torch.norm(original)
        
        if result_norm == 0 or original_norm == 0:
            return 0.0
            
        # Compute similarity metrics
        cosine_sim = torch.abs(torch.dot(result.flatten(), original.flatten())) / (result_norm * original_norm)
        magnitude_ratio = min(result_norm / original_norm, original_norm / result_norm)
        
        # Combine metrics
        confidence = 0.7 * cosine_sim + 0.3 * magnitude_ratio
        return float(confidence)

    def _convert_to_text(self, x: torch.Tensor) -> str:
        """Convert tensor output to human-readable text."""
        try:
            # Convert complex tensor to real values
            if x.is_complex():
                x = torch.abs(x)
            
            # Normalize values to [0, 1] range
            x = (x - x.min()) / (x.max() - x.min() + 1e-8)
            
            # Convert to ASCII characters (32-126 range)
            x = (x * 94 + 32).long()  # 94 printable ASCII characters
            
            # Convert to string
            chars = [chr(i.item()) for i in x.flatten() if 32 <= i.item() <= 126]
            return ''.join(chars)
        except Exception as e:
            print(f"[Transformer] Error converting tensor to text: {e}")
            return "Error converting output to text"

    async def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        generate_response: bool = False,
        num_tokens: int = 10,
    ) -> torch.Tensor:
        """Execute the full pipeline with enhanced bytecode execution monitoring."""
        try:
            # Log input tensor
            self.debug.log_tensor("Input", x)
            # print(f"[Transformer] Starting forward pass with input shape: {x.shape}")

            # 1. Input projection
            old_shape = x.shape
            # print(f"[Transformer] Starting input projection with shape: {x.shape}")
            x = self.input_proj(x)
            self.debug.log_shape_change("After input projection", old_shape, x.shape)
            self.debug.log_tensor("After input projection", x)
            # print(f"[Transformer] Completed input projection: {x.shape}")

            # 2. Positional encoding
            old_shape = x.shape
            # print(f"[Transformer] Starting positional encoding with shape: {x.shape}")
            x = self.pos_encoding(x)
            self.debug.log_shape_change("After positional encoding", old_shape, x.shape)
            self.debug.log_tensor("After positional encoding", x)
            # print(f"[Transformer] Completed positional encoding: {x.shape}")

            # 3. InfiniToeplitz attention
            old_shape = x.shape
            # print(f"[Transformer] Starting attention with shape: {x.shape}")
            x = self.attention(x)
            self.debug.log_shape_change(
                "After InfiniToeplitz attention",
                old_shape,
                x.shape,
            )
            self.debug.log_tensor("After InfiniToeplitz attention", x)
            # print(f"[Transformer] Completed attention: {x.shape}")

            # 4. Multivector encoding through transformer layers with concept graph updates
            for i, layer in enumerate(self.layers):
                old_shape = x.shape
                # print(f"[Transformer] Starting layer {i+1} with shape: {x.shape}")
                x = layer(x)
                self.debug.log_shape_change(
                    f"After transformer layer {i + 1}",
                    old_shape,
                    x.shape,
                )
                self.debug.log_tensor(f"After transformer layer {i + 1}", x)
                # print(f"[Transformer] Completed layer {i+1}: {x.shape}")

                # Update concept graph with layer state (but limit to avoid memory issues)
                if i < 2:  # Only update for first 2 layers to save memory
                    # print(f"[Transformer] Updating concept graph for layer {i}")
                    self._update_concept_graph(x, layer_idx=i)
                    # print(f"[Transformer] Completed concept graph update for layer {i}")

            bytecode = self._generate_bytecode(x)
            if bytecode:
                # Use a float32 tensor for logging to avoid dtype errors
                self.debug.log_tensor("Generated bytecode", torch.tensor([float(len(bytecode))], dtype=torch.float32))
                x = await self._execute_bytecode(bytecode, x)
                self.debug.log_tensor("After bytecode execution", x)
                # --- Robust shape fix for output projection ---
                d_model = self.config.d_model
                # Squeeze unnecessary dimensions
                while x.dim() > 2 and x.shape[0] == 1:
                    x = x.squeeze(0)
                while x.dim() > 2 and x.shape[1] == 1:
                    x = x.squeeze(1)
                if x.dim() == 0:
                    # Scalar: expand to [1, d_model] with value in first position
                    tmp = torch.zeros(d_model, dtype=x.dtype, device=x.device)
                    tmp[0] = x
                    x = tmp.unsqueeze(0)
                elif x.dim() == 1:
                    # 1D: pad or trim to d_model, then unsqueeze
                    if x.numel() < d_model:
                        tmp = torch.zeros(d_model, dtype=x.dtype, device=x.device)
                        tmp[:x.numel()] = x
                        x = tmp.unsqueeze(0)
                    elif x.numel() > d_model:
                        x = x[:d_model].unsqueeze(0)
                    else:
                        x = x.unsqueeze(0)
                elif x.dim() == 2:
                    # 2D: ensure last dim is d_model
                    if x.shape[-1] < d_model:
                        pad = torch.zeros((x.shape[0], d_model - x.shape[-1]), dtype=x.dtype, device=x.device)
                        x = torch.cat([x, pad], dim=-1)
                    elif x.shape[-1] > d_model:
                        x = x[..., :d_model]
                # Now x is [batch, d_model] or [1, d_model]

            
            # 6. Response generation based on task complexity
            if generate_response:
                # Estimate task complexity
                # print(f"[Transformer] Estimating task complexity")
                complexity = self.complexity_estimator(x)
                self.debug.log_tensor("Task complexity", complexity)
                print(f"[Transformer] Task complexity: {complexity.item()}")

                if complexity > 0.7:  # High complexity task
                    # Use simple output projection instead of TGU to save memory
                    print(f"[Transformer] Using simple output projection (high complexity)")
                    x = self.output_proj(x)
                    self.debug.log_tensor("Simple response", x)
                elif complexity > 0.3:  # Medium complexity task
                    # Use simple output projection
                    print(f"[Transformer] Using simple output projection (medium complexity)")
                    x = self.output_proj(x)
                    self.debug.log_tensor("Simple response", x)
                else:  # Low complexity task
                    # Use simple output projection
                    print(f"[Transformer] Using simple output projection (low complexity)")
                    x = self.output_proj(x)
                    self.debug.log_tensor("Simple response", x)
            else:
                # Standard output projection
                print(f"[Transformer] Using standard output projection")
                x = self.output_proj(x)
                self.debug.log_tensor("Output projection", x)

            # Reshape output to match target shape
            old_shape = x.shape
            # print(f"[Transformer] Reshaping output from {x.shape}")
            # Return the same shape as input instead of expanding
            self.debug.log_shape_change("Final output", old_shape, x.shape)
            self.debug.log_tensor("Final output", x)
            print(f"[Transformer] Final output shape: {x.shape}")

            # Convert to text if requested
            if generate_response:
                text_output = self._convert_to_text(x)
                print(f"[Transformer] Text output: {text_output}")

            return x
            
        except Exception as e:
            print(f"[Transformer] Error in forward pass: {e}")
            import traceback
            traceback.print_exc()
            # Return a safe fallback tensor with the same shape as input
            return torch.zeros_like(x)


class ComplexLinear(nn.Module):
    """Complex-valued linear layer with geometric algebra support."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        scale = 0.01
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, dtype=torch.complex64) * scale,
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_features, dtype=torch.complex64))
            if bias
            else None
        )
        self.eps = 1e-8

        print("[ComplexLinear] I got initialized successfully! :D")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(f"[ComplexLinear] input x shape: {x.shape}, dtype: {x.dtype}")
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))
        x = x + self.eps
        original_shape = x.shape
        was_0d = x.dim() == 0
        was_1d = x.dim() == 1
        if was_0d:
            x = x.unsqueeze(0).unsqueeze(0)
        elif was_1d:
            x = x.unsqueeze(0)
        # Failsafe: ensure x and weight are at least 2D
        if x.dim() < 2:
            print(f"[ComplexLinear][WARN] x is {x.dim()}D before F.linear, unsqueezing")
            x = x.unsqueeze(0)
        if self.weight.dim() < 2:
            print(f"[ComplexLinear][WARN] weight is {self.weight.dim()}D before F.linear, unsqueezing")
            weight = self.weight.unsqueeze(0)
        else:
            weight = self.weight
        if x.dim() > 2:
            batch_shape = x.shape[:-1]
            x_reshaped = x.reshape(-1, x.shape[-1])
            out = F.linear(x_reshaped, weight.t(), self.bias)
            out = out + self.eps
            out = out.reshape(*batch_shape, -1)
        else:
            out = F.linear(x, weight.t(), self.bias)
            out = out + self.eps
        if was_0d:
            out = out.squeeze(0).squeeze(0)
        elif was_1d:
            out = out.squeeze(0)
        print(f"[ComplexLinear] output shape: {out.shape}, dtype: {out.dtype}")
        return out
