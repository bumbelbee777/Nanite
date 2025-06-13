import logging
import math
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .concept import ConceptGraph
from .config import ModelConfig
from .ops import MultivectorOps
from .shared import get_shared_ops
from .tgu import ResponseGenerator
from .vm import BytecodeEngine, Opcode


class DebugLogger:
    """Professional tensor debugging and logging utility."""

    def __init__(
        self,
        log_dir: str = "logs",
        print_to_stdout: bool = False,
        warn_small_values: bool = False,
    ):
        self.print_to_stdout = print_to_stdout
        self.warn_small_values = warn_small_values
        self.log_dir = log_dir

        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Set up file logger
        self.logger = logging.getLogger("tensor_debug")
        self.logger.setLevel(logging.DEBUG)

        # Create a new log file for each run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"debug_{timestamp}.log")

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler (if enabled)
        if print_to_stdout:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_formatter = logging.Formatter("%(levelname)s: %(message)s")
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

    def log_tensor(self, name: str, tensor: torch.Tensor, level: str = "DEBUG"):
        """Log tensor information."""
        if not tensor.is_complex():
            tensor = torch.complex(tensor, torch.zeros_like(tensor))

        # Get tensor statistics using absolute values for complex tensors
        abs_tensor = torch.abs(tensor)
        stats = {
            "shape": tensor.shape,
            "mean": abs_tensor.mean().item(),
            "std": abs_tensor.std().item(),
            "min": abs_tensor.min().item(),
            "max": abs_tensor.max().item(),
            "is_complex": tensor.is_complex(),
            "device": str(tensor.device),
        }

        # Format message
        msg = f"{name}: {stats}"

        # Log with appropriate level
        if level.upper() == "WARNING":
            self.logger.warning(msg)
        else:
            self.logger.debug(msg)

    def log_small_values(self, tensor: torch.Tensor, threshold: float = 1e-10):
        """Log warning for small values in tensor."""
        if not self.warn_small_values:
            return

        if not tensor.is_complex():
            tensor = torch.complex(tensor, torch.zeros_like(tensor))

        min_val = torch.abs(tensor).min().item()
        if min_val < threshold:
            self.logger.warning(f"Small values detected in tensor (min: {min_val:.2e})")

    def log_shape_change(self, name: str, old_shape: torch.Size, new_shape: torch.Size):
        """Log tensor shape changes."""
        self.logger.debug(f"{name} shape changed: {old_shape} -> {new_shape}")


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
        """Estimate task complexity."""
        # Extract features
        if self.use_fft:
            # Use FFT for spectral analysis
            fft = torch.fft.fft2(x)
            features = torch.abs(fft).mean(dim=(-2, -1))
        else:
            # Use direct computation
            features = torch.norm(x, dim=(-2, -1))

        # Apply feature network
        scores = self.feature_net(features).sigmoid()
        return torch.clamp(scores, 0.0, 1.0)


class LinearLayer(nn.Module):
    """Linear layer with feature routing."""

    def __init__(self, in_dim: int, out_dim: int, ops: MultivectorOps | None = None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ops = ops or get_shared_ops()  # Use shared instance from shared module

        # Initialize weights
        self.weight = nn.Parameter(
            torch.randn(out_dim, in_dim, dtype=torch.complex64) * 0.01,
        )
        self.bias = nn.Parameter(torch.zeros(out_dim, dtype=torch.complex64))

        # Simple routing without FeatureRouter
        self.route_weight = nn.Parameter(torch.ones(in_dim, dtype=torch.complex64))
        print("[LinearLayer] I got initialized successfully! :D")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply routing weights
        x = x * self.route_weight
        # Linear transformation
        return F.linear(x, self.weight, self.bias)


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
    """Complex-valued dropout layer using MultivectorOps."""

    def __init__(self, p=0.1, seed=42):
        super().__init__()
        self.p = p
        self.ops = get_shared_ops()
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def forward(self, x):
        return self.ops.complex_dropout(x, p=self.p, training=self.training)


class LayerNorm(nn.Module):
    """Complex-valued layer normalization using MultivectorOps."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.ops = get_shared_ops()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))

        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected input dimension {self.dim}, but got {x.shape[-1]}",
            )

        return self.ops.complex_layer_norm(x, dim=-1)


class ComplexInputProjection(nn.Module):
    """Complex input projection using MultivectorOps."""

    def __init__(self, config, ops):
        super().__init__()
        self.ops = ops or get_shared_ops()
        # Calculate input dimension from spatial dimensions (4x4x4 = 64)
        self.input_dim = 64  # Fixed input dimension for 3D grid
        self.projection = LinearLayer(self.input_dim, config.d_model)
        self.norm = LayerNorm(config.d_model)
        self.dropout = ComplexDropout(config.dropout)

    def forward(self, x: torch.Tensor, ops=None) -> torch.Tensor:
        ops = ops or self.ops
        # Store original shape
        original_shape = x.shape

        # Reshape to combine batch and sequence dimensions
        batch_size = original_shape[0]
        seq_len = original_shape[1]
        spatial_dims = original_shape[2:]

        # Flatten spatial dimensions into a single dimension
        x = x.reshape(batch_size * seq_len, -1)

        # Apply projection
        x = self.projection(x, ops)

        # Reshape back to original dimensions, replacing last dim with d_model
        x = x.reshape(batch_size, seq_len, self.projection.out_dim)

        # Apply normalization and dropout
        x = self.norm(x)
        x = self.dropout(x)
        return x


class PositionalEncoding(nn.Module):
    """Complex positional encoding using MultivectorOps."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ops = get_shared_ops()
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len

        # Generate frequencies and phases for the full d_model dimension
        position = torch.arange(self.max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model),
        )

        # Create full d_model dimensional encoding
        pos_enc = torch.zeros(self.max_seq_len, self.d_model, dtype=torch.complex64)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("frequencies", div_term)
        self.register_buffer("phases", pos_enc)

    def forward(
        self,
        x: torch.Tensor,
        ops: MultivectorOps | None = None,
    ) -> torch.Tensor:
        ops = ops or self.ops
        # Add positional encoding to each sequence position
        return x + self.phases.unsqueeze(0)  # Add batch dimension


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


class DynamicActivation(nn.Module):
    """Dynamic activation using MultivectorOps."""

    def __init__(self, dim):
        super().__init__()
        self.ops = get_shared_ops()
        self.dim = dim
        self.alpha = nn.Parameter(torch.ones(dim, dtype=torch.complex64))
        self.beta = nn.Parameter(torch.ones(dim, dtype=torch.complex64))
        self.gamma = nn.Parameter(torch.ones(dim, dtype=torch.complex64))

    def forward(
        self,
        x: torch.Tensor,
        ops: MultivectorOps | None = None,
    ) -> torch.Tensor:
        ops = ops or self.ops
        return ops.complex_activation(
            x,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )


class MatrixTypeSelector(nn.Module):
    """Automatically selects the best structured matrix type based on input characteristics."""

    def __init__(self, d_model: int, ops: MultivectorOps):
        super().__init__()
        self.d_model = d_model
        self.ops = ops

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
            ),
            ops,
        )

        # Matrix types: 0=Toeplitz, 1=BlockToeplitz, 2=Circulant, 3=Hankel
        self.matrix_types = ["toeplitz", "block_toeplitz", "circulant", "hankel"]

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input to determine matrix type."""
        # Compute various statistics
        spectral_norm = torch.norm(x, dim=(-1, -2)).mean()  # Spectral norm
        sparsity = (torch.abs(x) < 1e-3).float().mean()  # Sparsity
        symmetry = torch.abs(x - x.transpose(-1, -2)).mean()  # Symmetry measure
        locality = torch.abs(x[:, :, 1:] - x[:, :, :-1]).mean()  # Locality measure

        # Stack features
        features = torch.stack(
            [spectral_norm, sparsity, symmetry, locality],
            dim=-1,
        ).to(torch.complex64)

        return features

    def forward(self, x: torch.Tensor) -> tuple[str, torch.Tensor]:
        """Select matrix type and return corresponding parameters."""
        # Extract features
        features = self._extract_features(x)

        # Get type scores
        scores = self.feature_net(features)
        scores = scores.real  # Use real part for type selection

        # Apply temperature-scaled softmax
        scores = scores / self.temperature
        probs = F.softmax(scores, dim=-1)

        # Select matrix type
        type_idx = torch.argmax(probs).item()
        matrix_type = self.matrix_types[type_idx]

        # Generate parameters based on selected type
        if matrix_type == "toeplitz":
            params = self._generate_toeplitz_params(x)
        elif matrix_type == "block_toeplitz":
            params = self._generate_block_toeplitz_params(x)
        elif matrix_type == "circulant":
            params = self._generate_circulant_params(x)
        else:  # hankel
            params = self._generate_hankel_params(x)

        return matrix_type, params

    def _generate_toeplitz_params(self, x: torch.Tensor) -> torch.Tensor:
        """Generate parameters for Toeplitz matrix."""
        diags = torch.zeros(
            self.d_model * 2 - 1,
            device=x.device,
            dtype=torch.complex64,
        )
        for i in range(-self.d_model + 1, self.d_model):
            diags[i + self.d_model - 1] = torch.mean(torch.diagonal(x, offset=i))
        return diags

    def _generate_block_toeplitz_params(self, x: torch.Tensor) -> torch.Tensor:
        """Generate parameters for block Toeplitz matrix."""
        # Get input dimensions
        batch_size, seq_len, d_model = x.shape

        # Calculate block size as the largest perfect square that divides d_model
        block_size = int(math.sqrt(d_model))
        while d_model % block_size != 0:
            block_size -= 1

        # Ensure block_size is at least 2
        block_size = max(2, block_size)

        # Calculate number of blocks needed
        num_blocks = d_model // block_size

        # Initialize blocks tensor with correct shape
        blocks = torch.zeros(
            num_blocks * 2 - 1,  # Number of blocks
            block_size,  # Block height
            block_size,  # Block width
            device=x.device,
            dtype=torch.complex64,
        )

        # Extract and average blocks
        for i in range(-num_blocks + 1, num_blocks):
            # Calculate valid indices for the current block
            start_idx = max(0, i)
            end_idx = min(d_model, d_model + i)

            if end_idx > start_idx:  # Only process if we have valid indices
                # Extract the block from the key tensor
                block = x[
                    :,
                    :,
                    start_idx:end_idx,
                ]  # Shape: [batch_size, seq_len, block_size]

                # Average across batch and sequence dimensions
                block_avg = torch.mean(block, dim=(0, 1))  # Shape: [block_size]

                # Create a block matrix by repeating the average
                block_matrix = torch.zeros(
                    block_size,
                    block_size,
                    device=x.device,
                    dtype=torch.complex64,
                )
                for j in range(block_size):
                    block_matrix[j, j:] = block_avg[: block_size - j]
                    block_matrix[j:, j] = block_avg[: block_size - j]

                # Store the block matrix
                blocks[i + num_blocks - 1] = block_matrix

        return blocks

    def _generate_circulant_params(self, x: torch.Tensor) -> torch.Tensor:
        """Generate parameters for circulant matrix."""
        return x[:, 0, :].mean(dim=0)

    def _generate_hankel_params(self, x: torch.Tensor) -> torch.Tensor:
        """Generate parameters for Hankel matrix."""
        anti_diags = torch.zeros(
            self.d_model * 2 - 1,
            device=x.device,
            dtype=torch.complex64,
        )
        for i in range(-self.d_model + 1, self.d_model):
            anti_diags[i + self.d_model - 1] = torch.mean(
                torch.diagonal(x.flip(-1), offset=i),
            )
        return anti_diags


class InfiniToeplitz(nn.Module):
    def __init__(self, config: ModelConfig, ops: MultivectorOps):
        super().__init__()
        self.d_model = config.d_model
        self.ops = ops
        self.eps = 1e-6

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

    def forward(self, x, ops=None):
        if ops is None:
            ops = self.ops

        # Normalize input
        x = x / (torch.norm(x, dim=-1, keepdim=True) + self.eps)

        # Project queries, keys, and values with proper scaling
        q = ops.complex_linear(x, self.query)
        k = ops.complex_linear(x, self.key)
        v = ops.complex_linear(x, self.value)

        # Normalize projections
        q = q / (torch.norm(q, dim=-1, keepdim=True) + self.eps)
        k = k / (torch.norm(k, dim=-1, keepdim=True) + self.eps)
        v = v / (torch.norm(v, dim=-1, keepdim=True) + self.eps)

        # Select matrix type and generate parameters
        matrix_type, params = ops.select_matrix_type(
            k,
            self.type_weights,
            self.temperature,
        )

        # Apply structured matrix to keys
        k = ops.complex_structured_matrix(k, matrix_type, params)

        # Normalize keys after matrix application
        k = k / (torch.norm(k, dim=-1, keepdim=True) + self.eps)

        # Compute attention with scaled dot product
        attn_output = ops.complex_attention(q, k, v, scale=1.0 / np.sqrt(self.d_model))

        # Final normalization
        attn_output = attn_output / (
            torch.norm(attn_output, dim=-1, keepdim=True) + self.eps
        )

        return attn_output


class ComplexMLP(nn.Module):
    """Complex-valued MLP with feature routing."""

    def __init__(self, config: ModelConfig, ops: MultivectorOps | None = None):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        self.ops = ops or get_shared_ops()  # Use shared instance

        # Create layers with same ops instance
        self.lin1a = LinearLayer(self.d_model, self.d_ff, ops=self.ops)
        self.lin1b = LinearLayer(self.d_model, self.d_ff, ops=self.ops)
        self.lin2 = LinearLayer(self.d_ff, self.d_model, ops=self.ops)

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
        self.attention = InfiniToeplitz(config, ops)
        self.norm1 = LayerNorm(config.d_model)
        self.norm2 = LayerNorm(config.d_model)
        self.mlp = ComplexMLP(config, ops)
        self.dropout = ComplexDropout(config.dropout)
        self.activation = DynamicActivation(config.d_model)
        self.eps = 1e-8

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

        attn_output = self.attention(x, ops)
        attn_output = self._sanitize_tensor(attn_output, "attention output")
        x = self.norm1(x + self.dropout(attn_output))
        x = self._sanitize_tensor(x, "after attention norm")

        ff_output = self.mlp(x, ops)
        ff_output = self._sanitize_tensor(ff_output, "feed-forward output")
        x = self.norm2(x + self.dropout(ff_output))
        x = self._sanitize_tensor(x, "after feed-forward norm")

        x = self.activation(x, ops)
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
        self.ops = ops or MultivectorOps()
        self.concept_graph = concept_graph or ConceptGraph(
            max_size=config.concept_graph_size,
            num_basis=config.d_model,
        )
        self.debug = debug_logger or DebugLogger(print_to_stdout=False)

        # Initialize components for the pipeline
        self.input_proj = ComplexInputProjection(config, self.ops)
        self.pos_encoding = PositionalEncoding(config)

        # InfiniToeplitz attention with structured matrices
        self.attention = InfiniToeplitz(config, self.ops)

        # Multivector encoding layers
        self.layers = nn.ModuleList(
            [
                TransformerLayer(config, self.ops, self.concept_graph)
                for _ in range(config.n_layers)
            ],
        )

        # Output projection
        self.output_proj = ComplexLinear(config.d_model, 1)

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

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (ComplexLinear, LinearLayer)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def _update_concept_graph(self, x: torch.Tensor):
        """Update concept graph with new relationships."""
        # Convert tensor to concept embeddings
        embeddings = x.detach().cpu().numpy()

        # Update concept graph
        for i in range(embeddings.shape[0]):
            for j in range(embeddings.shape[1]):
                if torch.norm(x[i, j]) > 0.1:  # Threshold for significant concepts
                    concept_name = f"concept_{i}_{j}"
                    self.concept_graph.add_concept(
                        name=concept_name,
                        embedding=torch.from_numpy(embeddings[i, j]),
                        basis_weights=x[i, j],
                    )

    def _generate_bytecode(self, x: torch.Tensor) -> list[tuple[Opcode, list[str]]]:
        """Generate bytecode from concept graph traversal."""
        # Get concept relationships
        concepts = self.concept_graph.get_top_concepts()

        # Generate bytecode instructions
        instructions = []
        for concept, _ in concepts:
            # Get related concepts
            related = self.concept_graph.get_related_concepts(concept)

            # Generate opcodes based on relationships
            for target, weight in related:
                if weight > 0.5:  # Significant relationship
                    # Map relationship to opcode
                    op = self._map_relationship_to_opcode(weight)
                    args = [concept, target]
                    instructions.append((op, args))

        return instructions

    def _map_relationship_to_opcode(self, weight: float) -> Opcode:
        """Map relationship weight to appropriate opcode."""
        if weight > 0.8:
            return Opcode.ADD
        elif weight > 0.6:
            return Opcode.MUL
        elif weight > 0.4:
            return Opcode.SUB
        else:
            return Opcode.DIV

    def _execute_bytecode(
        self,
        bytecode: list[tuple[Opcode, list[str]]],
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Execute bytecode instructions on input tensor."""
        # Create virtual machine
        vm = BytecodeEngine(bytecode)

        # Execute bytecode
        vm.run()

        # Get result from VM
        result = vm.engine.registers["R0"].value

        # Convert result back to tensor
        if isinstance(result, (int, float)):
            result = torch.tensor(result, dtype=torch.complex64, device=x.device)

        return result.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions

    async def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        generate_response: bool = False,
        num_tokens: int = 10,
    ) -> torch.Tensor:
        """Execute the full pipeline."""
        # Log input tensor
        self.debug.log_tensor("Input", x)

        # 1. Input projection
        old_shape = x.shape
        x = self.input_proj(x)
        self.debug.log_shape_change("After input projection", old_shape, x.shape)
        self.debug.log_tensor("After input projection", x)

        # 2. Positional encoding
        old_shape = x.shape
        x = self.pos_encoding(x)
        self.debug.log_shape_change("After positional encoding", old_shape, x.shape)
        self.debug.log_tensor("After positional encoding", x)

        # 3. InfiniToeplitz attention
        old_shape = x.shape
        x = self.attention(x)
        self.debug.log_shape_change(
            "After InfiniToeplitz attention",
            old_shape,
            x.shape,
        )
        self.debug.log_tensor("After InfiniToeplitz attention", x)

        # 4. Multivector encoding through transformer layers
        for i, layer in enumerate(self.layers):
            old_shape = x.shape
            x = layer(x)
            self.debug.log_shape_change(
                f"After transformer layer {i + 1}",
                old_shape,
                x.shape,
            )
            self.debug.log_tensor(f"After transformer layer {i + 1}", x)

        # 5. Concept graph construction
        if self.training:
            self._update_concept_graph(x)

        # 6. Response generation based on task complexity
        if generate_response:
            # Estimate task complexity
            complexity = self.complexity_estimator(x, self.ops)
            self.debug.log_tensor("Task complexity", complexity)

            if complexity > 0.7:  # High complexity task
                # Use full TGU-based response generation
                candidates = await self.response_generator.generate_candidates(
                    x,
                    num_tokens,
                )
                response = await self.response_generator.synthesize_response(candidates)
                if response is not None:
                    x = response
                    self.debug.log_tensor("TGU response", x)
            elif complexity > 0.3:  # Medium complexity task
                # Use bytecode execution with concept graph
                bytecode = self._generate_bytecode(x)
                x = self._execute_bytecode(bytecode, x)
                self.debug.log_tensor("Bytecode response", x)
            else:  # Low complexity task
                # Use simple output projection
                x = self.output_proj(x)
                self.debug.log_tensor("Simple response", x)
        else:
            # Standard output projection
            x = self.output_proj(x)
            self.debug.log_tensor("Output projection", x)

        # Reshape output to match target shape
        old_shape = x.shape
        x = x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = x.expand(*x.shape[:-3], 4, 4, 4)
        self.debug.log_shape_change("Final output", old_shape, x.shape)
        self.debug.log_tensor("Final output", x)

        return x


class ComplexLinear(nn.Module):
    """Complex-valued linear layer with geometric algebra support."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        scale = 0.01
        self.weight = nn.Parameter(
            torch.randn(in_features, out_features, dtype=torch.complex64) * scale,
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_features, dtype=torch.complex64))
            if bias
            else None
        )
        self.eps = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))

        x = x + self.eps

        if x.dim() > 2:
            batch_shape = x.shape[:-1]
            x_reshaped = x.reshape(-1, x.shape[-1])
            out = F.linear(x_reshaped, self.weight.t(), self.bias)
            out = out + self.eps
            return out.reshape(*batch_shape, -1)
        else:
            out = F.linear(x, self.weight.t(), self.bias)
            out = out + self.eps
            return out
