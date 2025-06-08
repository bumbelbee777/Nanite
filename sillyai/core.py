import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .ops import *
from .config import ModelConfig
from .tgu import ResponseGenerator

def complex_checkpoint(function, *args):
    """Checkpointing wrapper that handles complex tensors"""
    # Convert complex inputs to real/imag pairs
    real_args = []
    for arg in args:
        if torch.is_tensor(arg) and arg.is_complex():
            real_args.extend([arg.real, arg.imag])
        else:
            real_args.append(arg)
    
    # Define a wrapper that reconstructs complex tensors
    def wrapper(*real_args_flat):
        reconstructed_args = []
        i = 0
        for arg in args:
            if torch.is_tensor(arg) and arg.is_complex():
                real_part = real_args_flat[i]
                imag_part = real_args_flat[i+1]
                reconstructed_args.append(torch.complex(real_part, imag_part))
                i += 2
            else:
                reconstructed_args.append(real_args_flat[i])
                i += 1
        return function(*reconstructed_args)
    
    # Apply checkpointing to the wrapper
    return torch.utils.checkpoint.checkpoint(wrapper, *real_args, use_reentrant=False)

class TaskComplexityEstimator(nn.Module):
    def __init__(
        self,
        threshold: float = 1e-3,
        eps: float = 1e-6,
        feature_dim: int = 32,
        use_fft: bool = True
    ):
        """
        threshold: any multivector component with |value| > threshold
                   counts as "active" toward complexity.
        """
        super().__init__()
        self.threshold = threshold
        self.eps       = eps
        self.use_fft   = use_fft
        from .config import ModelConfig
        from .ops import MultivectorOps
        # Minimal config for feature_net: input_dim=4, mlp_dim=feature_dim, output_dim=1
        feature_config = ModelConfig(input_dim=4, mlp_dim=feature_dim, output_dim=1, n_heads=1, n_layers=1, dropout=0.0)
        self.feature_net = ComplexMLP(feature_config, MultivectorOps())
        # JIT compile expensive computations
        self.get_spectral_norm = torch.jit.script(self._get_spectral_norm)
        self.get_rank_estimate = torch.jit.script(self._get_rank_estimate)
    
    @torch.jit.script
    def _get_spectral_norm(self, x: torch.Tensor, eps: float) -> torch.Tensor:
        """Estimate spectral norm using power iteration"""
        if x.ndim < 2:
            return torch.ones(1, device=x.device)
        
        batch_size = x.shape[0]
        x_flat = x.reshape(batch_size, -1)
        feat_size = x_flat.shape[1]
        
        # Initialize u randomly on correct device
        u = torch.randn((feat_size, 1), device=x.device)
        u_norm = torch.norm(u) + eps
        u = u / u_norm
        
        # Power iteration with fixed steps
        v: Optional[torch.Tensor] = None
        for _ in range(3):  # Usually 3 iterations is sufficient
            # v = x @ u
            v = torch.matmul(x_flat, u)
            v_norm = torch.norm(v) + eps
            v = v / v_norm
            
            # u = x.T @ v
            u = torch.matmul(x_flat.transpose(0, 1), v)
            u_norm = torch.norm(u) + eps
            u = u / u_norm
        
        # Final power iteration estimate
        if v is None:  # Satisfy TorchScript type checking
            v = torch.matmul(x_flat, u)
        
        final_term = torch.matmul(x_flat, u)
        result = torch.abs(torch.sum(v * final_term, dim=0))
        return result
    
    @torch.jit.script 
    def _get_rank_estimate(self, x: torch.Tensor, eps: float) -> torch.Tensor:
        """Estimate effective rank using eigenvalue decomposition"""
        if x.ndim < 2:
            return torch.ones(1, device=x.device)
            
        # Reshape and handle large matrices with projection
        x_flat = x.reshape(x.shape[0], -1)
        feat_size = x_flat.shape[1]
        
        if feat_size > 128:  # Random projection for large matrices
            proj = torch.randn(feat_size, 128, device=x.device)
            proj_norm = torch.norm(proj, dim=0, keepdim=True) + eps
            proj = proj / proj_norm
            x_flat = torch.matmul(x_flat, proj)
            
        # Compute Gram matrix for stability
        gram = torch.matmul(x_flat.transpose(0, 1), x_flat)
        gram = gram / (x_flat.shape[0] + eps)  # Normalize by batch size
        
        # Get eigenvalues through SVD
        s = torch.linalg.svdvals(gram)
        s = torch.clamp(s, min=eps)  # Ensure numerical stability
        
        # Normalize singular values to probability distribution
        total = torch.sum(s) + eps
        s = s / total
        
        # Compute entropy-based rank estimate
        entropy = -torch.sum(s * torch.log(s + eps))
        rank_estimate = torch.exp(entropy) / x_flat.shape[1]
        
        return rank_estimate
    
    def forward(self, mv: torch.Tensor, ops: Optional[MultivectorOps] = None) -> torch.Tensor:
        """
        Estimate task complexity from complex multivector input
        Returns: [B, L] or [B] complexity scores in [0,1]
        """
        if ops is None:
            ops = MultivectorOps()
            
        with torch.no_grad():
            # Get base features using scripted functions
            spectral = self.get_spectral_norm(mv, self.eps)
            rank = self.get_rank_estimate(mv, self.eps)
            
            # Compute sparsity using ops for complex magnitude
            mv_abs = ops.dot_product(mv, mv).sqrt()
            sparsity = torch.mean((mv_abs > self.threshold).float(), dim=-1)
            
            # Optional frequency domain features using ops
            if self.use_fft and mv.dim() > 2:
                fft_result = ops.fft_sync(mv)
                freq_feat = torch.mean(ops.dot_product(fft_result, fft_result).sqrt(), dim=-1)
            else:
                freq_feat = torch.zeros_like(spectral)
            
            # Combine features in complex domain
            features = torch.stack([
                spectral, rank, sparsity, freq_feat
            ], dim=-1).to(torch.complex64)
            
            # Get final complexity score using complex MLP
            scores = self.feature_net(features, ops).real.sigmoid()
            return torch.clamp(scores, 0.0, 1.0)  # [B, L]

class FeatureRouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        threshold: float = 0.5,
        estimator: Optional[TaskComplexityEstimator] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden = hidden_dim or (d_model // 2)
        self.threshold = threshold
        self.estimator = estimator or TaskComplexityEstimator()
        from .config import ModelConfig
        from .ops import MultivectorOps
        # Minimal config for router: input_dim=2, mlp_dim=hidden, output_dim=1
        router_config = ModelConfig(input_dim=2, mlp_dim=self.hidden, output_dim=1, n_heads=1, n_layers=1, dropout=0.0)
        self.complex_mlp = ComplexMLP(router_config, MultivectorOps())

    async def forward(self, mv: torch.Tensor, ops: Optional[MultivectorOps] = None) -> torch.Tensor:
        """
        mv: Complex tensor of shape [B, L, D] or [B, L, D, mv_dim]
        Returns: Complex tensor of shape [B, L, 1]
        """
        if ops is None:
            ops = MultivectorOps()
            
        # Get complexity estimation
        with torch.no_grad():
            comp_score = self.estimator(mv, ops)  # [B, L]
            # Get magnitude using ops
            mag = ops.dot_product(mv, mv).sqrt().mean(dim=-1)  # [B, L]
            # Stack features as complex tensor
            feats = torch.stack([comp_score, mag], dim=-1).to(torch.complex64)  # [B, L, 2]

        # Route using complex MLP
        gate = self.complex_mlp(feats, ops)  # [B, L, 1]
        return gate

    def forward_sync(self, mv: torch.Tensor, ops: Optional[MultivectorOps] = None) -> torch.Tensor:
        if ops is None:
            ops = MultivectorOps()
            
        comp_score = self.estimator(mv, ops)
        mag = ops.dot_product(mv, mv).sqrt().mean(dim=-1)
        feats = torch.stack([comp_score, mag], dim=-1).to(torch.complex64)
        return self.complex_mlp(feats, ops)

class ComplexDropout(nn.Module):
    def __init__(self, p=0.1, seed=42):
        super().__init__()
        self.p = p
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        
        # Use the real part shape for dropout mask size
        real_tensor = x.real if x.is_complex() else x
        
        # Create dropout mask with generator to keep reproducibility local
        mask = (torch.rand(real_tensor.shape, generator=self.generator, device=real_tensor.device) > self.p).to(real_tensor.dtype)
        
        if x.is_complex():
            # Scale mask to maintain expectation
            scale = 1 / (1 - self.p)
            real = x.real * mask * scale
            imag = x.imag * mask * scale
            return torch.complex(real, imag)
        else:
            return x * mask / (1 - self.p)

    def _make_mask(self, x_real):
        cpu_rng_state = torch.get_rng_state()
        torch.manual_seed(self.seed)
        mask = (torch.rand_like(x_real) > self.p).to(x_real.dtype)
        torch.set_rng_state(cpu_rng_state)
        return mask

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # Initialize with the correct normalized shape - just normalize over the last dimension
        self.ln_r = nn.LayerNorm(dim, eps=eps)  # Changed to normalize over last dimension only
        self.ln_i = nn.LayerNorm(dim, eps=eps)  # Changed to normalize over last dimension only

    def forward(self, x: torch.Tensor):
        # Handle both real and complex inputs
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))
            
        # Get input shape
        batch_size, seq_len, dim = x.shape
        
        # Ensure we're normalizing over the correct dimension
        if dim != self.dim:
            raise ValueError(f"Expected input dimension {self.dim}, but got {dim}")
            
        # Reshape to [batch_size * seq_len, dim] for normalization
        x_reshaped = x.reshape(-1, dim)
        
        # Apply layer norm to real and imaginary parts separately
        real_norm = self.ln_r(x_reshaped.real)
        imag_norm = self.ln_i(x_reshaped.imag)
        
        # Reshape back to original shape
        real_norm = real_norm.reshape(batch_size, seq_len, dim)
        imag_norm = imag_norm.reshape(batch_size, seq_len, dim)
        
        return torch.complex(real_norm, imag_norm)

class LinearLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Initialize weights with proper scaling
        # Note: weight shape is [in_features, out_features] for F.linear
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim, dtype=torch.complex64) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_dim, dtype=torch.complex64))
        
    def forward(self, x: torch.Tensor, ops: Optional[MultivectorOps] = None) -> torch.Tensor:
        if ops is None:
            ops = MultivectorOps()
            
        # Ensure input is complex
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))
            
        # Handle batched inputs correctly
        if x.dim() > 2:
            batch_shape = x.shape[:-1]
            x_reshaped = x.reshape(-1, x.shape[-1])
            # F.linear expects weight shape [out_features, in_features]
            out = F.linear(x_reshaped, self.weight.t(), self.bias)
            return out.reshape(*batch_shape, -1)
        else:
            return F.linear(x, self.weight.t(), self.bias)

class ComplexInputProjection(nn.Module):
    def __init__(self, config, ops):
        super().__init__()
        self.config = config
        self.ops = ops
        self.linear = LinearLayer(in_dim=1, out_dim=config.input_dim)
        self.norm = LayerNorm(config.input_dim)
        self.act = ComplexPReLU(config)

    def forward(self, x: torch.Tensor, ops=None) -> torch.Tensor:
        ops = ops or self.ops
        if ops is None:
            ops = MultivectorOps()
            
        # Convert input to complex if real
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))
            
        # Project using complex linear layer
        x = self.linear(x, ops=ops)
        x = self.norm(x)
        return self.act(x)
    
class PositionalEncoding(nn.Module):
    """Positional encoding using geometric frequencies."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.d_model = config.d_model
        self.max_seq_len = config.max_seq_len
        
        # Initialize geometric frequencies
        min_freq = 1.0 / self.max_seq_len
        max_freq = 1.0
        self.frequencies = nn.Parameter(
            torch.logspace(
                math.log10(min_freq),
                math.log10(max_freq),
                self.d_model // 2
            )
        )
        
        # Initialize phases
        self.phases = nn.Parameter(torch.zeros(self.d_model // 2))
        
        print(f"[PositionalEncoding] I got initialized successfully with geometric frequencies: min={min_freq:.5f}, max={max_freq:.5f}! :D")
        
    def forward(self, x: torch.Tensor, ops: Optional[MultivectorOps] = None) -> torch.Tensor:
        """Apply positional encoding to input tensor.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            ops: Optional operations module
            
        Returns:
            Encoded tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Generate position indices
        positions = torch.arange(seq_len, device=x.device).float()
        
        # Expand frequencies and phases for broadcasting
        freqs = self.frequencies.unsqueeze(0)  # [1, d_model//2]
        phases = self.phases.unsqueeze(0)      # [1, d_model//2]
        
        # Compute complex exponential
        pos_enc = torch.exp(1j * (freqs * positions.unsqueeze(-1) + phases))  # [seq_len, d_model//2]
        
        # Split into real and imaginary parts
        pos_enc_real = pos_enc.real  # [seq_len, d_model//2]
        pos_enc_imag = pos_enc.imag  # [seq_len, d_model//2]
        
        # Concatenate real and imaginary parts
        pos_enc = torch.cat([pos_enc_real, pos_enc_imag], dim=-1)  # [seq_len, d_model]
        
        # Add batch dimension
        pos_enc = pos_enc.unsqueeze(0)  # [1, seq_len, d_model]
        
        # Add positional encoding to input
        return x + pos_enc

class ComplexPReLU(nn.Module):
    def __init__(self, config_or_dim):
        super().__init__()
        if hasattr(config_or_dim, 'input_dim'):
            input_dim = config_or_dim.input_dim
        else:
            input_dim = int(config_or_dim)
        self.input_dim = input_dim
        self.pos_weight = nn.Parameter(torch.ones(input_dim))
        self.neg_weight = nn.Parameter(torch.ones(input_dim) * 0.25)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        r = torch.abs(z)
        phi = torch.angle(z)
        r_out = self.pos_weight * r
        return r_out * torch.exp(1j * phi)

class DynamicActivation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.alpha = nn.Parameter(torch.ones(dim) * 0.1)  # Initialize with small values
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = 1e-8  # Small epsilon for numerical stability
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dynamic activation function with numerical stability.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, dim]
            
        Returns:
            Activated tensor of same shape
        """
        # Add small epsilon to prevent division by zero
        x = x + self.eps
        
        # Compute activation with gradient clipping
        alpha = torch.clamp(self.alpha, min=0.1, max=10.0)  # Prevent extreme values
        beta = torch.clamp(self.beta, min=-10.0, max=10.0)
        
        # Apply activation
        y = torch.tanh(alpha * x + beta)
        
        # Normalize output
        y = y / (torch.abs(y).max() + self.eps)
        
        return y

class MatrixTypeSelector(nn.Module):
    """Automatically selects the best structured matrix type based on input characteristics."""
    
    def __init__(self, d_model: int, ops: MultivectorOps):
        super().__init__()
        self.d_model = d_model
        self.ops = ops
        
        # Initialize parameters for matrix type selection
        self.type_weights = nn.Parameter(torch.ones(4))  # Weights for different matrix types
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)  # Temperature for softmax
        
        # Initialize feature extractors
        self.feature_net = ComplexMLP(
            ModelConfig(input_dim=4, mlp_dim=32, output_dim=4, n_heads=1, n_layers=1, dropout=0.0),
            ops
        )
        
        # Matrix types: 0=Toeplitz, 1=BlockToeplitz, 2=Circulant, 3=Hankel
        self.matrix_types = ['toeplitz', 'block_toeplitz', 'circulant', 'hankel']
        
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input to determine matrix type."""
        # Compute various statistics
        spectral_norm = torch.norm(x, dim=(-1, -2)).mean()  # Spectral norm
        sparsity = (torch.abs(x) < 1e-3).float().mean()     # Sparsity
        symmetry = torch.abs(x - x.transpose(-1, -2)).mean() # Symmetry measure
        locality = torch.abs(x[:, :, 1:] - x[:, :, :-1]).mean() # Locality measure
        
        # Stack features
        features = torch.stack([
            spectral_norm, sparsity, symmetry, locality
        ], dim=-1).to(torch.complex64)
        
        return features
        
    def forward(self, x: torch.Tensor) -> Tuple[str, torch.Tensor]:
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
        if matrix_type == 'toeplitz':
            params = self._generate_toeplitz_params(x)
        elif matrix_type == 'block_toeplitz':
            params = self._generate_block_toeplitz_params(x)
        elif matrix_type == 'circulant':
            params = self._generate_circulant_params(x)
        else:  # hankel
            params = self._generate_hankel_params(x)
            
        return matrix_type, params
        
    def _generate_toeplitz_params(self, x: torch.Tensor) -> torch.Tensor:
        """Generate parameters for Toeplitz matrix."""
        diags = torch.zeros(self.d_model * 2 - 1, device=x.device, dtype=torch.complex64)
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
            block_size,          # Block height
            block_size,          # Block width
            device=x.device,
            dtype=torch.complex64
        )
        
        # Extract and average blocks
        for i in range(-num_blocks + 1, num_blocks):
            # Calculate valid indices for the current block
            start_idx = max(0, i)
            end_idx = min(d_model, d_model + i)
            
            if end_idx > start_idx:  # Only process if we have valid indices
                # Extract the block from the key tensor
                block = x[:, :, start_idx:end_idx]  # Shape: [batch_size, seq_len, block_size]
                
                # Average across batch and sequence dimensions
                block_avg = torch.mean(block, dim=(0, 1))  # Shape: [block_size]
                
                # Create a block matrix by repeating the average
                block_matrix = torch.zeros(block_size, block_size, device=x.device, dtype=torch.complex64)
                for j in range(block_size):
                    block_matrix[j, j:] = block_avg[:block_size-j]
                    block_matrix[j:, j] = block_avg[:block_size-j]
                
                # Store the block matrix
                blocks[i + num_blocks - 1] = block_matrix
        
        return blocks
        
    def _generate_circulant_params(self, x: torch.Tensor) -> torch.Tensor:
        """Generate parameters for circulant matrix."""
        return x[:, 0, :].mean(dim=0)
        
    def _generate_hankel_params(self, x: torch.Tensor) -> torch.Tensor:
        """Generate parameters for Hankel matrix."""
        anti_diags = torch.zeros(self.d_model * 2 - 1, device=x.device, dtype=torch.complex64)
        for i in range(-self.d_model + 1, self.d_model):
            anti_diags[i + self.d_model - 1] = torch.mean(
                torch.diagonal(x.flip(-1), offset=i)
            )
        return anti_diags

class InfiniToeplitz(nn.Module):
    def __init__(self, config: ModelConfig, ops: MultivectorOps):
        super().__init__()
        self.d_model = config.d_model
        self.eps = 1e-8
        
        # Initialize complex parameters
        self.query = nn.Parameter(torch.randn(self.d_model, self.d_model, dtype=torch.complex64) * 0.02)
        self.key = nn.Parameter(torch.randn(self.d_model, self.d_model, dtype=torch.complex64) * 0.02)
        self.value = nn.Parameter(torch.randn(self.d_model, self.d_model, dtype=torch.complex64) * 0.02)
        
        # Initialize matrix type selector
        self.type_weights = nn.Parameter(torch.ones(4))  # Weights for different matrix types
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)  # Temperature for softmax
        
        # Matrix types: 0=Toeplitz, 1=BlockToeplitz, 2=Circulant, 3=Hankel
        self.matrix_types = ['toeplitz', 'block_toeplitz', 'circulant', 'hankel']
        
        # Scale factor for attention
        self.scale = 1.0 / math.sqrt(self.d_model)
        
    def _select_matrix_type(self, x: torch.Tensor) -> Tuple[str, torch.Tensor]:
        """Select matrix type based on input characteristics."""
        # Compute features
        spectral_norm = torch.norm(x, dim=(-1, -2)).mean()
        sparsity = (torch.abs(x) < 1e-3).float().mean()
        symmetry = torch.abs(x - x.transpose(-1, -2)).mean()
        locality = torch.abs(x[:, :, 1:] - x[:, :, :-1]).mean()
        
        # Stack features
        features = torch.stack([spectral_norm, sparsity, symmetry, locality])
        
        # Get type scores
        scores = features * self.type_weights
        scores = scores / self.temperature
        probs = F.softmax(scores, dim=0)
        
        # Select matrix type
        type_idx = torch.argmax(probs).item()
        matrix_type = self.matrix_types[type_idx]
        
        # Generate parameters based on selected type
        if matrix_type == 'toeplitz':
            params = self._generate_toeplitz_params(x)
        elif matrix_type == 'block_toeplitz':
            params = self._generate_block_toeplitz_params(x)
        elif matrix_type == 'circulant':
            params = self._generate_circulant_params(x)
        else:  # hankel
            params = self._generate_hankel_params(x)
            
        return matrix_type, params
        
    def _generate_toeplitz_params(self, x: torch.Tensor) -> torch.Tensor:
        """Generate parameters for Toeplitz matrix."""
        diags = torch.zeros(self.d_model * 2 - 1, device=x.device, dtype=torch.complex64)
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
            block_size,          # Block height
            block_size,          # Block width
            device=x.device,
            dtype=torch.complex64
        )
        
        # Extract and average blocks
        for i in range(-num_blocks + 1, num_blocks):
            # Calculate valid indices for the current block
            start_idx = max(0, i)
            end_idx = min(d_model, d_model + i)
            
            if end_idx > start_idx:  # Only process if we have valid indices
                # Extract the block from the key tensor
                block = x[:, :, start_idx:end_idx]  # Shape: [batch_size, seq_len, block_size]
                
                # Average across batch and sequence dimensions
                block_avg = torch.mean(block, dim=(0, 1))  # Shape: [block_size]
                
                # Create a block matrix by repeating the average
                block_matrix = torch.zeros(block_size, block_size, device=x.device, dtype=torch.complex64)
                for j in range(block_size):
                    block_matrix[j, j:] = block_avg[:block_size-j]
                    block_matrix[j:, j] = block_avg[:block_size-j]
                
                # Store the block matrix
                blocks[i + num_blocks - 1] = block_matrix
        
        return blocks
        
    def _generate_circulant_params(self, x: torch.Tensor) -> torch.Tensor:
        """Generate parameters for circulant matrix."""
        return x[:, 0, :].mean(dim=0)
        
    def _generate_hankel_params(self, x: torch.Tensor) -> torch.Tensor:
        """Generate parameters for Hankel matrix."""
        anti_diags = torch.zeros(self.d_model * 2 - 1, device=x.device, dtype=torch.complex64)
        for i in range(-self.d_model + 1, self.d_model):
            anti_diags[i + self.d_model - 1] = torch.mean(
                torch.diagonal(x.flip(-1), offset=i)
            )
        return anti_diags
        
    def _apply_structured_matrix(self, x: torch.Tensor, matrix_type: str, params: torch.Tensor) -> torch.Tensor:
        """Apply the selected structured matrix to input."""
        if matrix_type == 'toeplitz':
            return self._apply_toeplitz(x, params)
        elif matrix_type == 'block_toeplitz':
            return self._apply_block_toeplitz(x, params)
        elif matrix_type == 'circulant':
            return self._apply_circulant(x, params)
        else:  # hankel
            return self._apply_hankel(x, params)
            
    def _apply_toeplitz(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Apply Toeplitz matrix multiplication."""
        matrix = torch.zeros(self.d_model, self.d_model, device=x.device, dtype=torch.complex64)
        for i in range(self.d_model):
            for j in range(self.d_model):
                matrix[i, j] = params[i - j + self.d_model - 1]
        return torch.matmul(x, matrix)
        
    def _apply_block_toeplitz(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Apply block Toeplitz matrix multiplication."""
        # Get input dimensions
        batch_size, seq_len, d_model = x.shape
        
        # Get block size from params
        block_size = params.shape[1]  # Block height/width
        num_blocks = d_model // block_size
        
        # Initialize output matrix
        matrix = torch.zeros(d_model, d_model, device=x.device, dtype=torch.complex64)
        
        # Fill matrix with blocks
        for i in range(num_blocks):
            for j in range(num_blocks):
                block_idx = i - j + num_blocks - 1
                if 0 <= block_idx < params.shape[0]:  # Check if block index is valid
                    matrix[i*block_size:(i+1)*block_size, 
                          j*block_size:(j+1)*block_size] = params[block_idx]
        
        return torch.matmul(x, matrix)
        
    def _apply_circulant(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Apply circulant matrix multiplication using FFT."""
        params_fft = torch.fft.fft(params)
        x_fft = torch.fft.fft(x, dim=-1)
        y_fft = x_fft * params_fft.unsqueeze(0)
        return torch.fft.ifft(y_fft, dim=-1)
        
    def _apply_hankel(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Apply Hankel matrix multiplication."""
        matrix = torch.zeros(self.d_model, self.d_model, device=x.device, dtype=torch.complex64)
        for i in range(self.d_model):
            for j in range(self.d_model):
                matrix[i, j] = params[i + j]
        return torch.matmul(x, matrix)
        
    def forward(self, x, ops=None, tokens=None):
        """Forward pass with automatic matrix type selection."""
        # Project queries, keys, and values
        q = torch.matmul(x, self.query)
        k = torch.matmul(x, self.key)
        v = torch.matmul(x, self.value)
        
        # Select matrix type and get parameters
        matrix_type, params = self._select_matrix_type(k)
        
        # Apply selected structured matrix to keys
        k = self._apply_structured_matrix(k, matrix_type, params)
        
        # Compute attention scores
        scores = torch.matmul(q, k.conj().transpose(-2, -1)) * self.scale
        
        # Use magnitude for softmax
        scores_mag = torch.abs(scores)
        scores_mag = scores_mag - scores_mag.max(dim=-1, keepdim=True)[0]
        attn_weights = torch.softmax(scores_mag, dim=-1)
        
        # Convert attention weights back to complex
        attn_weights = attn_weights.to(torch.complex64)
        
        # Apply attention weights
        output = torch.matmul(attn_weights, v)
        
        return output

class ComplexMLP(nn.Module):
    def __init__(self, config, ops):
        super().__init__()
        self.config = config
        self.ops = ops
        # Use d_model for input/output dimensions
        self.norm1 = LayerNorm(config.d_model)  # For input normalization
        self.norm2 = LayerNorm(config.d_ff)    # For intermediate normalization
        # First linear layer: d_model -> d_ff
        self.lin1a = LinearLayer(config.d_model, config.d_ff)
        self.lin1b = LinearLayer(config.d_model, config.d_ff)
        self.act = DynamicActivation(config.d_ff)
        self.res_gate = nn.Parameter(torch.randn(1, dtype=torch.complex64))
        # Second linear layer: d_ff -> d_model
        self.lin2 = LinearLayer(config.d_ff, config.d_model)
        self.drop = ComplexDropout(config.dropout)
        self.scale = nn.Parameter(torch.ones(1, dtype=torch.float32) * 0.02)
        print("[ComplexMLP] I got initialized successfully! :D")

    def forward(self, x, ops=None):
        if ops is None:
            ops = self.ops
        residual = x  # [batch_size, seq_len, d_model]
        x = self.norm1(x)  # [batch_size, seq_len, d_model]
        xa = self.lin1a(x, ops)  # [batch_size, seq_len, d_ff]
        xb = self.lin1b(x, ops)  # [batch_size, seq_len, d_ff]
        x = self.act(xa * torch.sigmoid(self.res_gate) + xb * (1 - torch.sigmoid(self.res_gate)))  # [batch_size, seq_len, d_ff]
        x = self.norm2(x)  # [batch_size, seq_len, d_ff]
        x = self.lin2(x, ops)  # [batch_size, seq_len, d_model]
        return self.drop(x * self.scale) + residual  # [batch_size, seq_len, d_model]

class TransformerLayer(nn.Module):
    def __init__(self, config: ModelConfig, ops: MultivectorOps, concept_graph=None):
        super().__init__()
        self.config = config
        self.ops = ops
        
        # Initialize components
        self.attention = InfiniToeplitz(config, ops)
        self.norm1 = LayerNorm(config.d_model)
        self.norm2 = LayerNorm(config.d_model)
        self.mlp = ComplexMLP(config, ops)
        self.dropout = ComplexDropout(config.dropout)
        self.activation = DynamicActivation(config.d_model)
        
        print("[TransformerLayer] I got initialized successfully! :D")
        
    def forward(self, x: torch.Tensor, ops: Optional[MultivectorOps] = None) -> torch.Tensor:
        """Forward pass through transformer layer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            ops: Optional operations module
            
        Returns:
            Output tensor of same shape
        """
        # Self-attention
        attn_output = self.attention(x, ops)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.mlp(x, ops)
        x = self.norm2(x + self.dropout(ff_output))
        
        # Apply dynamic activation
        x = self.activation(x)
        
        return x

class Transformer(nn.Module):
    def __init__(self, config: ModelConfig, ops: Optional[MultivectorOps] = None, concept_graph=None):
        super().__init__()
        self.config = config
        self.ops = ops or MultivectorOps()
        
        # Initialize components
        self.input_proj = ComplexInputProjection(config, self.ops)
        self.pos_encoding = PositionalEncoding(config)
        self.dropout = ComplexDropout(config.dropout)
        
        # Create transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config, self.ops, concept_graph)
            for _ in range(config.n_layers)
        ])
        
        # Final projection layer to output_dim
        self.output_proj = LinearLayer(config.d_model, config.output_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    async def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        generate_response: bool = False,
        num_tokens: int = 10
    ) -> torch.Tensor:
        """Forward pass through transformer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            generate_response: Whether to generate response
            num_tokens: Number of tokens to generate if generating response
            
        Returns:
            Output tensor of shape [batch_size, seq_len, output_dim]
        """
        # Project input
        x = self.input_proj(x, self.ops)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Process through transformer layers
        for layer in self.layers:
            x = layer(x, self.ops)
            
        # Project to output_dim
        x = self.output_proj(x, self.ops)
            
        return x

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = LinearLayer(d_model, d_ff)
        self.linear2 = LinearLayer(d_ff, d_model)
        self.activation = nn.GELU()
        self.norm = LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return self.norm(x + residual)

class ComplexLinear(nn.Module):
    """Complex-valued linear layer with geometric algebra support."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights with proper scaling for complex numbers
        # Note: weight shape is [in_features, out_features] for F.linear
        self.weight = nn.Parameter(torch.randn(in_features, out_features, dtype=torch.complex64) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.complex64)) if bias else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_complex():
            x = torch.complex(x, torch.zeros_like(x))
            
        # Handle batched inputs correctly
        if x.dim() > 2:
            batch_shape = x.shape[:-1]
            x_reshaped = x.reshape(-1, x.shape[-1])
            # F.linear expects weight shape [out_features, in_features]
            out = F.linear(x_reshaped, self.weight.t(), self.bias)
            return out.reshape(*batch_shape, -1)
        else:
            return F.linear(x, self.weight.t(), self.bias)

class ComplexLoss(nn.Module):
    """Complex-valued loss function with concept alignment."""
    
    def __init__(self, concept_graph=None, alpha=0.1, beta=0.01):
        super().__init__()
        self.concept_graph = concept_graph
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the complex loss with concept alignment."""
        # MSE loss for magnitude
        mse_loss = F.mse_loss(torch.abs(pred), torch.abs(target))
        
        # Phase regularization
        phase_reg = torch.mean(torch.abs(torch.angle(pred) - torch.angle(target)))
        
        # Initialize concept loss
        concept_loss = torch.tensor(0.0, device=pred.device)
        
        # Compute concept alignment if concept graph is available
        if self.concept_graph is not None and isinstance(self.concept_graph, dict):
            # Get concept embeddings from the graph
            concepts = self.concept_graph.get('concepts', None)
            if concepts is not None and len(concepts) > 0:
                # Convert concepts to tensor if needed
                if not isinstance(concepts, torch.Tensor):
                    concepts = torch.tensor(concepts, device=pred.device, dtype=torch.complex64)
                
                # Compute cosine similarity between predictions and concept embeddings
                pred_norm = F.normalize(pred.reshape(-1, pred.shape[-1]), dim=-1)
                concept_norm = F.normalize(concepts, dim=-1)
                similarity = torch.matmul(pred_norm, concept_norm.t())
                
                # Encourage predictions to align with relevant concepts
                concept_loss = -torch.mean(torch.max(similarity, dim=-1)[0])
        
        # Combine losses with gradient clipping
        total_loss = mse_loss + self.alpha * concept_loss + self.beta * phase_reg
        
        # Clip loss to prevent NaN
        total_loss = torch.clamp(total_loss, min=0.0, max=1e6)
        
        return total_loss
