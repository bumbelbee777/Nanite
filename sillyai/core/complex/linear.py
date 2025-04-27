import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings

class ComplexLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        factorized: bool = False,
        kronecker_rank: int | None = None,
    ):
        super().__init__()
        
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                f"Input and output features must be positive, got "
                f"in_features={in_features}, out_features={out_features}"
            )
        
        self.in_features = in_features
        self.out_features = out_features
        self.factorized = factorized

        if factorized:
            if kronecker_rank is None:
                raise ValueError("kronecker_rank must be provided when factorized=True")
            
            # Enforce valid Kronecker rank
            max_rank = min(in_features, out_features)
            if kronecker_rank > max_rank:
                warnings.warn(
                    f"Kronecker rank {kronecker_rank} exceeds min(in, out)={max_rank}. "
                    f"Using {max_rank} instead."
                )
                kronecker_rank = max_rank
            
            self.kronecker_rank = kronecker_rank
            self.U_real = nn.Parameter(torch.empty(out_features, kronecker_rank))
            self.V_real = nn.Parameter(torch.empty(kronecker_rank, in_features))
            self.U_imag = nn.Parameter(torch.empty(out_features, kronecker_rank))
            self.V_imag = nn.Parameter(torch.empty(kronecker_rank, in_features))
            
            # Xavier initialization with complex variance scaling
            scale = math.sqrt(2.0 / (in_features + out_features))
            for param in [self.U_real, self.V_real, self.U_imag, self.V_imag]:
                nn.init.xavier_uniform_(param, gain=scale)
        else:
            self.weight_real = nn.Parameter(torch.empty(out_features, in_features))
            self.weight_imag = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.xavier_uniform_(self.weight_real)
            nn.init.xavier_uniform_(self.weight_imag)

        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_features))
            self.bias_imag = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input validation
        if x.dim() < 2:
            raise ValueError(f"Input must have at least 2 dimensions, got shape {x.shape}")
        if x.shape[-1] != 2:
            raise ValueError(
                f"Last dimension must be 2 (real/imag components), got shape {x.shape}"
            )
        if x.shape[-2] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} input features but got {x.shape[-2]}"
            )

        # Collapse all leading dimensions except last two
        original_shape = x.shape
        batch_dims = original_shape[:-2]
        x_flat = x.view(-1, self.in_features, 2)  # [batch, in_features, 2]

        # Split real/imag components
        x_r = x_flat[..., 0]  # [batch, in_features]
        x_i = x_flat[..., 1]  # [batch, in_features]

        # Compute weights
        if self.factorized:
            Wr = (self.U_real @ self.V_real) - (self.U_imag @ self.V_imag)
            Wi = (self.U_real @ self.V_imag) + (self.U_imag @ self.V_real)
        else:
            Wr, Wi = self.weight_real, self.weight_imag

        # Complex linear transformation
        out_r = F.linear(x_r, Wr) - F.linear(x_i, Wi)
        out_i = F.linear(x_r, Wi) + F.linear(x_i, Wr)

        # Add bias
        if self.bias_real is not None:
            out_r = out_r + self.bias_real
            out_i = out_i + self.bias_imag

        # Combine and reshape to original batch dimensions
        out_flat = torch.stack([out_r, out_i], dim=-1)  # [batch, out_features, 2]
        return out_flat.view(*batch_dims, self.out_features, 2)