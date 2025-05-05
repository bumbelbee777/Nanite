import torch
import torch.nn.functional as F
import numpy as np

from .config import ModelConfig
from .model import SillyAI
from .ops import MultivectorOps, tensor_to_multivector

# Utility to generate complex-valued 2D input grid
def generate_complex_grid(n: int = 100):
    re = torch.linspace(-1, 1, n)
    im = torch.linspace(-1, 1, n)
    Xr, Xi = torch.meshgrid(re, im, indexing='ij')
    z = Xr + 1j * Xi
    z = z.reshape(-1)  # Flatten
    return z  # [N^2] complex

# Converts complex tensor to multivector input for SillyAI
def complex_to_mv(z: torch.Tensor, ops: MultivectorOps):
    # z is complex tensor [N], e.g. z = x + iy
    x = z.real
    y = z.imag
    input_tensor = torch.stack([x, y], dim=1)  # [N, 2]
    # convert to multivector: you may want more sophisticated lifting here
    return tensor_to_multivector(input_tensor, n_dims=2)  # [N, mv_dim]

# Compile user-defined string into a PyTorch-evaluable function
def compile_function(expr: str):
    def f(z: torch.Tensor) -> torch.Tensor:
        # z is [N] complex tensor
        x = z.real
        y = z.imag
        # Safe namespace
        allowed = {
            'torch': torch, 'np': np, 'x': x, 'y': y, 'z': z,
            'abs': torch.abs, 'exp': torch.exp, 'sin': torch.sin, 'cos': torch.cos,
            'log': torch.log, 'tan': torch.tan
        }
        try:
            result = eval(expr, {"__builtins__": {}}, allowed)
        except Exception as e:
            raise ValueError(f"Invalid function: {e}")
        return result
    return f

# Training loop for SillyAI to fit f(z)
def train_sillyai(model, z, fz, steps=1000, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for step in range(steps):
        optimizer.zero_grad()
        pred = model(z.unsqueeze(0))  # [1, N, mv_dim]
        loss = F.mse_loss(pred, fz)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == steps - 1:
            print(f"[Step {step}] Loss: {loss.item():.6f}")
    return model

def start_repl():
    print("🧠 SillyAI Function Approximation REPL")
    print("Enter a complex function using 'z'. Example: sin(z) + i*z**2")
    print("Type 'quit' or 'exit' to stop.")
    
    # Ops and model setup
    ops = MultivectorOps()
    config = ModelConfig(
        dim=2,
        num_heads=2,
        mlp_dim=128,
        num_layers=4,
        dropout=0.1,
        precision='float16',
        cache_max_bytes=1 << 26,
        decomp_threshold=1_000_000,
        decomp_gain_ratio=0.5,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        modalities=None
    )
    model = SillyAI(config, num_classes=2)

    model_dtype = torch.complex64 if config.dtype == "complex64" else torch.complex128
    model_device = torch.device(config.device)
    while True:
        try:
            expr = input("f(z) = ").strip()
            if expr.lower() in {"exit", "quit"}:
                break

            func = compile_function(expr)

            # Input: z ∈ [-1 - i, 1 + i]
            z = generate_complex_grid(100).to(torch.complex64)  # [N]
            target = func(z).to(torch.complex64)  # [N]

            # Target: real/imag split
            fz = torch.stack([target.real, target.imag], dim=1)  # [N, 2]
            fz = fz.to(dtype=torch.float32, device=model_device)

            # Input multivectors
            z_mv = complex_to_mv(z, ops)  # [N, mv_dim]
            z_mv = z_mv.to(dtype=model_dtype, device=model_device)

            print(f"Training model to approximate: f(z) = {expr}")
            train_sillyai(model, z_mv, fz, steps=1000)

        except Exception as e:
            print(f"[Error] {e}")

if __name__ == "__main__":
    start_repl()
