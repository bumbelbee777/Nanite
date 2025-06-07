import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import numpy as np
import os

from datetime import timedelta
import time

from .model import SillyAI
from .config import ModelConfig, PrecisionLevel
from .utils import ChunkedMMapDataset, IterableMMapDataset, CACHE_DIR
from .ops import MultivectorOps
from .loss import ComplexLoss

class SchrödingerDataset(Dataset):
    def __init__(self, seq_len=64, potential_type='harmonic', num_samples=1000):
        self.seq_len = seq_len
        self.potential_type = potential_type
        self.num_samples = num_samples
        self.x = np.linspace(-1, 1, seq_len)

    def __len__(self):
        return self.num_samples

    def _make_sample(self):
        if self.potential_type=='harmonic':
            k = np.random.uniform(1.0,5.0)
            V = 0.5*k*self.x**2
            psi = np.exp(-np.sqrt(k)*self.x**2/2)
        else:
            V = np.zeros_like(self.x)
            psi = np.sin(np.pi*(self.x+1)/2)
        psi = psi/np.linalg.norm(psi)
        # Ensure correct shapes [seq_len, 1]
        V = V.astype(np.float32)[:,None]
        psi = psi.astype(np.float32)[:,None]
        return V, psi

    def __getitem__(self, idx):
        V, psi = self._make_sample()
        return torch.tensor(V), torch.tensor(psi)

class Trainer:
    def __init__(self, model, ops, train_dataset, val_dataset, lr=1e-3, batch_size=32, seq_len=64):
        self.model = model
        self.ops = ops
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.crit = ComplexLoss(concept_graph=model.concept_graph, alpha=0.1, beta=0.01)
        self.bs = batch_size
        self.x_tensor = torch.linspace(-1, 1, seq_len).to(self.device)
        
        # Use num_workers=0 to avoid multiprocessing issues with memory-mapped files
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=collate_fn,
            num_workers=0
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            collate_fn=collate_fn,
            num_workers=0
        )
        self.train_dataset = train_dataset
        self.scheduler = ReduceLROnPlateau(self.opt, 'min', patience=3, factor=0.5)
        self.early_stop_patience = 5

    def train(self, epochs=5):
        best_val = float('inf')
        no_imp = 0
        start_time = time.time()

        print("\033[95m✨ [SillyAI] Training started ✨\033[0m")
        print(f"\033[96m🖥️  Device: {self.device} | Epochs: {epochs} | Batch size: {self.bs}\033[0m\n")

        for ep in range(1, epochs + 1):
            epoch_start = time.time()

            # Train phase
            self.model.train()
            train_loss = 0
            num_batches = len(self.train_loader)
            for batch_idx, (V, ψ) in enumerate(self.train_loader, 1):
                self.opt.zero_grad()
                ψ_pred = self.model(V.to(self.device), self.ops)
                if not ψ.is_complex():
                    ψ = torch.complex(ψ, torch.zeros_like(ψ))
                loss = self.crit(ψ_pred, ψ.to(self.device))
                loss.backward()
                self.opt.step()
                train_loss += loss.item() * V.size(0)

                # Mini progress bar for batches
                print(f"\r\033[94m  🏃 Training batch {batch_idx}/{num_batches} "
                      f"[{'█' * int(20 * batch_idx / num_batches):20}]",
                      end='', flush=True)

            train_loss /= len(self.train_dataset)
            print()  # Newline after batch progress

            # Val phase
            val_loss = self.evaluate(self.val_loader)

            # Scheduler & early-stop
            self.scheduler.step(val_loss)
            if val_loss < best_val:
                best_val = val_loss
                no_imp = 0
                torch.save(self.model.state_dict(), 'best.pt')
                improved = True
            else:
                no_imp += 1
                improved = False

            epoch_time = time.time() - epoch_start
            elapsed = time.time() - start_time
            remaining_epochs = epochs - ep
            eta_sec = (elapsed / ep) * remaining_epochs
            lr = self.opt.param_groups[0]['lr']

            bar_len = 30
            progress = min(no_imp / self.early_stop_patience, 1.0)
            bar = f"\033[92m{'█' * int(progress * bar_len)}\033[90m{'-' * (bar_len - int(progress * bar_len))}\033[0m"

            print(f"\033[1mEpoch {ep:3d}/{epochs} \033[93m⏳{timedelta(seconds=int(epoch_time))}\033[0m "
                  f"| Train \033[91m{train_loss:.4e}\033[0m | Val \033[91m{val_loss:.4e}\033[0m | "
                  f"LR \033[96m{lr:.2e}\033[0m | ETA \033[95m{timedelta(seconds=int(eta_sec))}\033[0m\n"
                  f"  EarlyStop [{bar}] {no_imp}/{self.early_stop_patience} "
                  f"{'✅ Improved!' if improved else '❌ No improvement'}\n")

            if no_imp >= self.early_stop_patience:
                print(f"\033[91m🚫 No improvement for {no_imp} epochs — stopping early.\033[0m\n")
                break

        total_time = time.time() - start_time
        print(f"\033[92m🎉 Training complete in {timedelta(seconds=int(total_time))} 🎉\033[0m\n")

    def evaluate(self, loader):
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for V, ψ in loader:
                V, ψ = V.to(self.device), ψ.to(self.device)
                ψ_pred = self.model(V, self.ops)
                if not ψ.is_complex():
                    ψ = torch.complex(ψ, torch.zeros_like(ψ))
                loss = self.crit(ψ_pred, ψ)
                total += loss.item() * V.size(0)
        return total / len(loader.dataset)
            
    def self_learning(self, threshold=1e-3, num_samples=500):
        pool = SchrödingerDataset(seq_len=64, num_samples=num_samples)
        loader = DataLoader(pool, batch_size=self.bs, collate_fn=collate_fn)
        
        pseudo_data = []
        self.model.eval()
        with torch.no_grad():
            for V, _ in loader:
                V = V.to(self.device)
                ψ_pred = self.model(V, self.ops).detach()
                E_est = estimate_energy(ψ_pred, V).unsqueeze(-1)
                res = physics_loss(ψ_pred, V, self.x_tensor, E_est)

                if res.item() < threshold:
                    for v, ψ in zip(V, ψ_pred):
                        pseudo_data.append((v.cpu(), ψ.cpu()))

        if not pseudo_data:
            print("No high-confidence pseudo-labels found.")
            return

        # Fine-tune on augmented data
        true_dataset = SchrödingerDataset(seq_len=64, num_samples=2000)
        augmented_dataset = torch.utils.data.ConcatDataset([
            true_dataset,
            torch.utils.data.TensorDataset(*zip(*pseudo_data))
        ])
        aug_loader = DataLoader(augmented_dataset, batch_size=self.bs, shuffle=True, collate_fn=collate_fn)
        print("Starting fine-tuning on augmented dataset...")
        self.train_loader = aug_loader
        self.train(epochs=10)  # fine-tune for fewer epochs

def physics_loss(ψ_pred, V, x, E_est):
    # second spatial derivative via finite differences
    ψ = ψ_pred.squeeze(-1)  # [B, seq_len]
    d2ψ = ψ[:,:,2:] - 2*ψ[:,:,1:-1] + ψ[:,:,:-2]
    # align shapes: drop boundaries
    V_mid = V.squeeze(-1)[:,:,1:-1]
    ψ_mid=ψ[:,:,1:-1]
    resid = -0.5*d2ψ + V_mid*ψ_mid - E_est[:,:,None]*ψ_mid
    boundary_loss = ψ[:, :, [0, -1]].abs().mean()
    return torch.mean(resid.abs()**2) + 0.5 * boundary_loss

def estimate_energy(ψ: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Estimate the energy of a wavefunction ψ under potential V using
    the expectation value of the Hamiltonian.
    
    Args:
        ψ: [B, 1, N] tensor, wavefunction predictions
        V: [B, 1, N] tensor, potential energy function
    
    Returns:
        E_est: [B, 1] tensor, estimated energy per sample
    """
    # Finite difference: second derivative approximation (central difference)
    d2ψ = ψ[:, :, 2:] - 2 * ψ[:, :, 1:-1] + ψ[:, :, :-2]
    ψ_mid = ψ[:, :, 1:-1]        # drop boundaries
    V_mid = V[:, :, 1:-1]        # drop boundaries

    # Hamiltonian applied to ψ
    Hψ = -0.5 * d2ψ + V_mid * ψ_mid

    # Compute expectation value: ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩
    num = torch.sum(ψ_mid * Hψ, dim=2)  # [B, 1]
    denom = torch.sum(ψ_mid * ψ_mid, dim=2)  # [B, 1]

    E_est = num / denom  # [B, 1]
    return E_est

def collate_fn(batch):
    """Custom collate function that handles both in-memory and memory-mapped data"""
    if isinstance(batch[0][0], np.ndarray):
        # Memory-mapped numpy arrays
        Vs, psis = zip(*batch)
        V = torch.from_numpy(np.stack(Vs))
        psi = torch.from_numpy(np.stack(psis))
    else:
        # Regular PyTorch tensors
        Vs, psis = zip(*batch)
        V = torch.stack(Vs)
        psi = torch.stack(psis)
    
    # Ensure correct shapes [batch_size, seq_len, 1]
    if V.dim() == 2:
        V = V.unsqueeze(-1)
    if psi.dim() == 2:
        psi = psi.unsqueeze(-1)
        
    return V, psi

def main():
    print('[SillyAI] Version Alpha - Initializing')
    
    # Use memory-mapped datasets for efficient data loading
    print("\033[96m📂 Loading memory-mapped datasets from cache...\033[0m")
    try:
        train_dataset = ChunkedMMapDataset(cache_dir=CACHE_DIR)
        # Use 20% of chunks for validation
        val_chunks = os.path.join(CACHE_DIR, 'val')
        os.makedirs(val_chunks, exist_ok=True)
        val_dataset = ChunkedMMapDataset(cache_dir=val_chunks)
        print(f"\033[92m✓ Loaded {len(train_dataset)} training and {len(val_dataset)} validation samples\033[0m")
    except Exception as e:
        print("\033[93m⚠️  Memory-mapped datasets not found, falling back to on-the-fly generation\033[0m")
        train_dataset = SchrödingerDataset(seq_len=64, num_samples=2000)
        val_dataset = SchrödingerDataset(seq_len=64, num_samples=500)

    config = ModelConfig(
        input_dim=1,       # Input dimension (single value per timestep)
        output_dim=1,      # Output dimension (single wavefunction value)
        num_heads=8,       # Number of attention heads
        num_layers=4,      # Number of transformer layers
        d_model=64,        # Model dimension
        d_ff=256,          # Feed-forward dimension
        max_seq_len=64,    # Maximum sequence length
        dropout=0.1,       # Dropout rate
        device='cpu',      # Device to run on
        precision=PrecisionLevel.FP8,  # Use FP8 precision
        concept_graph_size=1000,  # Maximum number of concepts
        enabled_plugins=[]  # No longer needed as plugins are built-in
    )
    
    # Initialize ops and model
    ops = MultivectorOps()
    model = SillyAI(config, ops=ops)
    
    # Add shape validation
    def validate_shapes(V, ψ):
        batch_size = V.size(0)
        seq_len = V.size(1)
        if V.size(2) != 1 or ψ.size(2) != 1:
            raise ValueError(f"Expected input and output to have dimension 1, got V: {V.size(2)}, ψ: {ψ.size(2)}")
        if seq_len != config.max_seq_len:
            raise ValueError(f"Expected sequence length {config.max_seq_len}, got {seq_len}")
        return V, ψ
    
    # Update collate_fn to ensure correct shapes
    def collate_fn(batch):
        """Custom collate function that handles both in-memory and memory-mapped data"""
        if isinstance(batch[0][0], np.ndarray):
            # Memory-mapped numpy arrays
            Vs, psis = zip(*batch)
            V = torch.from_numpy(np.stack(Vs))
            psi = torch.from_numpy(np.stack(psis))
        else:
            # Regular PyTorch tensors
            Vs, psis = zip(*batch)
            V = torch.stack(Vs)
            psi = torch.stack(psis)
        
        # Ensure correct shapes [batch_size, seq_len, 1]
        if V.dim() == 2:
            V = V.unsqueeze(-1)
        if psi.dim() == 2:
            psi = psi.unsqueeze(-1)
            
        return validate_shapes(V, psi)
    
    trainer = Trainer(model, ops, train_dataset, val_dataset, lr=1e-3, batch_size=32, seq_len=64)
    
    # Load existing weights if found
    if os.path.exists("sillyai_model.pt.gz"):
        try:
            if model.load("sillyai_model.pt.gz"):
                print("\033[92m💾 Loaded full model snapshot\033[0m")
            else:
                print("\033[91m⚠️ Couldn't load model\033[0m")
        except Exception as e:
            print(f"\033[91m⚠️ Critical loading error: {e}\033[0m")

    # Train the model
    trainer.train(epochs=10)
    
    # Print top concepts
    model.print_concepts(top_k=10)

    # Generate and print example bytecode
    bytecode = model.generate_bytecode()
    print("[SillyAI] Example bytecode from concept graph:")
    for idx, (op, args) in enumerate(bytecode):
        print(f"  {idx:03d}: {op} {args}")

if __name__ == "__main__":
    main()