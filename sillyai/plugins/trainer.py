from ..plugin import SillyPlugin
from ..utils import ChunkedMMapDataset, CACHE_DIR
from ..config import ModelConfig
from ..loss import ComplexLoss
from .visualizer import ModelProfiler, WavefunctionVisualizer, load_best_model
import torch
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
import torch.nn as nn
import numpy as np
import os
import time
from datetime import timedelta

class SillyAITrainerPlugin(SillyPlugin):
    """
    SillyAI plugin for training, evaluation, and self-learning.
    Encapsulates all training logic previously in __main__.py.
    """
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.train_loader = None
        self.val_loader = None
        self.device = None
        self.early_stop_patience = 5
        self.batch_size = 32
        self.seq_len = 64
        self.lr = 1e-3
        self.epochs = 10
        self.verbose = True
        self.ops = None
        self.profiler = None
        self.visualizer = None
        self.ψ_history = []  # Store wavefunction history for animation
        self.weight_decay = 0.0  # Initialize weight decay parameter

    def setup(self, model, ops, train_dataset=None, val_dataset=None, batch_size=32, 
             seq_len=64, lr=1e-3, epochs=10, early_stop=5, weight_decay=0.0):
        """Setup trainer with model and datasets."""
        self.model = model
        self.ops = ops
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.epochs = epochs
        self.early_stop = early_stop
        self.weight_decay = weight_decay
        
        # Create data loaders
        if train_dataset:
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=self.collate_fn
            )
            
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.collate_fn
            )
            
        # Initialize optimizer with weight decay
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Initialize custom loss function
        self.criterion = ComplexLoss(
            concept_graph=model.concept_graph,
            alpha=0.1,  # Weight for concept graph regularization
            beta=0.01   # Weight for complex number regularization
        )
        
        # Log setup configuration
        print("\n[SillyAITrainerPlugin] Setup configuration:")
        print(f"  Device: {next(model.parameters()).device}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {lr}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Early stop patience: {early_stop}")
        if train_dataset:
            print(f"  Training samples: {len(train_dataset)}")
        if val_dataset:
            print(f"  Validation samples: {len(val_dataset)}")
            
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (V, psi) in enumerate(self.train_loader):
            # Move data to device
            V = V.to(self.model.device)
            psi = psi.to(self.model.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            psi_pred = self.model(V)
            
            # Calculate loss using complex loss
            loss = self.criterion(psi_pred, psi)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 10 == 0:
                print(f"\rBatch {batch_idx}/{len(self.train_loader)} - Loss: {loss.item():.6f}", end="")
                
        print()  # New line after progress
        return total_loss / num_batches if num_batches > 0 else 0
        
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for V, psi in self.val_loader:
                # Move data to device
                V = V.to(self.model.device)
                psi = psi.to(self.model.device)
                
                # Forward pass
                psi_pred = self.model(V)
                
                # Calculate loss using complex loss
                loss = self.criterion(psi_pred, psi)
                
                # Update statistics
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches if num_batches > 0 else 0

    def train(self, epochs=None):
        if epochs is not None:
            self.epochs = epochs
        best_val = float('inf')
        no_imp = 0
        start_time = time.time()
        print("\033[95m✨ [SillyAI] Training started (plugin) ✨\033[0m")
        print(f"\033[96m🖥️  Device: {self.device} | Epochs: {self.epochs} | Batch size: {self.batch_size}\033[0m\n")
        
        for ep in range(1, self.epochs + 1):
            self.profiler.start_epoch()
            epoch_start = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0
            num_batches = len(self.train_loader)
            for batch_idx, (V, ψ) in enumerate(self.train_loader, 1):
                self.profiler.start_batch()
                self.optimizer.zero_grad()
                ψ_pred = self.model(V.to(self.device))
                if not ψ.is_complex():
                    ψ = torch.complex(ψ, torch.zeros_like(ψ))
                loss = self.criterion(ψ_pred, ψ.to(self.device))
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * V.size(0)
                self.profiler.end_batch()
                
                if self.verbose:
                    print(f"\r\033[94m  🏃 Training batch {batch_idx}/{num_batches} "
                          f"[{'█' * int(20 * batch_idx / num_batches):20}]",
                          end='', flush=True)
            
            train_loss /= len(self.train_loader.dataset)
            print()  # Newline after batch progress
            
            # Validation phase
            val_loss = self.validate()
            
            # Update learning rate based on validation loss
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f"\n\033[93m📉 Reducing learning rate from {old_lr:.2e} to {new_lr:.2e}\033[0m")
            
            # Inference test
            test_loss = self.run_inference_test()
            
            # Update profiler with metrics
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'test_loss': test_loss,
                'learning_rate': self.optimizer.param_groups[0]['lr']
            }
            self.profiler.end_epoch(ep, metrics)
            
            # Plot metrics
            self.profiler.plot_metrics(f"profiles/metrics_epoch_{ep}.png")
            
            # Early stopping check with relative improvement threshold
            if val_loss < best_val * 0.999:  # Require at least 0.1% improvement
                best_val = val_loss
                no_imp = 0
                torch.save(self.model.state_dict(), f'best_epoch_{ep}.pt')
                improved = True
            else:
                no_imp += 1
                improved = False
                
            # Progress reporting
            epoch_time = time.time() - epoch_start
            elapsed = time.time() - start_time
            remaining_epochs = self.epochs - ep
            eta_sec = (elapsed / ep) * remaining_epochs
            lr = self.optimizer.param_groups[0]['lr']
            
            bar_len = 30
            progress = min(no_imp / self.early_stop_patience, 1.0)
            bar = f"\033[92m{'█' * int(progress * bar_len)}\033[90m{'-' * (bar_len - int(progress * bar_len))}\033[0m"
            
            print(f"\033[1mEpoch {ep:3d}/{self.epochs} \033[93m⏳{timedelta(seconds=int(epoch_time))}\033[0m "
                  f"| Train \033[91m{train_loss:.4e}\033[0m | Val \033[91m{val_loss:.4e}\033[0m | "
                  f"Test \033[91m{test_loss:.4e}\033[0m | LR \033[96m{lr:.2e}\033[0m | "
                  f"ETA \033[95m{timedelta(seconds=int(eta_sec))}\033[0m\n"
                  f"  EarlyStop [{bar}] {no_imp}/{self.early_stop_patience} "
                  f"{'✅ Improved!' if improved else '❌ No improvement'}\n")
            
            if no_imp >= self.early_stop_patience:
                print(f"\033[91m🚫 No improvement for {no_imp} epochs — stopping early.\033[0m\n")
                break
                
        total_time = time.time() - start_time
        print(f"\033[92m🎉 Training complete in {timedelta(seconds=int(total_time))} 🎉\033[0m\n")
        
        # Create final animation of wavefunction evolution
        self.visualizer.create_animation(
            self.x_tensor,
            torch.zeros_like(self.x_tensor),  # Placeholder for potential
            self.ψ_history,
            "visualizations/wavefunction_evolution.gif"
        )

    def run_inference_test(self):
        """Run inference test on a small batch of data."""
        self.model.eval()
        with torch.no_grad():
            # Generate test data
            x = torch.linspace(-1, 1, self.seq_len).unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
            V = 0.5 * x**2  # Simple harmonic potential
            ψ = torch.exp(-x**2/2)  # Ground state of harmonic oscillator
            ψ = ψ / torch.norm(ψ, dim=1, keepdim=True)
            
            # Run inference
            V = V.to(self.device)
            ψ = ψ.to(self.device)
            ψ_pred = self.model(V)
            
            # Store prediction for animation
            self.ψ_history.append(ψ_pred[0])
            
            # Visualize wavefunction
            self.visualizer.plot_wavefunction(
                x[0], V[0], ψ[0], ψ_pred[0],
                epoch=len(self.ψ_history),
                save=True
            )
            
            # Compute loss
            if not ψ.is_complex():
                ψ = torch.complex(ψ, torch.zeros_like(ψ))
            loss = self.criterion(ψ_pred, ψ)
            
            # Print sample prediction
            print("\n\033[95m📊 Sample Prediction:\033[0m")
            print(f"Input shape: {V.shape}")
            print(f"Output shape: {ψ_pred.shape}")
            print(f"Max amplitude: {torch.abs(ψ_pred).max().item():.4f}")
            print(f"Mean phase: {torch.angle(ψ_pred).mean().item():.4f}\n")
            
            return loss.item()

    def collate_fn(self, batch):
        """Custom collate function that handles both in-memory and memory-mapped data."""
        if not batch:
            return torch.Tensor(), torch.Tensor()
            
        if isinstance(batch[0], (list, tuple)) and len(batch[0]) == 2:
            Vs, psis = zip(*batch)
        else:
            Vs, psis = batch, batch
            
        def ensure_tensor(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            elif torch.is_tensor(x):
                return x
            else:
                return torch.tensor(x)
                
        V_batch = torch.stack([ensure_tensor(V) for V in Vs])
        psi_batch = torch.stack([ensure_tensor(psi) for psi in psis])
        
        # Ensure correct shapes [batch_size, seq_len, 1]
        if V_batch.dim() == 2:
            V_batch = V_batch.unsqueeze(-1)
        if psi_batch.dim() == 2:
            psi_batch = psi_batch.unsqueeze(-1)
            
        return V_batch, psi_batch

    def on_init(self, model):
        super().on_init(model)
        print(f"[SillyAITrainerPlugin] Initialized for model: {type(model).__name__}")

    def on_enable(self):
        super().on_enable()
        print("[SillyAITrainerPlugin] Enabled!")

    def on_disable(self):
        super().on_disable()
        print("[SillyAITrainerPlugin] Disabled!")

    def on_epoch_start(self, epoch):
        print(f"[SillyAITrainerPlugin] Epoch {epoch} started.")

    def on_epoch_end(self, epoch, metrics):
        print(f"[SillyAITrainerPlugin] Epoch {epoch} ended. Metrics: {metrics}")
