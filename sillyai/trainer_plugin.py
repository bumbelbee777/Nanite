from .plugin import SillyPlugin
from .utils import ChunkedMMapDataset, CACHE_DIR
from .config import ModelConfig
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

    def setup(self, model, train_dataset=None, val_dataset=None, batch_size=32, seq_len=64, lr=1e-3, epochs=10, early_stop=5):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.lr = lr
        self.epochs = epochs
        self.early_stop_patience = early_stop
        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.crit = nn.MSELoss()
        self.x_tensor = torch.linspace(-1, 1, seq_len).to(self.device)
        if train_dataset is not None:
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.collate_fn)
        if val_dataset is not None:
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=self.collate_fn)

    def train(self, epochs=None):
        if epochs is not None:
            self.epochs = epochs
        best_val = float('inf')
        no_imp = 0
        start_time = time.time()
        print("\033[95m✨ [SillyAI] Training started (plugin) ✨\033[0m")
        print(f"\033[96m🖥️  Device: {self.device} | Epochs: {self.epochs} | Batch size: {self.batch_size}\033[0m\n")
        for ep in range(1, self.epochs + 1):
            epoch_start = time.time()
            self.model.train()
            train_loss = 0
            num_batches = len(self.train_loader)
            for batch_idx, batch in enumerate(self.train_loader, 1):
                V, ψ = batch
                self.opt.zero_grad()
                ψ_pred = self.model(V.to(self.device))
                loss = self.crit(torch.abs(ψ_pred), torch.abs(ψ.to(self.device)))
                loss.backward()
                self.opt.step()
                train_loss += loss.item() * V.size(0)
                if self.verbose:
                    print(f"\r\033[94m  🏃 Training batch {batch_idx}/{num_batches} "
                          f"[{'█' * int(20 * batch_idx / num_batches):20}]",
                          end='', flush=True)
            train_loss /= len(self.train_loader.dataset)
            print()  # Newline after batch progress
            val_loss = self.evaluate()
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
            remaining_epochs = self.epochs - ep
            eta_sec = (elapsed / ep) * remaining_epochs
            lr = self.opt.param_groups[0]['lr']
            bar_len = 30
            progress = min(no_imp / self.early_stop_patience, 1.0)
            bar = f"\033[92m{'█' * int(progress * bar_len)}\033[90m{'-' * (bar_len - int(progress * bar_len))}\033[0m"
            print(f"\033[1mEpoch {ep:3d}/{self.epochs} \033[93m⏳{timedelta(seconds=int(epoch_time))}\033[0m "
                  f"| Train \033[91m{train_loss:.4e}\033[0m | Val \033[91m{val_loss:.4e}\033[0m | "
                  f"LR \033[96m{lr:.2e}\033[0m | ETA \033[95m{timedelta(seconds=int(eta_sec))}\033[0m\n"
                  f"  EarlyStop [{bar}] {no_imp}/{self.early_stop_patience} "
                  f"{'✅ Improved!' if improved else '❌ No improvement'}\n")
            if no_imp >= self.early_stop_patience:
                print(f"\033[91m🚫 No improvement for {no_imp} epochs — stopping early.\033[0m\n")
                break
        total_time = time.time() - start_time
        print(f"\033[92m🎉 Training complete in {timedelta(seconds=int(total_time))} 🎉\033[0m\n")

    def evaluate(self):
        self.model.eval()
        total = 0.0
        with torch.no_grad():
            for V, ψ in self.val_loader:
                V, ψ = V.to(self.device), ψ.to(self.device)
                ψ_pred = self.model(V)
                loss = self.crit(torch.abs(ψ_pred), torch.abs(ψ))
                total += loss.item() * V.size(0)
        return total / len(self.val_loader.dataset)

    def collate_fn(self, batch):
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
