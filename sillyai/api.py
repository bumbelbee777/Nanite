import torch
import torch.nn as nn
import torch.functional as F
from typing import List, Dict, Union
import gzip
import onnx
import os
from datetime import datetime, timedelta
import time

from .core import Transformer, TaskComplexityEstimator, FeatureRouter, complex_checkpoint
from .ops import MultivectorOps
from .concept import ConceptGraph, Concept
from .vm import BytecodeProgram, BytecodeEngine
from .config import ModelConfig
from .plugin import PluginManager

class SillyAI(nn.Module):
    def __init__(self, config: ModelConfig, device=None, concept_graph=None, path="sillyai_model.pt"):
        super().__init__()
        self.config = config
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.concept_graph = concept_graph or ConceptGraph()
        
        # Initialize member variables
        self.best_loss = float('inf')
        self.current_epoch = 0
        self.path = path
        
        # Initialize plugin system
        self.plugin_manager = PluginManager(self)
        
        # Initialize complexity estimator and router
        self.complexity_estimator = TaskComplexityEstimator(
            threshold=1e-3,
            eps=1e-6,
            feature_dim=32,
            use_fft=True
        )
        self.feature_router = FeatureRouter(
            d_model=config.input_dim,
            hidden_dim=config.input_dim // 4,
            threshold=0.5,
            estimator=self.complexity_estimator,
        )
    
        # Create transformer
        self.transformer = Transformer(
            num_layers=config.num_layers,
            embed_dim=config.input_dim,
            num_heads=config.num_heads,
            mlp_dim=config.mlp_dim,
            output_dim=config.output_dim,
            concept_graph=self.concept_graph
        )
        self.transformer.to(self.device)

        # Setup training components with mixed precision
        self.scaler = torch.amp.GradScaler()
        self.ops = MultivectorOps().compile()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        # Initialize caches
        self.activation_cache = {}
        self.forward_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        print(f"[SillyAI] Initialized on {self.device} with config: {config}! :D")
        
    def forward(self, x, tokens=None):
        x = x.to(self.device)
        if tokens is not None:
            tokens = tokens.to(self.device)
            
        # Check forward cache first
        cache_key = self._get_cache_key(x)
        if cache_key in self.forward_cache:
            self.cache_hits += 1
            return self.forward_cache[cache_key]
        self.cache_misses += 1
            
        # Plugin hook: before_forward
        x = self.plugin_manager.call_hook('before_forward', x)
        
        # Get complexity score and route features
        with torch.cuda.amp.autocast(enabled=True):
            complexity_score = self.complexity_estimator(x)
            route_decision = self.feature_router(x)
            
            # Use dynamic features based on complexity
            use_concepts = route_decision.mean() > 0.6
            use_bytecode = complexity_score.mean() > 0.7
            
            # Main forward pass with appropriate optimizations
            if use_concepts and use_bytecode:
                # High complexity: use all advanced features
                output = complex_checkpoint(
                    self.transformer.forward,
                    x, self.ops, tokens
                )
            else:
                # Lower complexity: basic pass with caching
                output = self.transformer(x, self.ops, tokens=None)  # Skip token processing
                
            # Cache result if not too complex
            if complexity_score.mean() < 0.5:
                self.forward_cache[cache_key] = output
                
            # Trim cache if needed
            if len(self.forward_cache) > 1000:
                # Remove oldest entries
                while len(self.forward_cache) > 900:  # Keep 900 most recent
                    self.forward_cache.pop(next(iter(self.forward_cache)))
        
        # Plugin hook: after_forward
        return self.plugin_manager.call_hook('after_forward', output)
        
    def _get_cache_key(self, x):
        """Generate cache key for input tensor"""
        if torch.is_tensor(x):
            # Use shape, dtype, and first/last few elements
            key_parts = [
                str(x.shape),
                str(x.dtype),
                str(x[:2].cpu().numpy().tobytes()),  # First 2 elements
                str(x[-2:].cpu().numpy().tobytes())  # Last 2 elements
            ]
            return hash(''.join(key_parts))
        return None
    
    def train_step(self, x, y, tokens=None):
        # REMOVE self.train() call here
        self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
        
        # Forward pass with mixed precision and plugin hooks
        with torch.cuda.amp.autocast(enabled=True):
            y_pred = self.forward(x, tokens=tokens)
            loss = self.criterion(torch.abs(y_pred), torch.abs(y.to(self.device)))
            
        # Plugin hooks for backward pass
        loss = self.plugin_manager.call_hook('before_backward', loss)
        
        # Scale loss and backward pass
        self.scaler.scale(loss).backward()
        self.plugin_manager.call_hook('after_backward')
        
        # Plugin hooks and optimizer step with gradient scaling
        self.plugin_manager.call_hook('before_optimize')
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.plugin_manager.call_hook('after_optimize')
        
        return loss.item()

    def evaluate(self, loader):
        # REMOVE self.eval() call here
        total = 0.0
        n = 0
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 3:
                    x, y, tokens = batch
                else:
                    x, y = batch
                    tokens = None
                y_pred = self.forward(x, tokens=tokens)
                loss = self.criterion(torch.abs(y_pred), torch.abs(y.to(self.device)))
                total += loss.item() * x.size(0)
                n += x.size(0)
        return total / max(n, 1)

    def fit(self, train_loader, val_loader=None, epochs=5, early_stop=5, lr_scheduler=None, verbose=True):
        best_val = float('inf')
        no_imp = 0
        start_time = time.time()
        
        # ADD explicit mode setting at start
        self.transformer.train()
        
        print("\n[SillyAI] Starting training...\n")
        
        for ep in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Plugin hook: epoch start
            self.plugin_manager.call_hook('on_epoch_start', ep)
            
            # Training phase
            train_loss = 0
            for batch in train_loader:
                if len(batch) == 3:
                    x, y, tokens = batch
                else:
                    x, y = batch
                    tokens = None
                train_loss += self.train_step(x, y, tokens=tokens) * x.size(0)
            train_loss /= len(train_loader.dataset)
            
            # Validation phase
            val_loss = None
            if val_loader is not None:
                self.transformer.eval()
                val_loss = self.evaluate(val_loader)
                self.transformer.train()
                
                if lr_scheduler is not None:
                    lr_scheduler.step(val_loss)
                if val_loss < best_val:
                    best_val = val_loss
                    no_imp = 0
                    self.save(loss=val_loss, epoch=ep)
                else:
                    no_imp += 1
            
            # Plugin hook: epoch end with metrics
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val': best_val,
                'no_improvement': no_imp
            }
            self.plugin_manager.call_hook('on_epoch_end', ep, metrics)
            
            # Time computations
            epoch_time = time.time() - epoch_start
            elapsed_time = time.time() - start_time
            remaining_epochs = epochs - ep
            eta_sec = (elapsed_time / ep) * remaining_epochs
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            if verbose:
                val_loss_str = f"{val_loss:.4e}" if val_loss is not None else 'N/A'
                print(f"\033[1mEpoch {ep:3d}/{epochs} \033[93m⏳{timedelta(seconds=int(epoch_time))}\033[0m "
                      f"| Train \033[91m{train_loss:.4e}\033[0m | Val \033[91m{val_loss_str}\033[0m | "
                      f"LR \033[96m{current_lr:.2e}\033[0m | ETA \033[95m{timedelta(seconds=int(eta_sec))}\033[0m")
            
            if no_imp >= early_stop:
                print("\n[SillyAI] Early stopping triggered!")
                break
                
        total_time = time.time() - start_time
        print(f"\n[SillyAI] Training completed in {timedelta(seconds=int(total_time))} ✨")

    def save_state(self, path=None, compress=True):
        """Save just the model state dict."""
        save_path = path or self.path
        try:
            if compress:
                with gzip.open(save_path + '.gz', 'wb') as f:
                    torch.save(self.state_dict(), f)
                print(f"[SillyAI] Saved compressed state to {save_path}.gz")
            else:
                torch.save(self.state_dict(), save_path)
                print(f"[SillyAI] Saved state to {save_path}")
            return save_path
        except Exception as e:
            print(f"[SillyAI] Failed to save state: {str(e)}")
            return None

    def save(self, path=None, loss=None, epoch=None, compress=True):
        """Save full model snapshot with metadata without recursion"""
        save_path = path or self.path
        
        # Create snapshot directory if needed
        snapshot_dir = os.path.dirname(save_path) or '.'
        os.makedirs(snapshot_dir, exist_ok=True)
        
        # Prepare snapshot data - avoid using self.state_dict()
        snapshot = {
            # Save transformer state directly
            'transformer_state': self.transformer.state_dict(),
            'config': self.config.__dict__,
            'best_loss': loss or self.best_loss,
            'current_epoch': epoch or self.current_epoch,
            'concept_graph': self.concept_graph,
            'plugin_states': self.plugin_manager.save_states() if hasattr(self, 'plugin_manager') else {},
            'active_plugins': self.plugin_manager.active_plugins if hasattr(self, 'plugin_manager') else [],
            'timestamp': datetime.now().isoformat(),
            'optimizer_state': self.optimizer.state_dict()
        }
        
        # Save snapshot
        try:
            if compress:
                with gzip.open(save_path + '.gz', 'wb') as f:
                    torch.save(snapshot, f)
                print(f"[SillyAI] Saved compressed snapshot to {save_path}.gz")
            else:
                torch.save(snapshot, save_path)
                print(f"[SillyAI] Saved snapshot to {save_path}")
            return save_path
        except Exception as e:
            print(f"[SillyAI] Failed to save snapshot: {str(e)}")
            return None

    def load(self, path: str = None) -> bool:
        """High-level loading with all metadata without recursion"""
        try:
            # Directly load data
            path = path or self.path
            if path.endswith('.gz'):
                with gzip.open(path, 'rb') as f:
                    data = torch.load(f, map_location=self.device, weights_only=False)
            else:
                data = torch.load(path, map_location=self.device, weights_only=False)
            
            # Load transformer state directly
            self.transformer.load_state_dict(data['transformer_state'])
            
            # Restore other states
            self.best_loss = data.get('best_loss', float('inf'))
            self.current_epoch = data.get('current_epoch', 0)
            
            if 'config' in data:
                self.config = ModelConfig(**data['config'])
                
            if 'concept_graph' in data:
                self.concept_graph = data['concept_graph']
                
            # Handle plugins
            if hasattr(self, 'plugin_manager'):
                for plugin_name in data.get('active_plugins', []):
                    self.plugin_manager.enable_plugin(plugin_name)
                self.plugin_manager.load_states(data.get('plugin_states', {}))
            
            # Load optimizer state
            if 'optimizer_state' in data:
                self.optimizer.load_state_dict(data['optimizer_state'])
                
            print(f"[SillyAI] Loaded complete model snapshot from {path}")
            return True
            
        except Exception as e:
            print(f"[SillyAI] Failed to load snapshot: {str(e)}")
            return False

    def save_onnx(self, path):
        # ONNX does not support complex tensors directly. We export real and imaginary parts as extra channels.
        dummy_input = torch.rand(1, 1, self.transformer.input_proj.real_proj.in_features).to(self.device)
        if torch.is_complex(dummy_input):
            dummy_input = torch.view_as_real(dummy_input)
        torch.onnx.export(self, dummy_input, path)
        print(f"[SillyAI] ONNX model saved to {path} (complex tensors exported as real+imag channels)")

    def load_onnx(self, path):
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        print(f"[SillyAI] ONNX model loaded from {path} (note: complex weights are not natively supported)")
        return onnx_model

    def generate_bytecode(self, start_concept=None, max_hops=5):
        return self.transformer.concept_graph_to_bytecode(start_concept=start_concept, max_hops=max_hops)

    def get_concept_graph(self):
        return self.concept_graph

    def print_concepts(self, top_k=10):
        cg = self.concept_graph
        print("[SillyAI] Top concepts by centrality:")
        for name, score in cg.centrality(k=top_k):
            c = cg.get(name)
            print(f"  {name}: energy={c.energy:.3f}, access={c.access_count}, tags={c.tags}")

    def optimize_for_cpu(self):
        # Use torch.set_num_threads and other tricks for CPU
        import os
        torch.set_num_threads(max(1, os.cpu_count() // 2))
        torch.set_num_interop_threads(1)
        print(f"[SillyAI] Optimized for CPU: num_threads={torch.get_num_threads()}")

    def optimize_for_low_memory(self):
        # Use gradient checkpointing, smaller batch, etc.
        for m in self.modules():
            if hasattr(m, 'gradient_checkpointing'):
                m.gradient_checkpointing = True
        print("[SillyAI] Enabled low-memory optimizations (if supported).")

    def to_device(self, device):
        self.device = device
        self.transformer.to(device)
        print(f"[SillyAI] Model moved to {device}")