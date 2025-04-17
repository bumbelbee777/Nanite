import os
import gzip
import math
from math import sqrt
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from concept import ConceptGraph

def complex_loss(pred, target, alpha=0.5, beta=0.3, gamma=0.2, eps=1e-8):
    """Enhanced complex-valued loss with magnitude/phase components"""
    # Component losses
    re_loss = F.mse_loss(pred[...,0], target[...,0])
    im_loss = F.mse_loss(pred[...,1], target[...,1])
    
    # Magnitude loss with log-cosh smoothing
    pred_mag = torch.norm(pred, dim=-1, keepdim=True)
    target_mag = torch.norm(target, dim=-1, keepdim=True)
    mag_loss = torch.log(torch.cosh(pred_mag - target_mag)).mean()
    
    # Phase alignment loss
    dot_product = (pred * target).sum(-1)
    cos_sim = dot_product / (pred_mag * target_mag + eps).squeeze()
    phase_loss = 1 - torch.clamp(cos_sim, -1, 1).mean()
    
    return alpha*(re_loss + im_loss)/2 + beta*mag_loss + gamma*phase_loss

def hybrid_loss(pred, target, concept_graph=None, alpha=0.7):
    """Combines complex loss with concept alignment"""
    base_loss = complex_loss(pred, target)
    
    if concept_graph:
        concept_sim = torch.mean(torch.stack([
            torch.cosine_similarity(pred, graph_node.embedding)
            for graph_node in concept_graph.concepts.values()
        ]))
        return alpha*base_loss + (1-alpha)*(1-concept_sim)
    return base_loss

def complex_kaiming_init(tensor):
    """Specialized initialization for complex weights"""
    nn.init.kaiming_uniform_(tensor[...,0], a=sqrt(5))
    nn.init.kaiming_uniform_(tensor[...,1], a=sqrt(5), mode='fan_in', nonlinearity='linear')

class ComplexPhaseAugmentation(nn.Module):
    """Rotates complex features to learn phase-invariant representations"""
    def __init__(self, dim):
        super().__init__()
        self.phase_net = nn.Sequential(
            nn.Linear(2, dim//4),
            nn.ReLU(),
            nn.Linear(dim//4, 1)
        )
        
    def forward(self, x):
        phases = torch.atan2(x[...,1], x[...,0])
        mags = torch.norm(x, dim=-1)
        phase_shifts = self.phase_net(torch.stack([phases, mags], -1))
        return torch.stack([
            mags * torch.cos(phases + phase_shifts.squeeze()),
            mags * torch.sin(phases + phase_shifts.squeeze())
        ], -1)
    
class AdaptiveDynamicActivation(nn.Module):
    """Learns optimal dynamic adjustment rates"""
    def __init__(self, dim):
        super().__init__()
        self.adaptation_rate = nn.Parameter(torch.ones(1)*0.1)
        self.dynamic_net = nn.Linear(dim, dim*2)
        
    def forward(self, x):
        rates = torch.sigmoid(self.dynamic_net(x.mean(dim=1)) * self.adaptation_rate)
        return x * rates[:,:x.size(-1)] + rates[:,x.size(-1):]
    
class GraphEnergyModulator(nn.Module):
    """Modulates features based on concept energy patterns"""
    def __init__(self, dim):
        super().__init__()
        self.energy_proj = nn.Linear(1, dim)
        self.attention = nn.MultiheadAttention(dim, 1)
        
    def forward(self, x, concepts):
        energies = torch.tensor([[c.energy] for c in concepts], device=x.device)
        energy_feats = self.energy_proj(energies)
        attn_out, _ = self.attention(x, energy_feats, energy_feats)
        return x + attn_out
    
class GradientCheckpointedBlock(nn.Module):
    """Memory-efficient transformer block using gradient checkpointing"""
    def forward(self, x):
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(inputs[0])
            return custom_forward
        
        return torch.utils.checkpoint.checkpoint(
            create_custom_forward(self.block), 
            x
        )
    
class LocalityAwareAttention(nn.Module):
    """Combines local window attention with global concept attention"""
    def __init__(self, dim, heads, window_size):
        super().__init__()
        self.local_attn = nn.MultiheadAttention(dim, heads)
        self.global_attn = nn.MultiheadAttention(dim, 1)
        self.window_size = window_size
        
    def forward(self, x, concepts):
        # Local attention
        local_mask = torch.ones(x.size(0), x.size(0), device=x.device)
        for i in range(0, x.size(0), self.window_size):
            local_mask[i:i+self.window_size, i:i+self.window_size] = 0
        local_out, _ = self.local_attn(x, x, x, attn_mask=local_mask)
        
        # Global concept attention
        concept_feats = torch.stack([c.embedding for c in concepts])
        global_out, _ = self.global_attn(x, concept_feats, concept_feats)
        
        return local_out + global_out

class ComplexLinear(nn.Module):
    """Complex-valued linear transformation"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.real = nn.Linear(in_features, out_features)
        self.imag = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        return torch.stack([
            self.real(x[...,0]) - self.imag(x[...,1]),
            self.real(x[...,1]) + self.imag(x[...,0])
        ], dim=-1)

class QuantizedComplexLinear(nn.Module):
    """8-bit quantized complex linear layer"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, 2))
        self.register_buffer('scale', torch.tensor([1.0]))
        
    def forward(self, x):
        quant_weight = torch.quantize_per_tensor(
            self.weight, scale=self.scale, zero_point=0, dtype=torch.qint8
        ).dequantize()
        return ComplexLinear(x, quant_weight)
    
class SpikingComplexActivation(nn.Module):
    """Spiking neuron inspired complex activation"""
    def __init__(self, threshold=1.0, decay=0.9):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.register_buffer('mem_potential', torch.zeros(1))
        
    def forward(self, x):
        mag = x.norm(dim=-1)
        spike = (mag > self.threshold).float()
        self.mem_potential = self.decay * self.mem_potential + mag - spike * self.threshold
        phase = torch.atan2(x[...,1], x[...,0])
        return torch.stack([
            spike * self.mem_potential * torch.cos(phase),
            spike * self.mem_potential * torch.sin(phase)
        ], -1)

class DynamicActivation(nn.Module):
    """Mode-aware dynamic activation function"""
    def __init__(self, dim, real_mode=False):
        super().__init__()
        self.real_mode = real_mode
        hidden_dim = max(4, dim//8)
        
        if real_mode:
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.mag_net = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.phase_net = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

    def forward(self, x):
        if self.real_mode:
            slopes = self.net(x)
            return torch.where(x >= 0, x, slopes * x)
        else:
            mag = x.norm(dim=-1, keepdim=True)
            phase = torch.atan2(x[...,1], x[...,0]).unsqueeze(-1)
            
            mag_scale = torch.sigmoid(self.mag_net(torch.cat([mag, phase], -1)))
            phase_shift = 0.1 * torch.tanh(self.phase_net(torch.cat([phase, mag], -1)))
            
            new_mag = mag * mag_scale
            new_phase = phase + phase_shift
            return torch.cat([
                new_mag * torch.cos(new_phase),
                new_mag * torch.sin(new_phase)
            ], dim=-1)
        
class AdaptiveDepthScheduler(nn.Module):
    """Dynamically skips layers based on input complexity"""
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.router = nn.Linear(layers[0].dim, 1)
        
    def forward(self, x):
        outputs = []
        for layer in self.layers:
            if torch.sigmoid(self.router(x.mean(dim=1))) > 0.5:
                x = layer(x)
                outputs.append(x)
        return torch.stack(outputs).mean(dim=0)

class TransformerBlock(nn.Module):
    """Dual-mode transformer block"""
    def __init__(self, d_model, nhead, dim_ff, real_mode=False, dynamic_mode=False):
        super().__init__()
        self.real_mode = real_mode
        factor = 1 if real_mode else 2
        
        # Attention
        self.attn = nn.MultiheadAttention(d_model*factor, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model*factor)
        
        # Feedforward
        self.linear1 = ComplexLinear(d_model, dim_ff) if not real_mode else nn.Linear(d_model, dim_ff)
        self.linear2 = ComplexLinear(dim_ff, d_model) if not real_mode else nn.Linear(dim_ff, d_model)
        self.norm2 = nn.LayerNorm(d_model*factor)
        
        # Dynamic activation
        self.dynamic_mode = dynamic_mode
        if dynamic_mode:
            self.activation = DynamicActivation(dim_ff*factor, real_mode)
        else:
            self.activation = nn.ReLU() if real_mode else lambda x: torch.view_as_complex(x)

    def forward(self, x):
        # Attention
        x_flat = x if self.real_mode else x.flatten(-2)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x = self.norm1(x_flat + attn_out)
        
        # Feedforward
        if not self.real_mode:
            x = x.view(*x.shape[:-1], -1, 2)
        ff_out = self.linear1(x)
        ff_out = self.activation(ff_out)
        ff_out = self.linear2(ff_out)
        
        if not self.real_mode:
            ff_out = ff_out.flatten(-2)
        return self.norm2(x + ff_out)

class SillyAI(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, nhead, dim_ff,
                 output_dim=None, real_mode=False, dynamic_mode=False,
                 window_size=32, concept_dim=64, **kwargs):
        super().__init__()
        
        # Core configuration
        self.real_mode = real_mode
        self.dynamic_mode = dynamic_mode
        self.input_dim = input_dim
        self.d_model = d_model
        self.concept_dim = concept_dim
        
        # Initialize components
        self._init_components(output_dim, num_layers, nhead, dim_ff, window_size)
        
        # Training state tracking
        self.best_loss = float('inf')
        self.loss_history = []
        self.current_epoch = 0
        
        # Snapshot configuration
        self.snapshot_dir = "snapshots"
        os.makedirs(self.snapshot_dir, exist_ok=True)
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('mixed_precision', True))
        
    def _init_components(self, output_dim, num_layers, nhead, dim_ff, window_size):
        """Initialize all model components"""
        output_dim = output_dim or self.input_dim
        factor = 1 if self.real_mode else 2
        
        # Input projections
        self.input_proj = ComplexLinear(self.input_dim, self.d_model) if not self.real_mode \
                         else nn.Linear(self.input_dim, self.d_model)
        
        # Positional encoding
        self.pos_enc = nn.Parameter(torch.randn(5000, self.d_model * factor))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=self.d_model,
                nhead=nhead,
                dim_ff=dim_ff,
                real_mode=self.real_mode,
                dynamic_mode=self.dynamic_mode,
                window_size=window_size
            ) for _ in range(num_layers)
        ])
        
        # Output projections
        self.output_proj = ComplexLinear(self.d_model, output_dim) if not self.real_mode \
                          else nn.Linear(self.d_model, output_dim)
        
        # Concept graph system
        self.concept_graph = ConceptGraph()
        self.graph_modulator = GraphEnergyModulator(self.d_model * factor)
        self.concept_projector = nn.Linear(self.d_model * factor, self.concept_dim)
        
        # Adaptive components
        self.phase_augmenter = ComplexPhaseAugmentation(self.d_model)
        self.depth_scheduler = AdaptiveDepthScheduler(self.layers) if kwargs.get('adaptive_depth', False) else None
        
        # Initialize with custom methods
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Custom weight initialization"""
        if isinstance(module, (nn.Linear, ComplexLinear)):
            if self.real_mode:
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            else:
                complex_kaiming_init(module.weight)
            if module.bias is not None:
                nn.init.uniform_(module.bias, -1/module.in_features, 1/module.in_features)

    def forward(self, x):
        # Input processing
        if not self.real_mode and x.dim() < 3:
            x = torch.stack([x, torch.zeros_like(x)], -1)
        
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        seq_len = x.size(-2)
        x = x + self.pos_enc[:seq_len]
        
        # Phase augmentation
        if not self.real_mode:
            x = self.phase_augmenter(x)
        
        # Concept graph modulation
        if len(self.concept_graph.concepts) > 0:
            x = self.graph_modulator(x, list(self.concept_graph.concepts.values()))
        
        # Transformer layers
        if self.depth_scheduler:
            x = self.depth_scheduler(x)
        else:
            for layer in self.layers:
                x = layer(x)
        
        return self.output_proj(x)

    def train_step(self, batch, optimizer):
        inputs, targets = batch
        optimizer.zero_grad()
        
        # Mixed precision forward
        with torch.cuda.amp.autocast():
            preds = self(inputs)
            loss = hybrid_loss(preds, targets, self.concept_graph)
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        
        # Update concept energies
        self._update_concept_energies(loss.item())
        
        return loss.item()

    def _update_concept_energies(self, loss):
        """Update concept graph based on training dynamics"""
        for concept in self.concept_graph.concepts.values():
            concept.energy = min(1.0, concept.energy + 0.1 * loss)
        
        # Project concept embeddings
        with torch.no_grad():
            for name, concept in self.concept_graph.concepts.items():
                if not hasattr(concept, 'embedding'):
                    concept.embedding = self.concept_projector(
                        torch.randn(1, self.d_model * (1 if self.real_mode else 2))
                    )

    def save_snapshot(self, loss, epoch=None, is_best=False, compress=True):
        """Save complete model state with concept graph"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        epoch = epoch or self.current_epoch
        
        state = {
            'epoch': epoch,
            'model_state': self.state_dict(),
            'loss': loss,
            'loss_history': self.loss_history,
            'best_loss': self.best_loss,
            'concept_graph': self.concept_graph,
            'scaler_state': self.scaler.state_dict()
        }
        
        # Update best loss
        if loss < self.best_loss:
            self.best_loss = loss
            is_best = True
        
        # Save regular snapshot
        snapshot_path = os.path.join(self.snapshot_dir, f"sillyai_{timestamp}_epoch{epoch}.pt")
        torch.save(state, snapshot_path)
        
        # Save compressed version
        if compress:
            with gzip.open(f"{snapshot_path}.gz", 'wb') as f:
                torch.save(state, f)
        
        # Save best snapshot separately
        if is_best:
            best_path = os.path.join(self.snapshot_dir, "best_model.pt")
            torch.save(state, best_path)
            if compress:
                with gzip.open(f"{best_path}.gz", 'wb') as f:
                    torch.save(state, f)
        
        return snapshot_path

    def load_snapshot(self, path, optimizer=None):
        """Load model state from snapshot"""
        # Handle compressed snapshots
        if path.endswith('.gz'):
            with gzip.open(path, 'rb') as f:
                state = torch.load(f)
        else:
            state = torch.load(path)
        
        # Load model state
        self.load_state_dict(state['model_state'])
        self.loss_history = state.get('loss_history', [])
        self.best_loss = state.get('best_loss', float('inf'))
        self.current_epoch = state.get('epoch', 0)
        
        # Load concept graph if available
        if 'concept_graph' in state:
            self.concept_graph = state['concept_graph']
        
        # Load optimizer state if provided
        if optimizer and 'optimizer_state' in state:
            optimizer.load_state_dict(state['optimizer_state'])
        
        # Load gradient scaler if available
        if 'scaler_state' in state:
            self.scaler.load_state_dict(state['scaler_state'])
        
        print(f"Loaded snapshot from {path} (epoch {self.current_epoch}, loss {state['loss']:.4f})")

    def train(self, dataloader, epochs, optimizer, snapshot_interval=10):
        """Complete training loop with automatic snapshotting"""
        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            self.train()
            epoch_loss = 0.0
            
            for batch in dataloader:
                loss = self.train_step(batch, optimizer)
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(dataloader)
            self.loss_history.append(avg_loss)
            self.current_epoch += 1
            
            # Periodic snapshotting
            if epoch % snapshot_interval == 0 or avg_loss < self.best_loss:
                self.save_snapshot(avg_loss, epoch, avg_loss < self.best_loss)
            
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f} {'(Best)' if avg_loss < self.best_loss else ''}")
        
        return self.loss_history

    def add_concept(self, name, initial_energy=0.5):
        """Add a new concept to the graph with initialization"""
        self.concept_graph.add_concept(name, initial_energy)
        
        # Initialize embedding
        with torch.no_grad():
            embedding = self.concept_projector(
                torch.randn(1, self.d_model * (1 if self.real_mode else 2))
            )
            setattr(self.concept_graph.concepts[name], 'embedding', embedding)

    def relate_concepts(self, source, target, weight=1.0, relationship="related"):
        """Add relationship between concepts with optional embedding mixing"""
        self.concept_graph.add_connection(source, target, weight, relationship)
        
        # Update embeddings
        if hasattr(self.concept_graph.concepts[source], 'embedding') and \
           hasattr(self.concept_graph.concepts[target], 'embedding'):
            mix = 0.1 * (self.concept_graph.concepts[source].embedding + 
                         self.concept_graph.concepts[target].embedding)
            self.concept_graph.concepts[source].embedding += mix
            self.concept_graph.concepts[target].embedding += mix