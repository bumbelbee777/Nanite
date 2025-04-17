import os
import gzip
import math
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from concept import ConceptGraph

class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, factorized=False, rank=4):
        super().__init__()
        self.factorized = factorized
        
        if factorized:  # Kronecker-factorized implementation
            self.A_real = nn.Parameter(torch.randn(out_features//rank, in_features))
            self.A_imag = nn.Parameter(torch.randn(out_features//rank, in_features))
            self.B_real = nn.Parameter(torch.randn(rank, in_features))
            self.B_imag = nn.Parameter(torch.randn(rank, in_features))
        else:
            self.real = nn.Linear(in_features, out_features)
            self.imag = nn.Linear(in_features, out_features)

    def forward(self, x):
        if self.factorized:
            W_real = torch.kron(self.A_real, self.B_real) - torch.kron(self.A_imag, self.B_imag)
            W_imag = torch.kron(self.A_real, self.B_imag) + torch.kron(self.A_imag, self.B_real)
            real = F.linear(x[...,0], W_real) - F.linear(x[...,1], W_imag)
            imag = F.linear(x[...,0], W_imag) + F.linear(x[...,1], W_real)
        else:
            real = self.real(x[...,0]) - self.imag(x[...,1])
            imag = self.real(x[...,1]) + self.imag(x[...,0])
        return torch.stack([real, imag], -1)

class ComplexActivation(nn.Module):
    """Original complex activation from earlier versions"""
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

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, 
                 real_mode=False, dynamic=False, 
                 window_size=64, use_toeplitz=False):
        super().__init__()
        factor = 1 if real_mode else 2
        self.use_toeplitz = use_toeplitz
        
        # Attention mechanism
        if use_toeplitz:
            self.attn = ToeplitzAttention(d_model*factor)
        else:
            self.attn = nn.MultiheadAttention(d_model*factor, nhead)
        
        # Feedforward with optional LoRA
        self.ff = FeedForward(d_model, dim_ff, real_mode, dynamic)
        
        # Local/global attention
        self.locality_attn = LocalityAwareAttention(d_model*factor, nhead, window_size)
        
        self.norm1 = nn.LayerNorm(d_model*factor)
        self.norm2 = nn.LayerNorm(d_model*factor)

    def forward(self, x):
        # Attention path
        x_attn, _ = self.attn(x, x, x)
        x = self.norm1(x + x_attn)
        
        # Locality attention
        x = self.locality_attn(x)
        
        # Feedforward path
        x_ff = self.ff(x)
        return self.norm2(x + x_ff)

class SillyAI(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, nhead, dim_ff,
                 output_dim=None, real_mode=False, dynamic_mode=False,
                 window_size=64, concept_dim=128, optim_args=None):
        super().__init__()
        self.real_mode = real_mode
        self.optim_args = optim_args or {}
        
        # Input processing
        self.input_proj = self._create_projection(input_dim, d_model)
        self.pos_enc = nn.Parameter(torch.randn(5000, d_model * (1 if real_mode else 2)))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            self._create_transformer_block(d_model, nhead, dim_ff, window_size)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = self._create_projection(d_model, output_dim or input_dim)
        
        # Concept system
        self.concept_graph = ConceptGraph()
        self._init_concept_system(d_model, concept_dim)
        
        # Training state
        self._init_training_state()

        self.activation = ComplexActivation if not optim_args.get('new_activations') else OptimizedActivation

    def complex_loss(self, pred, target, alpha=0.5, beta=0.3, gamma=0.2, eps=1e-8):
        """Original complex loss function with phase/magnitude components"""
        re_loss = F.mse_loss(pred[...,0], target[...,0])
        im_loss = F.mse_loss(pred[...,1], target[...,1])
        
        # Magnitude loss with log-cosh
        pred_mag = torch.norm(pred, dim=-1, keepdim=True)
        target_mag = torch.norm(target, dim=-1, keepdim=True)
        mag_loss = torch.log(torch.cosh(pred_mag - target_mag)).mean()
        
        # Phase alignment
        dot_product = (pred * target).sum(-1)
        cos_sim = dot_product / (pred_mag * target_mag + eps).squeeze()
        phase_loss = 1 - torch.clamp(cos_sim, -1, 1).mean()
        
        return alpha*(re_loss + im_loss)/2 + beta*mag_loss + gamma*phase_loss

    def hybrid_loss(self, pred, target, concept_graph=None, alpha=0.7):
        """Original hybrid loss with concept alignment"""
        base_loss = self.complex_loss(pred, target)
        
        if concept_graph:
            concept_sim = torch.mean(torch.stack([
                torch.cosine_similarity(pred, graph_node.embedding)
                for graph_node in concept_graph.concepts.values()
            ]))
            return alpha*base_loss + (1-alpha)*(1-concept_sim)
        return base_loss

    def forward(self, x):
        # Preserve original complex value encoding
        if not self.real_mode and x.dim() < 3:
            x = torch.stack([x, torch.zeros_like(x)], -1)
        
        # Original phase augmentation
        if not self.real_mode:
            phases = torch.atan2(x[...,1], x[...,0])
            mags = torch.norm(x, dim=-1)
            phase_shifts = self.phase_net(torch.stack([phases, mags], -1))
            x = torch.stack([
                mags * torch.cos(phases + phase_shifts.squeeze()),
                mags * torch.sin(phases + phase_shifts.squeeze())
            ], -1)

    def _create_projection(self, in_dim, out_dim):
        return ComplexLinear(in_dim, out_dim, 
            factorized=self.optim_args.get('factorized_linear', False),
            rank=self.optim_args.get('kronecker_rank', 4)
        ) if not self.real_mode else nn.Linear(in_dim, out_dim)

    def _create_transformer_block(self, d_model, nhead, dim_ff, window_size):
        return TransformerBlock(
            d_model=d_model,
            nhead=nhead,
            dim_ff=dim_ff,
            real_mode=self.real_mode,
            dynamic=self.optim_args.get('dynamic_ffn', False),
            window_size=window_size,
            use_toeplitz=self.optim_args.get('use_toeplitz', False)
        )

    def _init_concept_system(self, d_model, concept_dim):
        factor = 1 if self.real_mode else 2
        self.concept_projector = nn.Linear(d_model * factor, concept_dim)
        self.graph_modulator = GraphEnergyModulator(
            d_model * factor,
            factorized=self.optim_args.get('factorized_graph', False)
        )

    def _init_training_state(self):
        self.best_loss = float('inf')
        self.loss_history = []
        self.current_epoch = 0
        self.snapshot_dir = "snapshots"
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.optim_args.get('mixed_precision', True)
        )

    def train_step(self, batch, optimizer):
        inputs, targets = batch
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            preds = self(inputs)
            loss = self.hybrid_loss(preds, targets)

        self.scaler.scale(loss).backward()
        self._apply_gradient_sparsity(0.9)
        self.scaler.step(optimizer)
        self.scaler.update()

        self._update_concept_graph(loss.item())
        return loss.item()

    def hybrid_loss(self, pred, target):
        base_loss = complex_loss(pred, target)
        concept_loss = self.concept_alignment_loss(pred)
        return 0.7*base_loss + 0.3*concept_loss

    def concept_alignment_loss(self, pred):
        concepts = torch.stack([c.embedding for c in self.concept_graph.concepts.values()])
        return 1 - F.cosine_similarity(pred, concepts).mean()

    def save_snapshot(self, loss, epoch=None, compress=True):
        state = {
            'epoch': epoch or self.current_epoch,
            'model_state': self.state_dict(),
            'loss_history': self.loss_history,
            'best_loss': min(loss, self.best_loss),
            'concept_graph': self.concept_graph,
            'config': self._get_config(),
            'scaler_state': self.scaler.state_dict()
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.snapshot_dir, f"sillyai_{timestamp}.pt")
        torch.save(state, gzip.open(path+".gz", 'wb') if compress else path)
        return path

    def load_snapshot(self, path):
        load_func = gzip.open if path.endswith('.gz') else open
        with load_func(path, 'rb') as f:
            state = torch.load(f)
            
        self.load_state_dict(state['model_state'])
        self.concept_graph = state.get('concept_graph', ConceptGraph())
        self.scaler.load_state_dict(state.get('scaler_state', None))
        self.best_loss = state.get('best_loss', float('inf'))
        self.loss_history = state.get('loss_history', [])
        self.current_epoch = state.get('epoch', 0)
        return self

    def _get_config(self):
        return {
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'nhead': self.nhead,
            'real_mode': self.real_mode,
            'dynamic_mode': self.dynamic_mode,
            'quantize_bits': self.quantize_bits
        }

    def _quantize_input(self, x):
        if self.real_mode:
            scale = x.abs().max() / (2**(self.quantize_bits-1)-1)
            return torch.round(x/scale) * scale
        else:
            mag = x.norm(dim=-1, keepdim=True)
            phase = torch.atan2(x[...,1], x[...,0]).unsqueeze(-1)
            q_mag = torch.exp2(torch.round(torch.log2(mag)))
            q_phase = torch.round(phase * (2**self.quantize_bits)) / (2**self.quantize_bits)
            return torch.stack([q_mag*torch.cos(q_phase), q_mag*torch.sin(q_phase)], -1)

    def _apply_gradient_sparsity(self, density):
        for param in self.parameters():
            if param.grad is not None:
                mask = torch.rand_like(param.grad) < density
                param.grad *= mask / density

    def _update_concept_graph(self, loss):
        for concept in self.concept_graph.concepts.values():
            concept.energy = (concept.energy + 0.1 * (1 - loss)).clamp(0, 1)
            if not hasattr(concept, 'embedding'):
                concept.embedding = self.concept_projector(torch.randn(1, self.d_model*(1 if self.real_mode else 2)))

class ToeplitzAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value):
        q, k = self.q_proj(query), self.k_proj(key)
        kernel = F.conv1d(q.unsqueeze(1), k.flip(-1).unsqueeze(0))
        attn_weights = F.softmax(kernel / math.sqrt(q.size(-1)), dim=-1)
        return torch.einsum('bnk,bkd->bnd', attn_weights, value), None

class LocalityAwareAttention(nn.Module):
    def __init__(self, dim, heads, window_size):
        super().__init__()
        self.window_size = window_size
        self.local_attn = nn.MultiheadAttention(dim, heads)
        self.global_attn = nn.MultiheadAttention(dim, 1)

    def forward(self, x):
        seq_len = x.size(0)
        local_out = []
        for i in range(0, seq_len, self.window_size):
            window = x[i:i+self.window_size]
            local, _ = self.local_attn(window, window, window)
            local_out.append(local)
        local_out = torch.cat(local_out)
        
        global_out, _ = self.global_attn(x, x, x)
        return local_out + global_out

class FeedForward(nn.Module):
    def __init__(self, d_model, dim_ff, real_mode, dynamic):
        super().__init__()
        factor = 1 if real_mode else 2
        self.linear1 = ComplexLinear(d_model, dim_ff) if not real_mode else nn.Linear(d_model, dim_ff)
        self.linear2 = ComplexLinear(dim_ff, d_model) if not real_mode else nn.Linear(dim_ff, d_model)
        self.activation = DynamicActivation(dim_ff*factor, real_mode) if dynamic else (
            nn.ReLU() if real_mode else ComplexActivation()
        )

class GraphEnergyModulator(nn.Module):
    def __init__(self, dim, factorized=False):
        super().__init__()
        self.energy_proj = ComplexLinear(1, dim, factorized) if factorized else nn.Linear(1, dim)
        self.attention = nn.MultiheadAttention(dim, 1)