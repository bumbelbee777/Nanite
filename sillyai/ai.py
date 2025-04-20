import os
import math
import gzip
import shutil
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Any

from .plugin import PluginManager
from .vm import SillyVM

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import fft

from .concept import ConceptGraph

class ComplexOperationMixin:
    @staticmethod
    def complex_forward(Wr, Wi, x_real, x_imag):
        y_real = F.linear(x_real, Wr) - F.linear(x_imag, Wi)
        y_imag = F.linear(x_real, Wi) + F.linear(x_imag, Wr)
        return y_real, y_imag

class ComplexLinear(nn.Module):
    """
    Complex‐valued linear layer that:
      - Batches the two real‐and‐imaginary F.linear calls into one block‐matrix multiply.
      - Optionally offloads that block‐matrix multiply to its own CUDA stream.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        factorized: bool = False,
        rank: int = 4,
        use_async: bool = False,
        weight_init: str = "xavier"  # Add weight_init parameter
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.factorized = factorized
        self.rank = rank
        self.use_async = use_async
        self.weight_init = weight_init  # Store weight_init preference

        # parameters
        if self.factorized:
            assert out_features % rank == 0
            d1 = out_features // rank
            self.A_real = nn.Parameter(torch.empty(d1, in_features))
            self.A_imag = nn.Parameter(torch.empty(d1, in_features))
            self.B_real = nn.Parameter(torch.empty(rank, in_features))
            self.B_imag = nn.Parameter(torch.empty(rank, in_features))
            for p in (self.A_real, self.A_imag, self.B_real, self.B_imag):
                if self.weight_init == "xavier":
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.kaiming_normal_(p)
        else:
            self.weight_real = nn.Parameter(torch.empty(out_features, in_features))
            self.weight_imag = nn.Parameter(torch.empty(out_features, in_features))
            
            # Initialize weights based on weight_init preference
            if self.weight_init == "xavier":
                nn.init.xavier_uniform_(self.weight_real)
                nn.init.xavier_uniform_(self.weight_imag)
            else:
                nn.init.kaiming_normal_(self.weight_real)
                nn.init.kaiming_normal_(self.weight_imag)
                
            self.bias = nn.Parameter(torch.zeros(out_features, 2))

    def forward(self, x):
        # x: (..., in_features, 2) where [..., :,0]=real, [..., :,1]=imag
        *batch, dim, last = x.shape
        assert last == 2 and dim == self.in_features

        x_r, x_i = x.unbind(-1)                         # both (..., in)
        if self.factorized:
            Wr = torch.kron(self.A_real, self.B_real) - torch.kron(self.A_imag, self.B_imag)
            Wi = torch.kron(self.A_real, self.B_imag) + torch.kron(self.A_imag, self.B_real)
        else:
            Wr, Wi = self.weight_real, self.weight_imag

        # Build the 2-out × 2-in block matrix:
        #   [ Wr  -Wi ]
        #   [ Wi   Wr ]
        W11 = torch.cat([Wr, -Wi], dim=1)  # (out, 2*in)
        W21 = torch.cat([Wi,  Wr], dim=1)
        W_block = torch.cat([W11, W21], dim=0)  # (2*out, 2*in)

        # Flatten inputs: (..., 2*in)
        x_cat = torch.cat([x_r, x_i], dim=-1)

        def _compute(mat, vec, bsz):
            y = F.linear(vec, mat)      # (..., 2*out)
            # reshape back to (..., out, 2)
            y = y.view(*bsz, self.out_features, 2)
            return y

        # if async and on CUDA, run the block‐matrix multiply in a separate stream
        if self.use_async and x.is_cuda:
            cur = torch.cuda.current_stream()
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                y = _compute(W_block, x_cat, batch)
            # make sure the result is ready on the main stream
            cur.wait_stream(stream)
            return y

        # synchronous fallback
        return _compute(W_block, x_cat, batch)
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InfiniAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        local_window: int = 512,
        memory_size: int = 1024,
        compress_ratio: int = 4,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim     = embed_dim
        self.num_heads     = num_heads
        self.head_dim      = embed_dim // num_heads
        self.local_window  = local_window
        self.memory_size   = memory_size
        self.compress_ratio= compress_ratio

        # per‑head projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out    = nn.Linear(embed_dim, embed_dim)

        # circular buffers for memory
        # each is (M_total, embed_dim)
        self.register_buffer('mem_k', torch.zeros(0, embed_dim))
        self.register_buffer('mem_v', torch.zeros(0, embed_dim))

    def _compress_and_trim(self):
        # if we've stored more than memory_size * compress_ratio,
        # compress blocks of size compress_ratio by averaging
        M_tot = self.mem_k.size(0)
        thresh = self.memory_size * self.compress_ratio
        if M_tot > thresh:
            # reshape to (memory_size, C, D), average over C
            new_k = self.mem_k[-thresh:]
            new_v = self.mem_v[-thresh:]
            new_k = new_k.view(self.memory_size, self.compress_ratio, -1).mean(dim=1)
            new_v = new_v.view(self.memory_size, self.compress_ratio, -1).mean(dim=1)
            self.mem_k = new_k
            self.mem_v = new_v
        # then trim to last memory_size
        if self.mem_k.size(0) > self.memory_size:
            self.mem_k = self.mem_k[-self.memory_size:]
            self.mem_v = self.mem_v[-self.memory_size:]

    def forward(self, x):
        """
        x: (B, L, D)
        returns: (B, L, D)
        """
        B, L, D = x.shape
        q = self.q_proj(x)  # (B,L,D)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 1) Local (masked) attention
        # We'll do it in batched form by building a banded mask
        # and using scaled dot‐product
        qh = q.view(B, L, self.num_heads, self.head_dim).transpose(1,2)  # (B, H, L, d)
        kh = k.view(B, L, self.num_heads, self.head_dim).transpose(1,2)
        vh = v.view(B, L, self.num_heads, self.head_dim).transpose(1,2)

        # compute full scores then mask out-of-window
        scores = torch.matmul(qh, kh.transpose(-2,-1))  # (B,H,L,L)
        scores = scores / math.sqrt(self.head_dim)
        # create band mask
        idxs = torch.arange(L, device=x.device)
        mask = (idxs[None, :] - idxs[:, None]).abs() > self.local_window
        scores = scores.masked_fill(mask[None,None,:,:], float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out_local = torch.matmul(attn, vh)               # (B,H,L,d)
        out_local = out_local.transpose(1,2).reshape(B, L, D)

        # 2) Update and compress memory
        # flatten batch into M_new = B*L entries
        new_k = k.reshape(B*L, D)
        new_v = v.reshape(B*L, D)
        self.mem_k = torch.cat([self.mem_k, new_k], dim=0)
        self.mem_v = torch.cat([self.mem_v, new_v], dim=0)
        self._compress_and_trim()

        # 3) Linear global attention to memory via φ(·)=ELU(·)+1
        phi_q = F.elu(q) + 1.0         # (B,L,D)
        phi_k = F.elu(self.mem_k) + 1.0# (M,  D)

        # KV aggregate: Kᵀ·V → (D, D)
        kv = torch.einsum('md,me->de', phi_k, self.mem_v)

        # context_mem = φ(q) @ kv  → (B,L,D)
        out_mem = torch.einsum('bld,df->blf', phi_q, kv)

        # 4) combine and final projection
        out = out_local + out_mem
        return self.out(out)

class ToeplitzAttention(nn.Module):
    """
    - real_mode: depth‑wise conv1d Toeplitz via a single grouped conv call.
    - complex: fully batched FFT circular‐conv, with optional CUDA‐stream asynchrony.
    """
    def __init__(
        self,
        embed_dim: int,
        real_mode: bool = False,
        factorized: bool = False,
        rank: int = 4,
        use_async: bool = False,
        weight_init: str = "xavier"
    ):
        super().__init__()
        self.real_mode = real_mode
        self.use_async = use_async

        if real_mode:
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
        else:
            self.q_proj = ComplexLinear(embed_dim, embed_dim, factorized, rank, weight_init=weight_init)
            self.k_proj = ComplexLinear(embed_dim, embed_dim, factorized, rank, weight_init=weight_init)
            self.v_proj = ComplexLinear(embed_dim, embed_dim, factorized, rank, weight_init=weight_init)

    def forward(self, query, key, value):
        """
        query/key/value:
          real_mode → (B, L, D)
          complex   → (B, L, D, 2)
        returns: (B, L, D, (*2 if complex))
        """
        if self.real_mode:
            # 1) project
            q = self.q_proj(query)    # (B,L,D)
            k = self.k_proj(key)
            v = self.v_proj(value)

            B, L, D = q.shape
            # 2) depthwise conv correlation
            q_t = q.permute(0,2,1)                      # (B,D,L)
            k_t = k.permute(0,2,1).flip(-1)             # (B,D,L)
            qc = q_t.reshape(1, B*D, L)                 # (1, B·D, L)
            kc = k_t.reshape(B*D, 1, L)                 # (B·D, 1, L)

            corr = F.conv1d(qc, kc, padding=L-1, groups=B*D)  # (1, B·D, 2L-1)
            corr = corr.view(B, D, 2*L-1)                     # (B, D, 2L-1)
            kernel = corr[:, :, L-1:]                         # (B, D, L)

            scores = kernel.mean(dim=1) / math.sqrt(D)        # (B, L)
            attn   = torch.softmax(scores, dim=-1)            # (B, L)

            out = attn.unsqueeze(-1) * v                      # (B, L, D)
            return out, None

        # --- complex branch via FFT ---
        # 1) project and cast to complex
        q = torch.view_as_complex(self.q_proj(query))       # (B,L,D)
        k = torch.view_as_complex(self.k_proj(key))
        v = torch.view_as_complex(self.v_proj(value))

        B, L, D = q.shape
        n = 2*L - 1

        def _fft_path():
            # a) correlation → real kernel
            qf = fft.rfft(q,       n=n, dim=1)             # (B, n_freq, D)
            kf = fft.rfft(k.flip(1), n=n, dim=1)
            corr = fft.irfft(qf * kf, n=n, dim=1)           # (B, n, D)
            kernel = corr[:, L-1:, :]                      # (B, L, D)

            # b) circular conv
            kernel_c = torch.complex(kernel, torch.zeros_like(kernel))
            Kf = fft.fft(kernel_c, n=L, dim=1)
            Vf = fft.fft(v,        n=L, dim=1)
            out_c = fft.ifft(Kf * Vf, n=L, dim=1)          # (B, L, D)
            return out_c

        if self.use_async and query.is_cuda:
            main = torch.cuda.current_stream()
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                out_c = _fft_path()
            main.wait_stream(stream)
        else:
            out_c = _fft_path()

        # back to real/imag tensor
        out = torch.view_as_real(out_c)  # (B, L, D, 2)
        return out, None
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # (max_len, d_model)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return self.pe[:seq_len].unsqueeze(0)  # (1, seq_len, d_model)

class DynamicActivation(nn.Module):
    """
    - real_mode: per‑batch GRUCell over mean features → learn slope & offset
    - complex: per‑time‑step GRUCell on [mag,phase] → learn mag & phase adjustments
    """
    def __init__(
        self,
        dim: int,
        real_mode: bool = False,
        hidden_size: int = None
    ):
        super().__init__()
        self.real_mode = real_mode
        h = hidden_size or (dim // (4 if real_mode else 8))
        self.adaptation_rate = nn.Parameter(torch.tensor(0.1))

        # real‑valued path
        self.gru_real = nn.GRUCell(dim, h)
        self.out_real = nn.Sequential(
            nn.ReLU(),
            nn.Linear(h, 2),   # produces [slope, offset]
            nn.Sigmoid()
        )

        # complex‑valued path
        self.gru_mag   = nn.GRUCell(2, h)
        self.out_mag   = nn.Sequential(nn.ReLU(), nn.Linear(h, 1), nn.Sigmoid())
        self.gru_phase = nn.GRUCell(2, h)
        self.out_phase = nn.Sequential(nn.ReLU(), nn.Linear(h, 1), nn.Tanh())

        self.mag_mix   = nn.Parameter(torch.ones(1))
        self.phase_mix = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if self.real_mode:
            # x: (B, L, D)
            B, L, D = x.shape
            # summarize sequence
            stats = x.mean(dim=1)                    # (B, D)
            h0 = x.new_zeros(B, self.gru_real.hidden_size)
            h1 = self.gru_real(stats, h0)            # (B, h)
            slope, offset = self.out_real(h1).chunk(2, dim=-1)
            slope  = slope.view(B, 1, D)
            offset = offset.view(B, 1, D)
            # elementwise leaky with learned slope
            activated = torch.where(x >= 0, x, slope * x)
            return activated + offset * self.adaptation_rate

        # complex mode: x (B, L, D, 2)
        B, L, D, _ = x.shape
        xr, xi = x.unbind(-1)
        mag   = torch.sqrt(xr**2 + xi**2).unsqueeze(-1)   # (B, L, 1)
        phase = torch.atan2(xi, xr).unsqueeze(-1)         # (B, L, 1)
        inp   = torch.cat([mag, phase], dim=-1)           # (B, L, 2)

        # run GRUCell across time, vectorized over batch
        h_mag = x.new_zeros(B, self.gru_mag.hidden_size)
        h_ph  = x.new_zeros(B, self.gru_phase.hidden_size)
        mag_adj_seq, ph_adj_seq = [], []

        for t in range(L):
            xt = inp[:, t, :]                   # (B, 2)
            h_mag = self.gru_mag(xt, h_mag)     # (B, h)
            h_ph  = self.gru_phase(xt, h_ph)    # (B, h)
            mag_adj_seq.append(self.out_mag(h_mag).unsqueeze(1))    # (B,1,1)
            ph_adj_seq.append(self.out_phase(h_ph).unsqueeze(1))    # (B,1,1)

        mag_adj   = torch.cat(mag_adj_seq, dim=1) * self.mag_mix    # (B, L, 1)
        phase_adj = torch.cat(ph_adj_seq, dim=1) * self.phase_mix  # (B, L, 1)

        new_mag   = mag * (1 + mag_adj * self.adaptation_rate)
        new_phase = phase + phase_adj * self.adaptation_rate

        real = new_mag * torch.cos(new_phase)
        imag = new_mag * torch.sin(new_phase)
        return torch.cat([real, imag], dim=-1)                   # (B, L, D, 2)

@dataclass
class ModelConfig:
    # core dims
    output_dim: int
    concept_dim: int
    input_dim: int
    d_model: int
    num_layers: int
    nhead: int
    dim_ff: int

    # modes
    real_mode: bool = False
    dynamic_mode: bool = True

    # optimizations
    optim_args: Optional[List[Any]] = None

    # complex‐linear factorization
    factorized_linear: bool = False
    kronecker_rank: int = 4

    # activation hidden sizes (overrides defaults)
    dynamic_hidden_dim: Optional[int] = None
    activation_hidden_dim: Optional[int] = None

    # Toeplitz implementation: 'fft' or 'explicit'
    toeplitz_complex_method: str = 'fft'

    # weight init: 'xavier' or 'kaiming'
    weight_init: str = 'xavier'

    # training / checkpointing
    mixed_precision: bool = True
    snapshot_dir: str = "./snapshots"
    keep_best_only: bool = True

    plugin_dir: str = "./plugins"

    use_infini: bool = False
    infini_local_window: int = 512
    infini_mem_size: int = 1024
    infini_compress_ratio: int = 4

    def __post_init__(self):
        # Initialize default optim_args if None
        if self.optim_args is None:
            self.optim_args = {
                'use_toeplitz': False,
                'factorized_linear': self.factorized_linear,
                'mixed_precision': self.mixed_precision
            }

class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        factor = 1 if config.real_mode else 2
        self.linear1 = self._create_linear(config.d_model, config.dim_ff, config)
        self.linear2 = self._create_linear(config.dim_ff, config.d_model, config)
        self.activation = self._create_activation(config)

    def _create_linear(self, in_dim, out_dim, config):
        if config.real_mode:
            return nn.Linear(in_dim, out_dim)
        return ComplexLinear(in_dim, out_dim, config.optim_args.get('factorized_linear', False))

    def _create_activation(self, config):
        if config.dynamic_mode:
            return DynamicActivation(
                config.dim_ff * (1 if config.real_mode else 2),
                real_mode=config.real_mode,
                hidden_size=config.dynamic_hidden_dim  # Pass from config
            )
        return nn.ReLU() if config.real_mode else DynamicActivation(config.dim_ff, config.real_mode)

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        factor = 1 if config.real_mode else 2

        if getattr(config, 'use_infini', False):
            # use Infini-attention
            self.attn = InfiniAttention(
                embed_dim=config.d_model * factor,
                num_heads=config.nhead,
                local_window=config.infini_local_window,
                memory_size=config.infini_mem_size,
                compress_ratio=config.infini_compress_ratio,
            )
        else:
            # your existing Toeplitz/standard attention fallback
            self.attn = self._create_attention(config)

        self.ff    = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.d_model * factor)
        self.norm2 = nn.LayerNorm(config.d_model * factor)

    def _create_attention(self, config):
        use_toeplitz = config.optim_args.get('use_toeplitz', False) if config.optim_args else False
        
        return ToeplitzAttention(
            embed_dim=config.d_model * (1 if config.real_mode else 2),
            real_mode=config.real_mode,
            factorized=config.factorized_linear,
            rank=config.kronecker_rank,
            weight_init=config.weight_init  # Pass weight_init from config
        )

    def forward(self, x):
        x_attn, _ = self.attn(x, x, x)
        x = self.norm1(x + x_attn)
        return self.norm2(x + self.ff(x))

class SillyAI(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        factor = 1 if config.real_mode else 2

        os.makedirs(config.snapshot_dir, exist_ok=True)

        # projections
        self.input_proj  = self._make_proj(config.input_dim, config.d_model)
        self.output_proj = self._make_proj(
            config.d_model,
            config.output_dim or config.input_dim
        )

        # positional
        self.pos_enc = PositionalEncoding(config.d_model * factor)

        # transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # concept system
        self.concept_graph    = ConceptGraph()
        self.concept_projector = nn.Linear(config.d_model * factor, config.concept_dim)

        # training state
        self.best_loss = float('inf')
        self.current_epoch = 0
        self.scaler = torch.amp.GradScaler(
            enabled=self.config.optim_args.get('mixed_precision', True)
        )

        self.plugin_manager = PluginManager(config.plugin_dir)
        self.plugin_manager.discover()
        self.plugin_manager.apply_on_init(self)

        # Add SillyVM integration
        self.vm = SillyVM()
        self.proof_cache = {}

    def _make_proj(self, in_dim, out_dim):
        if self.config.real_mode:
            return ComplexLinear(
                in_dim, out_dim,
                factorized=self.config.factorized_linear,  # Use direct field
                rank=self.config.kronecker_rank
            )
    
    def _forward_core(self, x):
        # complex fallback
        if not self.config.real_mode and x.dim() == 2:
            x = torch.stack([x, torch.zeros_like(x)], dim=-1)

        x = self.input_proj(x) + self.pos_enc(x if x.dim()==3 else x.unsqueeze(1))

        if self.concept_graph.concepts:
            energies = torch.tensor(
                [[c.energy] for c in self.concept_graph.concepts.values()],
                device=x.device, dtype=x.dtype
            )
            x = x + energies.mean()

        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x)

    def forward(self, x):
        # plugin hook: before
        x = self.plugin_manager.apply_before_forward(self, x)

        out = self._forward_core(x)

        # plugin hook: after
        out = self.plugin_manager.apply_after_forward(self, x, out)

        # Update concept energies based on proof results
        for problem, success in self.proof_cache.items():
            concepts = self.concept_graph.get_related_concepts(problem)
            energy_delta = 0.1 if success else -0.05
            for concept in concepts:
                self.concept_graph.update_concept_energy(concept, energy_delta)

        return out

    def solve_problem(self, problem: str, proof_path: Optional[str] = None) -> bool:
        """
        Attempts to solve a problem using SillyISA proof verification
        
        Args:
            problem: Problem description or identifier
            proof_path: Optional path to SillyISA proof file
            
        Returns:
            bool: True if proof succeeds, False otherwise
        """
        # Check proof cache first
        if problem in self.proof_cache:
            return self.proof_cache[problem]
            
        try:
            # If proof file provided, use it
            if proof_path:
                success = self._verify_proof(proof_path)
            else:
                # Generate proof based on concept graph
                proof = self._generate_proof(problem)
                success = self._verify_proof_code(proof)
                
            self.proof_cache[problem] = success
            return success
            
        except Exception as e:
            print(f"Proof verification failed: {str(e)}")
            return False
            
    def _verify_proof(self, proof_path: str) -> bool:
        """Verifies a SillyISA proof file"""
        try:
            self.vm.run_bytecode_from_file(proof_path)
            return True
        except AssertionError:
            return False
            
    def _verify_proof_code(self, proof_code: str) -> bool:
        """Verifies proof from generated code string"""
        try:
            self.vm.parse(proof_code)
            self.vm.run()
            return True
        except AssertionError:
            return False
            
    def _generate_proof(self, problem: str) -> str:
        """Generates SillyISA proof code from problem description"""
        # Get relevant concepts
        concepts = self.concept_graph.get_related_concepts(problem)
        
        # Generate proof template
        proof = [
            f".Proof_{problem.replace(' ', '_')}() {{",
            "    // Initialize variables"
        ]
        
        # Add concept bindings
        for i, concept in enumerate(concepts):
            safe_concept = concept.replace('"', '\\"')  # Escape quotes
            proof.append(f'    LOADC C{i}, "{safe_concept}"')
            
        # Add proof logic based on concept relationships
        edges = self.concept_graph.get_concept_edges()
        for edge in edges:
            if edge.source in concepts and edge.target in concepts:
                proof.append(f"    CORR {edge.source}, {edge.target}, {edge.weight}")
                
        # Add assertions
        proof.extend([
            "    ASSERT true",
            "    HLT",
            "}",
            "",
            f"Proof_{problem.replace(' ', '_')}()"
        ])
        
        return "\n".join(proof)

    def save_snapshot(self, loss, epoch=None, compress=True):
        # only ever keep the best
        if loss < self.best_loss:
            self.best_loss = loss
            fname = f"{self.__class__.__name__}_best.pt"
            if compress:
                fname += ".gz"
                path = os.path.join(self.config.snapshot_dir, fname)
                with gzip.open(path, "wb") as f:
                    torch.save(self._snapshot_dict(loss, epoch), f)
            else:
                torch.save(self._snapshot_dict(loss, epoch),
                           os.path.join(self.config.snapshot_dir, fname))

            # cleanup everything else
            for f in os.listdir(self.config.snapshot_dir):
                if f != fname:
                    os.remove(os.path.join(self.config.snapshot_dir, f))

    def _snapshot_dict(self, loss, epoch):
        return {
            'epoch': epoch or self.current_epoch,
            'model_state': self.state_dict(),
            'config': self.config,
            'concept_graph': self.concept_graph,
            'best_loss': self.best_loss,
            'scaler': self.scaler.state_dict()
        }

    def load_snapshot(self, path: str = None):
        # default to the best one
        if path is None:
            fname = f"{self.__class__.__name__}_best.pt"
            # check for gz
            gz = fname + ".gz"
            p1 = os.path.join(self.config.snapshot_dir, gz)
            p2 = os.path.join(self.config.snapshot_dir, fname)
            path = p1 if os.path.exists(p1) else p2

        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rb") as f:
            state = torch.load(f)
        self.load_state_dict(state['model_state'])
        self.concept_graph = state.get('concept_graph', ConceptGraph())
        self.scaler.load_state_dict(state.get('scaler', {}))
        self.best_loss = state.get('best_loss', float('inf'))
        self.current_epoch = state.get('epoch', 0)
        return self

    def train_step(self, batch, optimizer, epoch=None):
        inputs, targets = batch
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            preds = self(inputs)
            loss = self.hybrid_loss(preds, targets)
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()

        # update concept energies
        for c in self.concept_graph.concepts.values():
            c.energy = (c.energy + 0.1*(1 - loss.item())).clamp(0,1)

        # auto-save on new best
        if loss.item() < self.best_loss:
            self.save_snapshot(loss.item(), epoch)
        self.current_epoch = epoch if epoch is not None else self.current_epoch + 1
        loss_value = loss.item()

        # plugin hook
        self.plugin_manager.apply_on_train_step(self, batch, loss_value)
        return loss_value
