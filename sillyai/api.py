import os
import gzip
import json
import logging
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core.complex.linear import ComplexLinear
from .core.transformer import PositionalEncoding, TransformerBlock
from .core.config import ModelConfig
from .core.routing import TaskComplexityEstimator, FeatureRouter
from .plugins.plugin import PluginManager
from .graph.concept import ConceptGraph
from .core.vm import SillyVM
from .modalities.vision.image import ImageModality
from .modalities.text.tokenizer import Word2VecTokenizer

logger = logging.getLogger(__name__)

class SillyAI(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Validate dimensions
        if config.input_dim <= 0 or config.d_model <= 0 or config.output_dim <= 0:
            raise ValueError(
                f"Invalid dimensions: input_dim={config.input_dim}, "
                f"d_model={config.d_model}, output_dim={config.output_dim}"
            )
        
        # --- Fix kronecker_rank for ComplexLinear ---
        def valid_kronecker_rank(rank, in_dim, out_dim):
            # Always at least 1, at most min(in_dim, out_dim)//2 or 4
            max_rank = max(1, min(in_dim, out_dim) // 2)
            # If min(in_dim, out_dim) < 4, allow rank=1
            return max(1, min(rank, max_rank))
        
        # Core components with proper dimension handling
        # Input projection: [B, L, input_dim] -> [B, L, d_model, 2]
        self.input_proj = ComplexLinear(
            in_features=config.input_dim,
            out_features=config.d_model,
            factorized=config.factorized_linear,
            kronecker_rank=valid_kronecker_rank(
                config.kronecker_rank,
                config.input_dim,
                config.d_model
            )
        )
        
        # Output projection: [B, L, d_model, 2] -> [B, L, output_dim, 2]  
        self.output_proj = ComplexLinear(
            in_features=config.d_model,
            out_features=config.output_dim,
            factorized=config.factorized_linear,
            kronecker_rank=valid_kronecker_rank(
                config.kronecker_rank,
                config.d_model,
                config.output_dim
            )
        )
        
        # Initialize pos encoding and layers
        self.pos_enc = PositionalEncoding(config.d_model)  # Takes d_model directly
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Concept components with proper dimensions
        self.concept_graph = ConceptGraph()
        self.concept_projector = ComplexLinear(
            in_features=config.d_model,
            out_features=config.concept_dim,
            factorized=config.factorized_linear,
            kronecker_rank=valid_kronecker_rank(
                config.kronecker_rank,
                config.d_model,
                config.concept_dim
            )
        )
        
        # Task routing components
        self.complexity_estimator = TaskComplexityEstimator(config)
        self.pos_enc_router = FeatureRouter(config)
        
        # Plugin system
        self.plugin_manager = PluginManager(config.plugin_dir)
        self.plugin_manager.discover()
        
        # VM for proof verification
        self.vm = SillyVM()

        # Initialize image modality if specified
        if hasattr(config, 'image_modality') and config.image_modality:
            self.image_modality = ImageModality(
                input_channels=config.image_modality['input_channels'],
                output_dim=config.d_model,
                input_size=config.image_modality.get('input_size', (32, 32)),
                dropout_rate=config.image_modality.get('dropout_rate', 0.5)
            )
        else:
            self.image_modality = None

        # Initialize text modality if specified
        if hasattr(config, 'text_modality') and config.text_modality:
            self.text_tokenizer = Word2VecTokenizer(
                w2v_path=config.text_modality['w2v_path'],
                unk_token=config.text_modality.get('unk_token', '<UNK>'),
                lowercase=config.text_modality.get('lowercase', True)
            )
        else:
            self.text_tokenizer = None

    def _make_proj(self, in_dim: int, out_dim: int) -> nn.Module:
        return ComplexLinear(
            in_dim, 
            out_dim, 
            factorized=self.config.factorized_linear,
            kronecker_rank=self.config.kronecker_rank
        )
    
    def _forward_core(self, x):
        # Project input
        x = self.input_proj(x)
        
        # Route through positional encoding
        x = self.pos_enc_router(x, self.pos_enc)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
            
        return x

    def forward(self, x):
        """Forward pass with input shape [B, L, D, 2]"""
        out = self._forward_core(x)
        return self.output_proj(out)

    def get_concept_projections(self, x: torch.Tensor) -> torch.Tensor:
        """Project input features onto concept space.
        Args:
            x: Input tensor of shape [B, L, D, 2]
        Returns:
            Concept projections of shape [B, L, C, 2] where C is concept_dim
        """
        # Project inputs to concept space using concept_projector
        concept_feats = self.concept_projector(x)  # [B, L, C, 2]
        
        # Apply magnitude scaling
        norms = torch.norm(concept_feats, dim=-1, keepdim=True)
        concept_feats = concept_feats / (norms + 1e-8)
        
        return concept_feats

    def hybrid_loss(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute hybrid loss combining MSE and concept alignment."""
        # Get model predictions
        preds = self(x)  # [B, L, O, 2]
        
        # MSE loss on predictions
        mse_loss = F.mse_loss(preds, targets)
        
        # Get concept projections and compute alignment loss
        concept_proj = self.get_concept_projections(x)  # [B, L, C, 2]
        
        # Compute energy-based weighting for concepts
        concept_weights = []
        for name, concept in self.concept_graph.concepts.items():
            concept_weights.append(concept.energy)
        concept_weights = torch.tensor(concept_weights, device=x.device)
        concept_weights = F.softmax(concept_weights, dim=0)
        
        # Compute weighted norm of concept projections
        alignment_loss = -torch.mean(
            concept_weights.view(1, 1, -1, 1) * 
            torch.norm(concept_proj, dim=-1)
        )
        
        # Combine losses with weighting
        total_loss = mse_loss + self.config.concept_loss_weight * alignment_loss
        return total_loss

    def train_step(self, batch, optimizer, epoch=None):
        """Single training step with gradient clipping."""
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            x, y = batch
            loss = self.hybrid_loss(x, y)
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.grad_clip)
        optimizer.step()
        
        return loss.item()

    def solve_problem(self, problem: str, proof_path: Optional[str] = None) -> bool:
        """Solve a problem by generating and verifying a proof."""
        # Generate proof if not provided
        if proof_path is None:
            proof_code = self._generate_proof(problem)
            return self._verify_proof_code(proof_code)
        return self._verify_proof(proof_path)
            
    def _verify_proof(self, proof_path: str) -> bool:
        """Verify a proof file by executing in VM."""
        if not os.path.exists(proof_path):
            return False
        with open(proof_path, 'r') as f:
            proof_code = f.read()
        return self._verify_proof_code(proof_code)
            
    def _verify_proof_code(self, proof_code: str) -> bool:
        """Execute proof code in VM and verify result."""
        try:
            self.vm.parse(proof_code)
            return self.vm.execute()
        except Exception as e:
            logger.error(f"Proof verification failed: {e}")
            return False
            
    def _generate_proof(self, problem: str) -> str:
        """Generate proof code for a given problem."""
        # TODO: Implement proof generation
        raise NotImplementedError

    def save_snapshot(self, loss, epoch=None, compress=True):
        """Save model snapshot with metadata."""
        os.makedirs(self.config.snapshot_dir, exist_ok=True)
        path = os.path.join(self.config.snapshot_dir, f"{self.__class__.__name__}_best.pt")
        
        # Create snapshot dict
        snapshot = self._snapshot_dict(loss, epoch)
        
        # Save compressed or uncompressed
        if compress:
            with gzip.open(path + '.gz', 'wb') as f:
                torch.save(snapshot, f)
        else:
            torch.save(snapshot, path)
        
        # Store best loss for future reference
        self.best_loss = loss
        
        logger.info(f"Saved snapshot to {path}")
        return path

    def _snapshot_dict(self, loss, epoch):
        """Create snapshot dictionary with all relevant state."""
        return {
            'model_state': self.state_dict(),
            'config': self.config.__dict__,
            'loss': loss,
            'epoch': epoch,
            'concept_graph': self.concept_graph,
            'plugins': self.plugin_manager.active_plugins,
            'timestamp': datetime.now().isoformat()
        }

    def load_snapshot(self, path: str = None):
        """Load model snapshot."""
        if path is None:
            # Find latest snapshot
            snapshots = [f for f in os.listdir('snapshots') if f.endswith('.pt')]
            if not snapshots:
                return
            path = os.path.join('snapshots', max(snapshots))
            
        # Handle compressed snapshots
        if path.endswith('.gz'):
            with gzip.open(path, 'rb') as f:
                snapshot = torch.load(f)
        else:
            torch.load(path)
            
        # Load state
        self.load_state_dict(snapshot['model_state'])
        self.config.__dict__.update(snapshot['config'])
        self.concept_graph = snapshot['concept_graph']
        self.best_loss = snapshot['loss']  # Store the loss value
        
        # Restore plugins
        for plugin in snapshot['plugins']:
            self.plugin_manager.enable_plugin(plugin)
            
        logger.info(f"Loaded snapshot from {path}")
        return snapshot

    def process_image(self, image_path):
        if not self.image_modality:
            raise ValueError("Image modality is not enabled in the configuration.")
        image_tensor = ImageModality.load_and_process_image(
            image_path, self.image_modality.input_size
        )
        return self.image_modality(image_tensor)

    def process_text(self, text):
        if not self.text_tokenizer:
            raise ValueError("Text modality is not enabled in the configuration.")
        token_indices = self.text_tokenizer.encode(text)
        embeddings = self.text_tokenizer.embed_indices(token_indices)
        return torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0)
