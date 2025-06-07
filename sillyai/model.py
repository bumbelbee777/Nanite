import torch
import torch.nn as nn
import torch.functional as F
from typing import Optional, Dict, List, Union, Any
import weakref

from .config import ModelConfig
from .core import Transformer
from .concept import ConceptGraph
from .plugin import PluginManager
from .ops import MultivectorOps

class SillyAI(nn.Module):
    """Main SillyAI model interface combining transformer, concept graph, and plugins."""
    
    def __init__(self, config: ModelConfig, ops: Optional[MultivectorOps] = None):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize core components
        self.ops = ops or MultivectorOps().compile()
        self.concept_graph = ConceptGraph(max_size=config.concept_graph_size, decay_rate=0.1)
        self.transformer = Transformer(config, self.ops, self.concept_graph)
        
        # Plugin system
        self.plugin_manager = PluginManager()
        for plugin_name in config.enabled_plugins:
            self.plugin_manager.load_plugin(plugin_name)
            
        # Training components
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scaler = torch.amp.GradScaler(device=self.device)
        
        # Move model to device
        self.to(self.device)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the model."""
        # Plugin hook: before_forward
        x = self.plugin_manager.call_hook('before_forward', x)
        
        # Main forward pass
        output = self.transformer(x, mask)
        
        # Plugin hook: after_forward
        output = self.plugin_manager.call_hook('after_forward', output)
        
        return output
        
    def train_step(self, x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
        """Single training step with mixed precision."""
        self.optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            output = self(x, mask)
            loss = F.mse_loss(output, y)
            
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
        
    def update_concept_graph(self, concepts: List[Dict[str, Any]]):
        """Update the concept graph with new concepts."""
        for concept in concepts:
            self.concept_graph.add_concept(**concept)
            
    def save(self, path: str):
        """Save model state."""
        state = {
            'model_state': self.state_dict(),
            'config': self.config,
            'concept_graph': self.concept_graph,
            'plugin_states': self.plugin_manager.get_states()
        }
        torch.save(state, path)
        
    def load(self, path: str):
        """Load model state."""
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state['model_state'])
        self.concept_graph = state['concept_graph']
        self.plugin_manager.load_states(state['plugin_states'])
        
    def generate_bytecode(self, start_concept: Optional[str] = None, max_hops: int = 5) -> List[tuple]:
        """Generate bytecode from concept graph."""
        return self.concept_graph.to_bytecode(start_concept, max_hops)
        
    def get_concept_graph(self) -> ConceptGraph:
        """Get the current concept graph."""
        return self.concept_graph
        
    def print_concepts(self, top_k: int = 10):
        """Print top concepts by centrality."""
        concepts = self.concept_graph.get_top_concepts(top_k)
        print("\nTop concepts by centrality:")
        for name, score in concepts:
            print(f"  {name}: {score:.3f}")
            
    def optimize_for_cpu(self):
        """Optimize model for CPU inference."""
        torch.set_num_threads(max(1, torch.get_num_threads() // 2))
        torch.set_num_interop_threads(1)
        
    def optimize_for_low_memory(self):
        """Enable memory optimizations."""
        for module in self.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True 