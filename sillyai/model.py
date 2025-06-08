import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Union
import logging
import os
import json
from datetime import datetime

from .core import Transformer
from .config import ModelConfig, Modality, PrecisionLevel
from .ops import MultivectorOps
from .plugin import SillyPlugin

class SillyAI(nn.Module):
    """Exported SillyAI class for interfacing with the model."""
    
    def __init__(self, config: ModelConfig, ops: Optional[MultivectorOps] = None):
        super().__init__()
        self.config = config
        self.ops = ops or MultivectorOps()
        self.device = torch.device(config.device)
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create log directory if it doesn't exist
        os.makedirs('logs', exist_ok=True)
        
        # Add file handler
        fh = logging.FileHandler(f'logs/sillyai_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        # Initialize the core Transformer model
        self.transformer = Transformer(config, self.ops)
        
        # Initialize plugins
        self.plugins: Dict[str, SillyPlugin] = {}
        self._load_plugins()
        
        # Move model to device
        self.to(self.device)
        
    def _load_plugins(self):
        """Load enabled plugins."""
        for plugin_name in self.config.enabled_plugins:
            try:
                if plugin_name == 'trainer':
                    from sillyai.plugins.trainer import SillyAITrainerPlugin
                    self.plugins['trainer'] = SillyAITrainerPlugin(self.config)
                elif plugin_name == 'visualizer':
                    from sillyai.plugins.visualizer import SillyAIVisualizerPlugin
                    self.plugins['visualizer'] = SillyAIVisualizerPlugin()
                # Add more plugins here as needed
                
                # Initialize plugin
                plugin = self.plugins[plugin_name]
                plugin.on_init(self)
                self.logger.info(f"Loaded plugin: {plugin_name}")
            except Exception as e:
                self.logger.error(f"Failed to load plugin {plugin_name}: {str(e)}")
                
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass through the model."""
        # Notify plugins before forward pass
        for plugin in self.plugins.values():
            plugin.before_forward(x)
            
        # Forward through transformer
        output = self.transformer(x, mask)
        
        # Notify plugins after forward pass
        for plugin in self.plugins.values():
            plugin.after_forward(output)
            
        return output
        
    def save(self, path: str):
        """Save model state and plugins."""
        # Save model state
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'transformer_state_dict': self.transformer.state_dict()
        }, path)
        
        # Save plugin states
        for name, plugin in self.plugins.items():
            plugin.on_save(path)
            
        self.logger.info(f"Saved model to {path}")
        
    def load(self, path: str):
        """Load model state and plugins."""
        # Load model state
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.transformer.load_state_dict(checkpoint['transformer_state_dict'])
        
        # Load plugin states
        for name, plugin in self.plugins.items():
            plugin.on_load(path)
            
        self.logger.info(f"Loaded model from {path}")
        
    def print_concepts(self, top_k: int = 10):
        """Print top concepts from the concept graph."""
        if not hasattr(self.transformer, 'concept_graph') or not self.transformer.concept_graph:
            self.logger.warning("Concept graph is empty")
            return
            
        # Sort concepts by frequency
        sorted_concepts = sorted(
            self.transformer.concept_graph.items(),
            key=lambda x: x[1]['frequency'],
            reverse=True
        )
        
        # Print top k concepts
        self.logger.info(f"Top {top_k} concepts:")
        for i, (concept, data) in enumerate(sorted_concepts[:top_k]):
            self.logger.info(f"{i+1}. {concept}: {data['frequency']} occurrences")
            
    def generate_bytecode(self) -> List[Tuple[str, List[float]]]:
        """Generate bytecode from concept graph."""
        bytecode = []
        
        if not hasattr(self.transformer, 'concept_graph') or not self.transformer.concept_graph:
            return bytecode
            
        # Sort concepts by frequency
        sorted_concepts = sorted(
            self.transformer.concept_graph.items(),
            key=lambda x: x[1]['frequency'],
            reverse=True
        )
        
        # Generate bytecode for top concepts
        for concept, data in sorted_concepts:
            # Get concept embedding
            embedding = self.transformer.concept_embeddings[data['id']].detach().cpu().numpy()
            
            # Add to bytecode
            bytecode.append((concept, embedding.tolist()))
            
        return bytecode

    @property
    def concept_graph(self):
        """Access the concept graph from the transformer."""
        return getattr(self.transformer, 'concept_graph', {})
        
    @concept_graph.setter
    def concept_graph(self, value):
        """Set the concept graph in the transformer."""
        self.transformer.concept_graph = value 

    async def generate_response(
        self,
        input_tensor: torch.Tensor,
        num_tokens: int = 10,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> torch.Tensor:
        """Generate a response using the multi-token prediction system.
        
        Args:
            input_tensor: Input tensor [batch_size, seq_len, input_dim]
            num_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Number of top tokens to consider
            top_p: Nucleus sampling probability
            
        Returns:
            Generated response tensor [batch_size, num_tokens, output_dim]
        """
        # Move input to device
        input_tensor = input_tensor.to(self.device)
        
        # Set temperature for TGUs
        for tgu in self.transformer.response_generator.tgus:
            tgu.temperature = temperature
            
        # Generate response
        with torch.no_grad():
            output = await self.transformer(
                input_tensor,
                generate_response=True,
                num_tokens=num_tokens
            )
            
        # Log generation metrics
        self.logger.info(f"Generated response with {num_tokens} tokens")
        self.logger.info(f"Temperature: {temperature}, Top-k: {top_k}, Top-p: {top_p}")
        
        return output
        
    async def generate_with_retry(
        self,
        input_tensor: torch.Tensor,
        num_tokens: int = 10,
        max_retries: int = 3,
        min_score: float = 15.0
    ) -> torch.Tensor:
        """Generate response with retries if quality is insufficient.
        
        Args:
            input_tensor: Input tensor [batch_size, seq_len, input_dim]
            num_tokens: Number of tokens to generate
            max_retries: Maximum number of generation retries
            min_score: Minimum acceptable score
            
        Returns:
            Generated response tensor [batch_size, num_tokens, output_dim]
        """
        best_response = None
        best_score = -float('inf')
        
        for attempt in range(max_retries):
            # Generate response
            response = await self.generate_response(
                input_tensor,
                num_tokens=num_tokens
            )
            
            # Score the response
            candidates = await self.transformer.response_generator.generate_candidates(
                response, num_tokens=1  # Just score the response
            )
            
            if candidates:
                score = candidates[0].score
                if score > best_score:
                    best_score = score
                    best_response = response
                    
                if score >= min_score:
                    self.logger.info(f"Generated acceptable response on attempt {attempt + 1}")
                    return response
                    
            self.logger.warning(f"Attempt {attempt + 1} failed with score {score:.2f}")
            
        if best_response is not None:
            self.logger.warning(f"Using best response with score {best_score:.2f}")
            return best_response
            
        raise RuntimeError("Failed to generate acceptable response after all retries") 