import os
import gzip
import logging
from datetime import datetime
from typing import Optional, List, Union, Dict

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
        
        # Initialize core components
        self._init_core_components()
        
        # Initialize modalities if specified in config
        if hasattr(config, 'modalities'):
            self._init_modalities(config.modalities)
        
        # Initialize concept system
        self.concept_graph = ConceptGraph()
        self._init_concept_projector()
        
        # Initialize routing and plugin systems
        self.complexity_estimator = TaskComplexityEstimator(config)
        self.pos_enc_router = FeatureRouter(config)
        self.plugin_manager = PluginManager(config.plugin_dir)
        self.plugin_manager.discover()
        
        # Initialize VM
        self.vm = SillyVM()
        self.proof_cache = {}

    def _init_core_components(self):
        """Initialize the core transformer components"""
        # Input projection handles both complex and real inputs
        self.input_proj = ComplexLinear(
            self.config.input_dim,
            self.config.d_model,
            factorized=self.config.factorized_linear,
            kronecker_rank=self._valid_kronecker_rank(
                self.config.kronecker_rank,
                self.config.input_dim,
                self.config.d_model
            )
        )
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(self.config.d_model)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(self.config) 
            for _ in range(self.config.num_layers)
        ])
        
        # Output projection
        self.output_proj = ComplexLinear(
            self.config.d_model,
            self.config.output_dim,
            factorized=self.config.factorized_linear,
            kronecker_rank=self._valid_kronecker_rank(
                self.config.kronecker_rank,
                self.config.d_model,
                self.config.output_dim
            )
        )

    def _init_modalities(self, modality_config):
        """Initialize modality-specific components"""
        # Image modality
        if modality_config.get('image'):
            self.image_encoder = ImageModality(
                input_channels=modality_config['image'].get('input_channels', 3),
                output_dim=self.config.d_model,
                input_size=modality_config['image'].get('input_size', (224, 224)),
                dropout_rate=modality_config['image'].get('dropout_rate', 0.1)
            )
            self.image_proj = ComplexLinear(
                self.config.d_model,
                self.config.d_model,
                factorized=self.config.factorized_linear
            )
        
        # Text modality
        if modality_config.get('text'):
            self.text_tokenizer = Word2VecTokenizer(
                w2v_path=modality_config['text']['w2v_path'],
                unk_token=modality_config['text'].get('unk_token', '<UNK>'),
                pad_token=modality_config['text'].get('pad_token', '<PAD>'),
                lowercase=modality_config['text'].get('lowercase', True),
                device=self.config.device
            )
            self.text_proj = ComplexLinear(
                self.text_tokenizer.get_embedding_matrix().shape[1],
                self.config.d_model,
                factorized=self.config.factorized_linear
            )

    def _init_concept_projector(self):
        """Initialize concept projection system"""
        self.concept_projector = ComplexLinear(
            self.config.d_model,
            self.config.concept_dim,
            factorized=self.config.factorized_linear,
            kronecker_rank=self._valid_kronecker_rank(
                self.config.kronecker_rank,
                self.config.d_model,
                self.config.concept_dim
            )
        )

    def _valid_kronecker_rank(self, rank, in_dim, out_dim):
        """Ensure valid Kronecker rank with safety checks"""
        max_rank = min(in_dim, out_dim)
        return min(max(1, rank), max(1, max_rank // 2))

    def forward(
        self,
        inputs: Union[torch.Tensor, Dict[str, torch.Tensor], str, List[str]],
        input_type: Optional[str] = None
    ):
        """
        Unified forward pass handling:
        - Raw tensors (auto-detected as image or text)
        - Text strings
        - Dictionary with {'text':..., 'image':...}
        """
        # Automatic input type detection
        if input_type is None:
            if isinstance(inputs, dict):
                input_type = 'multimodal'
            elif isinstance(inputs, str) or (isinstance(inputs, list) and isinstance(inputs[0], str)):
                input_type = 'text'
            elif inputs.dim() == 4:  # Image tensor [B,C,H,W]
                input_type = 'image'
            else:
                input_type = 'default'  # Use core processing

        # Process based on detected type
        if input_type == 'text':
            return self._process_text(inputs)
        elif input_type == 'image':
            return self._process_image(inputs)
        elif input_type == 'multimodal':
            return self._process_multimodal(inputs)
        else:
            return self._process_core(inputs)

    def _process_core(self, x):
        """Process through core transformer pipeline"""
        x = self.input_proj(x)
        x = self.pos_enc_router(x, self.pos_enc)
        
        for layer in self.layers:
            x = layer(x)
            
        return self.output_proj(x)

    def _process_text(self, text):
        """Process text input through text modality pipeline"""
        if not hasattr(self, 'text_tokenizer'):
            raise ValueError("Text modality not initialized")
            
        # Handle both single string and batch
        if isinstance(text, str):
            text = [text]
            
        # Tokenize and embed
        encoded = self.text_tokenizer.batch_encode(text)
        text_emb = self.text_tokenizer.embed(encoded['input_ids'])
        
        # Project and process through core
        x = self.text_proj(text_emb)
        return self._process_core(x)

    def _process_image(self, images):
        """Process image input through vision pipeline"""
        if not hasattr(self, 'image_encoder'):
            raise ValueError("Image modality not initialized")
            
        # Extract features and project
        img_features = self.image_encoder(images)
        x = self.image_proj(img_features)
        
        # Process through core
        return self._process_core(x)

    def _process_multimodal(self, inputs):
        """Fuse multiple modalities"""
        # Process each modality
        text_emb = self.text_proj(
            self.text_tokenizer.embed(inputs['text'])
        ) if 'text' in inputs else None
        
        img_emb = self.image_proj(
            self.image_encoder(inputs['image'])
        ) if 'image' in inputs else None
        
        # Fuse modalities
        if text_emb is not None and img_emb is not None:
            # Simple concatenation fusion (can be enhanced)
            x = torch.cat([text_emb, img_emb], dim=-1)
            x = self.fusion(x) if hasattr(self, 'fusion') else x
        elif text_emb is not None:
            x = text_emb
        else:
            x = img_emb
            
        return self._process_core(x)

    def _make_proj(self, in_dim: int, out_dim: int) -> nn.Module:
        return ComplexLinear(
            in_dim, 
            out_dim, 
            factorized=self.config.factorized_linear,
            kronecker_rank=self.config.kronecker_rank
        )

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
            proof_code = self._generate_proof_with_text_tokenizer(problem)
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

    def parse_problem(self, problem: str) -> Optional[Dict]:
        """Parse a problem statement into the concept graph."""
        # Example: Simple rule-based parsing for arithmetic problems
        if "sum of" in problem and "greater than" in problem:
            root = self.add_concept("SumGreaterThan")
            self.add_relationship(root, "a", {"value": 3})
            self.add_relationship(root, "b", {"value": 4})
            self.add_relationship(root, "c", {"value": 5})
            return root
        else:
            # Return None for unsupported problems
            return None

    def parse_problem_with_text_tokenizer(self, problem: str) -> Optional[Dict]:
        """Parse a problem statement into the concept graph using the text tokenizer."""
        if not hasattr(self, 'text_tokenizer'):
            raise ValueError("Text modality not initialized")

        # Tokenize and embed the problem statement
        tokens = self.text_tokenizer.tokenize(problem)
        embeddings = self.text_tokenizer.embed(tokens)

        # Example: Use embeddings to identify key concepts and relationships
        root = self.concept_graph.add_concept("ParsedProblem")
        for token, embedding in zip(tokens, embeddings):
            # Add each token as a concept and link it to the root
            concept = self.concept_graph.add_concept(token, metadata={"embedding": embedding})
            self.concept_graph.add_relationship(root, concept, {"type": "contains"})

        return root

    def _generate_proof_with_text_tokenizer(self, problem: str) -> str:
        """Generate proof code for a given problem using the text tokenizer and concept graph."""
        used_registers = set()
        proof_code = ".Proof() {\n"

        # Step 1: Parse the problem into the concept graph using the text tokenizer
        root_concept = self.parse_problem_with_text_tokenizer(problem)
        if not root_concept:
            proof_code += "    ASSERT false  // Unable to parse problem\n"
            proof_code += "    HLT\n"
            proof_code += "}\n"
            return proof_code

        # Step 2: Identify the relevant subgraph
        relevant_subgraph = self.concept_graph.get_relevant_subgraph(root_concept)
        if not relevant_subgraph:
            proof_code += "    ASSERT false  // No relevant subgraph found\n"
            proof_code += "    HLT\n"
            proof_code += "}\n"
            return proof_code

        # Step 3: Walk the subgraph and generate bytecode
        for node in relevant_subgraph.walk():
            operation = node.get("operation")
            operands = node.get("operands", [])
            target = node.get("target")
            comment = node.get("comment", "")

            # Allocate registers for operands and target
            operand_registers = [self._allocate_register(used_registers) for _ in operands]
            target_register = self._allocate_register(used_registers)

            # Generate bytecode for the operation
            if operation == "assign":
                proof_code += f"    {target_register} = {operands[0]}  // {comment}\n"
            elif operation == "add":
                proof_code += f"    {target_register} = ADD {operand_registers[0]}, {operand_registers[1]}  // {comment}\n"
            elif operation == "compare":
                proof_code += f"    CJM {operand_registers[0]} {node['operator']} {operand_registers[1]}, Fail()  // {comment}\n"
            # Add more operations as needed

        # Step 4: Add success and failure handlers
        proof_code += "    ASSERT true  // Proof holds\n"
        proof_code += "    HLT\n"
        proof_code += "}\n\n"
        proof_code += ".Fail() {\n"
        proof_code += "    ASSERT false  // Proof failed\n"
        proof_code += "    HLT\n"
        proof_code += "}\n"

        return proof_code

    def _allocate_register(self, used_registers: set, register_type: str = 'R') -> str:
        """Allocate a new register of the specified type that is not in use."""
        reg_index = 0
        while True:
            reg = f"{register_type}{reg_index}"
            if reg not in used_registers:
                used_registers.add(reg)
                return reg
            reg_index += 1

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

    def generate_response(self, prompt: Union[str, Dict[str, torch.Tensor]]) -> str:
        """
        Generate a human-like response based on the input prompt.
        
        Args:
            prompt: Input text, image, or multimodal data.
        
        Returns:
            A human-like response as a string.
        """
        # Step 1: Detect input type and preprocess
        if isinstance(prompt, str):
            input_type = 'text'
            processed_input = self._process_text(prompt)
        elif isinstance(prompt, dict) and 'image' in prompt:
            input_type = 'multimodal'
            processed_input = self._process_multimodal(prompt)
        else:
            raise ValueError("Unsupported input type. Provide text or multimodal input.")

        # Step 2: Perform reasoning if needed
        reasoning_result = None
        if input_type == 'text' and "why" in prompt.lower():
            # Example: Use SillyISA for reasoning if the prompt is a question
            problem_statement = f"Explain: {prompt}"
            try:
                reasoning_result = self.solve_problem(problem_statement)
            except NotImplementedError:
                reasoning_result = "Reasoning not implemented yet."

        # Step 3: Generate response candidates
        response_candidates = []
        if input_type == 'text':
            response_candidates.append(f"Processed text: {prompt[::-1]}")  # Example placeholder
        if reasoning_result:
            response_candidates.append(f"Reasoning result: {reasoning_result}")
        if hasattr(self, 'concept_graph'):
            related_concepts = list(self.concept_graph.concepts.keys())[:3]  # Example: top 3 concepts
            response_candidates.append(f"Related concepts: {', '.join(related_concepts)}")

        # Step 4: Reflect and select the best response
        best_response = max(response_candidates, key=len)  # Example: pick the longest response

        return best_response

    def describe_image(self, image_path: str) -> str:
        """
        Generate a textual description of an image.

        Args:
            image_path: Path to the image file.

        Returns:
            A textual description of the image.
        """
        if not hasattr(self, 'image_encoder'):
            raise ValueError("Image modality is not initialized")

        # Load and process the image
        image_tensor = ImageModality.load_and_process_image(
            image_path, self.config.modalities['image'].get('input_size', (224, 224))
        )

        # Encode the image
        image_features = self.image_encoder(image_tensor)

        # Project the features through the core pipeline
        processed_features = self._process_core(self.image_proj(image_features))

        # Generate a description (placeholder logic for now)
        description = f"This image contains features with mean value {processed_features.mean().item():.2f}."

        return description
