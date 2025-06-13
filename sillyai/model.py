import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import logging
import os
from datetime import datetime

from .core import Transformer
from .config import ModelConfig, Modality
from .ops import MultivectorOps
from .plugin import SillyPlugin
from .concept import ConceptGraph
from .complex_tokens import ComplexTokenizer
from .vm import BytecodeProgram, BytecodeEngine, Opcode
from .modalities import ModalityManager


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
        os.makedirs("logs", exist_ok=True)

        # Add file handler
        fh = logging.FileHandler(
            f"logs/sillyai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Initialize tokenizer
        self.tokenizer = ComplexTokenizer(num_basis=config.d_model)

        # Initialize transformer with concept graph
        self.transformer = Transformer(config, self.ops)

        # Initialize modality manager
        self.modality_manager = ModalityManager(config)

        # Initialize modality fusion layers
        self.fusion_layers = nn.ModuleDict()
        for modality in config.supported_modalities:
            if modality != Modality.TEXT:  # Text is handled by default transformer
                self.fusion_layers[modality.value] = nn.Sequential(
                    nn.Linear(config.d_model * 2, config.d_model),
                    nn.LayerNorm(config.d_model),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                )

        # Move model to device
        self.to(self.device)

    async def forward(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        mask: Optional[torch.Tensor] = None,
        generate_response: bool = False,
        num_tokens: int = 10,
    ) -> torch.Tensor:
        """Execute the full pipeline using the transformer.

        Args:
            x: Input tensor or dictionary of modality tensors
            mask: Optional attention mask
            generate_response: Whether to generate a response
            num_tokens: Number of tokens to generate

        Returns:
            Output tensor
        """
        if isinstance(x, dict):
            # Multi-modal input
            return await self._forward_multimodal(
                x, mask, generate_response, num_tokens
            )
        else:
            # Single modal input (assumed to be text)
            return await self.transformer(x, mask, generate_response, num_tokens)

    async def _forward_multimodal(
        self,
        inputs: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        generate_response: bool = False,
        num_tokens: int = 10,
    ) -> torch.Tensor:
        """Process multi-modal inputs and fuse them.

        Args:
            inputs: Dictionary mapping modality names to input tensors or raw inputs
            mask: Optional attention mask
            generate_response: Whether to generate a response
            num_tokens: Number of tokens to generate

        Returns:
            Fused output tensor
        """
        # Process text input first (required)
        if "text" not in inputs:
            raise ValueError("Text input is required for multi-modal processing")

        # Handle text input (could be raw text or pre-processed tensor)
        text_input = inputs["text"]
        if isinstance(text_input, (str, list)):
            text_features = self.modality_manager.process_text(text_input)
        else:
            text_features = text_input

        # Process through transformer
        text_features = await self.transformer(
            text_features, mask, generate_response=False
        )

        # Process other modalities and fuse with text
        fused_features = text_features
        for modality, tensor in inputs.items():
            if modality == "text":
                continue

            if modality not in self.config.supported_modalities:
                self.logger.warning(f"Unsupported modality: {modality}")
                continue

            # Process modality
            if modality == "image":
                modality_features = self.modality_manager.process_image(tensor)
            elif modality == "audio":
                modality_features = self.modality_manager.process_audio(tensor)
            else:
                # For other text-like modalities, process through text pipeline
                modality_features = self.modality_manager.process_text(tensor)

            # Fuse with text features
            fusion_layer = self.fusion_layers[modality]
            combined = torch.cat([fused_features, modality_features], dim=-1)
            fused_features = fusion_layer(combined)

        # Generate response if requested
        if generate_response:
            return await self.transformer(
                fused_features, mask, generate_response=True, num_tokens=num_tokens
            )

        return fused_features

    def add_modality(self, name: str, modality_config: dict):
        """Add a new modality to the model.

        Args:
            name: Name of the modality
            modality_config: Configuration for the modality
        """
        self.modality_manager.add_modality(name, modality_config)

    def process_text(self, text: Union[str, List[str]]) -> torch.Tensor:
        """Process text input using Word2VecTokenizer.

        Args:
            text: Text or list of texts to process

        Returns:
            Processed text features
        """
        return self.modality_manager.process_text(text)

    def process_image(self, images: torch.Tensor) -> torch.Tensor:
        """Process image input.

        Args:
            images: Image tensor [B, C, H, W]

        Returns:
            Processed image features
        """
        return self.modality_manager.process_image(images)

    def process_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Process audio input.

        Args:
            audio: Audio tensor [B, C, T] where:
                  B = batch size
                  C = number of channels (1=mono, 2=stereo)
                  T = number of time steps

        Returns:
            Processed audio features
        """
        return self.modality_manager.process_audio(audio)

    def save(self, path: str):
        """Save model state."""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": self.config,
                "concept_graph": self.transformer.concept_graph,
                "modality_manager": self.modality_manager,
            },
            path,
        )

        self.logger.info(f"Saved model to {path}")

    def load(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.transformer.concept_graph = checkpoint["concept_graph"]
        self.modality_manager = checkpoint["modality_manager"]

        self.logger.info(f"Loaded model from {path}")

    def print_concepts(self, top_k: int = 10):
        """Print top concepts from the concept graph."""
        concepts = self.transformer.concept_graph.get_top_concepts(top_k)

        self.logger.info(f"Top {top_k} concepts:")
        for i, (concept, score) in enumerate(concepts):
            self.logger.info(f"{i + 1}. {concept}: {score:.4f}")

    def generate_bytecode(self) -> List[Tuple[str, List[float]]]:
        """Generate bytecode from concept graph."""
        return self.transformer.concept_graph.to_bytecode()

    @property
    def concept_graph(self):
        """Access the concept graph."""
        return self.transformer.concept_graph

    @concept_graph.setter
    def concept_graph(self, value):
        """Set the concept graph."""
        self.transformer.concept_graph = value

    async def generate_response(
        self,
        input_tensor: torch.Tensor,
        num_tokens: int = 10,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
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
                input_tensor, generate_response=True, num_tokens=num_tokens
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
        min_score: float = 15.0,
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
        best_score = -float("inf")

        for attempt in range(max_retries):
            # Generate response
            response = await self.generate_response(input_tensor, num_tokens=num_tokens)

            # Score the response
            candidates = await self.transformer.response_generator.generate_candidates(
                response,
                num_tokens=1,  # Just score the response
            )

            if candidates:
                score = candidates[0].score
                if score > best_score:
                    best_score = score
                    best_response = response

                if score >= min_score:
                    self.logger.info(
                        f"Generated acceptable response on attempt {attempt + 1}"
                    )
                    return response

            self.logger.warning(f"Attempt {attempt + 1} failed with score {score:.2f}")

        if best_response is not None:
            self.logger.warning(f"Using best response with score {best_score:.2f}")
            return best_response

        raise RuntimeError("Failed to generate acceptable response after all retries")
