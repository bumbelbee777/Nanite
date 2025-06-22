import logging
import os
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import traceback

from .complex_tokens import ComplexTokenizer
from .config import Modality, ModelConfig
from .core import Transformer
from .modalities import ModalityManager
from .ops import MultivectorOps, ComplexLoss


class Nanite(nn.Module):
    """Exported Nanite class for interfacing with the model."""

    def __init__(self, config: ModelConfig, ops: MultivectorOps | None = None):
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
            f"logs/nanite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Initialize transformer with concept graph
        self.transformer = Transformer(config, self.ops)

        # Initialize modality manager
        self.modality_manager = ModalityManager(config)
        
        # Ensure text modality is initialized
        if "text" not in self.modality_manager.modalities:
            self.modality_manager.add_modality("text", {"type": "text"})

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
        x: torch.Tensor | dict[str, torch.Tensor],
        mask: torch.Tensor | None = None,
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
                x,
                mask,
                generate_response,
                num_tokens,
            )
        else:
            # Single modal input (assumed to be text)
            return await self.transformer(x, mask, generate_response, num_tokens)

    async def _forward_multimodal(
        self,
        inputs: dict[str, torch.Tensor],
        mask: torch.Tensor | None = None,
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
            text_features = await self.modality_manager.process_text(text_input)
        else:
            text_features = text_input

        # Process through transformer
        text_features = await self.transformer(
            text_features,
            mask,
            generate_response=False,
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
                modality_features = await self.modality_manager.process_text(tensor)

            # Fuse with text features
            fusion_layer = self.fusion_layers[modality]
            combined = torch.cat([fused_features, modality_features], dim=-1)
            fused_features = fusion_layer(combined)

        # Generate response if requested
        if generate_response:
            return await self.transformer(
                fused_features,
                mask,
                generate_response=True,
                num_tokens=num_tokens,
            )

        return fused_features

    def add_modality(self, name: str, modality_config: dict):
        """Add a new modality to the model.

        Args:
            name: Name of the modality
            modality_config: Configuration for the modality
        """
        self.modality_manager.add_modality(name, modality_config)
        # Ensure the modality is properly initialized
        if name == "text" and name not in self.modality_manager.modalities:
            raise ValueError("Text modality not properly initialized!")
        elif name == "text" and "encoder" not in self.modality_manager.modalities[name]:
            raise ValueError("Text encoder not properly initialized!")

    async def process_text(self, text: str | list[str]) -> torch.Tensor:
        """Process text input using SubwordTokenizer.

        Args:
            text: Text or list of texts to process

        Returns:
            Processed text features
        """
        return await self.modality_manager.process_text(text)

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

    def generate_bytecode(self) -> list[tuple[str, list[float]]]:
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
                input_tensor,
                generate_response=True,
                num_tokens=num_tokens,
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
                        f"Generated acceptable response on attempt {attempt + 1}",
                    )
                    return response

            self.logger.warning(f"Attempt {attempt + 1} failed with score {score:.2f}")

        if best_response is not None:
            self.logger.warning(f"Using best response with score {best_score:.2f}")
            return best_response

        raise RuntimeError("Failed to generate acceptable response after all retries")

    async def train_step(self, text: str) -> float:
        """Perform a single training step on a text input.
        
        Args:
            text: Input text to train on
            
        Returns:
            Loss value for this step
        """
        def ensure_tensor(x):
            if isinstance(x, torch.Tensor):
                print(f"[DEBUG] ensure_tensor: already torch.Tensor, shape={x.shape}")
                return x
            elif isinstance(x, np.ndarray):
                print(f"[DEBUG] ensure_tensor: np.ndarray, shape={x.shape}, dtype={x.dtype}")
                if x.dtype == object:
                    print(f"[DEBUG] Cannot convert object array to tensor directly: {x}")
                    raise TypeError("Cannot convert object array to tensor directly")
                return torch.from_numpy(x)
            elif isinstance(x, list):
                print(f"[DEBUG] ensure_tensor: list of length {len(x)}")
                if len(x) == 0:
                    print("[DEBUG] Empty list passed to ensure_tensor")
                    raise TypeError("Empty list cannot be converted to tensor")
                # Recursively ensure elements are tensors or numbers
                if all(isinstance(e, (int, float, complex, np.number)) for e in x):
                    return torch.tensor(x)
                elif all(isinstance(e, torch.Tensor) for e in x):
                    print("[DEBUG] ensure_tensor: concatenating list of tensors")
                    return torch.cat(x)
                else:
                    print(f"[DEBUG] List with unsupported element types: {x}")
                    raise TypeError("List contains unsupported element types")
            else:
                print(f"[DEBUG] Unsupported type for ensure_tensor: {type(x)}")
                raise TypeError(f"Unsupported type for ensure_tensor: {type(x)}")

        try:
            # Process input text
            input_tensor = await self.modality_manager.process_text(text)
            input_tensor = ensure_tensor(input_tensor)
            input_tensor = input_tensor.to(self.device)
            print(f"[DEBUG] train_step input_tensor shape: {input_tensor.shape}")
            # Forward pass
            output = await self.transformer(input_tensor)
            output = ensure_tensor(output)
            print(f"[DEBUG] train_step output shape: {output.shape}")
            # Calculate loss using ComplexLoss
            target = input_tensor.clone()
            loss_fn = ComplexLoss(concept_graph=self.transformer.concept_graph)
            loss = loss_fn(output, target)
            print(f"[DEBUG] train_step raw loss: {loss}")
            # Sanitize loss tensor
            loss = self.ops.sanitize_tensor(loss, "train_step loss")
            print(f"[DEBUG] train_step sanitized loss: {loss}")
            # Add embedding loss if using Word2Vec reference
            print(f"[DEBUG] modality_manager.modalities keys: {list(self.modality_manager.modalities.keys())}")
            text_tokenizer = self.modality_manager.modalities["text_tokenizer"] if "text_tokenizer" in self.modality_manager.modalities else None
            if text_tokenizer is not None and hasattr(text_tokenizer, "compute_embedding_loss"):
                embedding_loss = text_tokenizer.compute_embedding_loss(text_tokenizer.embeddings)
                print(f"[DEBUG] train_step embedding_loss: {embedding_loss}")
                # Ensure embedding_loss is a float or tensor
                if isinstance(embedding_loss, np.ndarray):
                    print(f"[DEBUG] Converting embedding_loss from np.ndarray to float: {embedding_loss}")
                    embedding_loss = float(embedding_loss)
                elif hasattr(embedding_loss, 'item'):
                    embedding_loss = embedding_loss.item()
                loss = loss + 0.1 * embedding_loss
                loss = self.ops.sanitize_tensor(loss, "train_step total loss after embedding")
                print(f"[DEBUG] train_step total loss after embedding: {loss}")
            # Ensure loss is a torch tensor with grad before backward
            if not isinstance(loss, torch.Tensor):
                print(f"[DEBUG] Converting loss to torch.Tensor for backward: {loss}")
                loss = torch.tensor(loss, dtype=torch.complex64, requires_grad=True, device=self.device)
            elif not loss.requires_grad:
                loss.requires_grad_()
            # Ensure loss is real-valued for backward
            if torch.is_complex(loss):
                print(f"[DEBUG] Loss is complex, taking real part for backward: {loss}")
                loss = loss.real
            # Backward pass
            loss.backward()
            print(f"[DEBUG] train_step loss after backward: {loss}")
            # Update weights if optimizer exists
            if hasattr(self, 'optimizer'):
                self.optimizer.step()
                self.optimizer.zero_grad()
                # Update embeddings based on Word2Vec reference
                if text_tokenizer is not None and hasattr(text_tokenizer, "update_embeddings"):
                    text_tokenizer.update_embeddings()
            # Final sanitization before returning
            # Only sanitize if loss is inf or NaN
            if (isinstance(loss, torch.Tensor) and (torch.isnan(loss) or torch.isinf(loss))) or \
               (not isinstance(loss, torch.Tensor) and (np.isnan(loss) or np.isinf(loss))):
                print(f"[DEBUG] Loss is inf or NaN, sanitizing: {loss}")
                loss = self.ops.sanitize_tensor(loss, "train_step final loss")

            print(f"[DEBUG] train_step final sanitized loss: {loss}")

            # Ensure returned loss is a real float
            if not isinstance(loss, torch.Tensor):
                # If it's a numpy array or python float, convert to tensor first
                loss = torch.tensor(loss, dtype=torch.float32, device=self.device)
            if torch.is_complex(loss):
                loss = loss.real
            return float(loss.item())
        except TypeError as e:
            if "Concatenation operation is not implemented for NumPy arrays" in str(e):
                print(f"[DEBUG] NumPy concatenation error in train_step: {e}")
                traceback.print_exc()
                self.logger.error(f"NumPy concatenation error in train_step: {str(e)}")
                return float('inf')
            else:
                print(f"[DEBUG] Exception in train_step: {e}")
                traceback.print_exc()
                self.logger.error(f"Error in train_step: {str(e)}")
                return float('inf')
        except Exception as e:
            print(f"[DEBUG] Exception in train_step: {e}")
            traceback.print_exc()
            self.logger.error(f"Error in train_step: {str(e)}")
            return float('inf')  # Return infinity for failed steps
