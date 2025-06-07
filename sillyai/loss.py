import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ComplexLoss(nn.Module):
    """Custom loss function that handles complex numbers and incorporates concept graph regularization."""
    def __init__(self, concept_graph=None, alpha=0.1, beta=0.01):
        super().__init__()
        self.concept_graph = concept_graph
        self.alpha = alpha  # Weight for concept graph regularization
        self.beta = beta   # Weight for complex number regularization
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between complex predictions and targets.
        
        Args:
            pred: Complex tensor of shape [batch_size, seq_len, dim]
            target: Complex tensor of shape [batch_size, seq_len, dim]
            
        Returns:
            Total loss combining MSE, concept graph regularization, and complex number regularization
        """
        # Ensure inputs are complex
        if not pred.is_complex():
            pred = torch.complex(pred, torch.zeros_like(pred))
        if not target.is_complex():
            target = torch.complex(target, torch.zeros_like(target))
            
        # Compute MSE loss for real and imaginary parts separately
        real_loss = F.mse_loss(pred.real, target.real)
        imag_loss = F.mse_loss(pred.imag, target.imag)
        mse_loss = real_loss + imag_loss
        
        # Add complex number regularization to encourage meaningful phase
        phase_reg = torch.mean(torch.abs(torch.angle(pred)))  # Penalize large phase angles
        
        # Add concept graph regularization if available
        concept_loss = torch.tensor(0.0, device=pred.device)
        if self.concept_graph is not None and self.concept_graph.concept_embeddings:
            # Get concept embeddings from the graph
            concepts = list(self.concept_graph.concept_embeddings.values())
            if concepts:
                # Stack embeddings into a tensor
                concept_tensor = torch.stack(concepts)
                
                # Compute cosine similarity between predictions and concept embeddings
                pred_norm = F.normalize(pred.reshape(-1, pred.shape[-1]), dim=-1)
                concept_norm = F.normalize(concept_tensor, dim=-1)
                similarity = torch.matmul(pred_norm, concept_norm.t())
                
                # Encourage predictions to align with relevant concepts
                concept_loss = -torch.mean(torch.max(similarity, dim=-1)[0])
        
        # Combine losses
        total_loss = mse_loss + self.alpha * concept_loss + self.beta * phase_reg
        
        return total_loss 