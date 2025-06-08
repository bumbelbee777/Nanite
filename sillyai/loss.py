import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ComplexLoss(nn.Module):
    """Custom loss function for complex numbers that handles both real and imaginary parts."""
    
    def __init__(self, concept_graph=None, alpha=0.1, beta=0.01):
        super().__init__()
        self.concept_graph = concept_graph
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss for complex predictions and targets.
        
        Args:
            pred: Complex tensor of shape [batch_size, seq_len, dim]
            target: Real or complex tensor of shape [batch_size, seq_len, dim]
            
        Returns:
            Scalar loss value
        """
        # Convert target to complex if it's not already
        if not torch.is_complex(target):
            target = torch.complex(target, torch.zeros_like(target))
            
        # Split into real and imaginary parts
        pred_real, pred_imag = pred.real, pred.imag
        target_real, target_imag = target.real, target.imag
        
        # Compute MSE for real and imaginary parts separately
        real_loss = torch.mean((pred_real - target_real) ** 2)
        imag_loss = torch.mean((pred_imag - target_imag) ** 2)
        
        # Combine losses
        mse_loss = real_loss + imag_loss
        
        # Add concept graph regularization if available
        if self.concept_graph is not None:
            concept_loss = self._compute_concept_loss(pred)
            mse_loss = mse_loss + self.alpha * concept_loss
            
        # Add phase regularization
        phase_loss = self._compute_phase_loss(pred, target)
        mse_loss = mse_loss + self.beta * phase_loss
        
        return mse_loss
        
    def _compute_concept_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """Compute regularization loss based on concept graph."""
        if not self.concept_graph:
            return torch.tensor(0.0, device=pred.device)
            
        # Extract relevant concepts from prediction
        concepts = self._extract_concepts(pred)
        
        # Compute energy-based regularization
        energy_loss = 0.0
        for concept in concepts:
            if concept in self.concept_graph:
                energy = self.concept_graph[concept]['energy']
                energy_loss += torch.abs(energy)
                
        return energy_loss
        
    def _compute_phase_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute phase difference loss between prediction and target."""
        pred_phase = torch.atan2(pred.imag, pred.real)
        target_phase = torch.atan2(target.imag, target.real)
        
        # Compute phase difference
        phase_diff = torch.abs(pred_phase - target_phase)
        
        # Normalize to [0, π]
        phase_diff = torch.min(phase_diff, 2 * torch.pi - phase_diff)
        
        return torch.mean(phase_diff)
        
    def _extract_concepts(self, pred: torch.Tensor) -> list:
        """Extract relevant concepts from prediction tensor."""
        # This is a placeholder - implement actual concept extraction logic
        return [] 