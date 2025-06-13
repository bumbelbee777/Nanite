import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import asyncio
from dataclasses import dataclass
import numpy as np
import logging
import os
from datetime import datetime


@dataclass
class CandidateResponse:
    """Represents a generated response candidate with its score."""

    tokens: torch.Tensor
    score: float
    coherence: float
    accuracy: float
    relevancy: float
    factuality: float


class TokenGenerationUnit(nn.Module):
    """Asynchronous token generation unit for multi-token prediction."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_tokens: int = 5,
        temperature: float = 0.7,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Set up debug logger
        self.logger = logging.getLogger("tgu")
        self.logger.setLevel(logging.DEBUG)

        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # Create a new log file for each run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join("logs", f"tgu_{timestamp}.log")

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Cross-attention for TGU communication
        self.cross_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Token prediction head
        self.token_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        # Latent state management
        self.latent_state = None
        self.latent_lock = asyncio.Lock()

    def _log_generation_step(self, step: int, details: Dict):
        """Log token generation step details."""
        self.logger.debug(f"Generation step {step}: {details}")

    async def forward(
        self,
        hidden_states: torch.Tensor,
        other_tgu_states: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Generate next token prediction asynchronously.

        Args:
            hidden_states: Hidden states from transformer trunk [batch_size, seq_len, d_model]
            other_tgu_states: List of hidden states from other TGUs for cross-attention

        Returns:
            Predicted next token embedding [batch_size, d_model]
        """
        async with self.latent_lock:
            # Update latent state
            if self.latent_state is None:
                self.latent_state = hidden_states[:, -1:]  # Use last token
            else:
                # Combine with new hidden states
                self.latent_state = torch.cat(
                    [self.latent_state, hidden_states[:, -1:]], dim=1
                )

            # Cross-attention with other TGUs if available
            if other_tgu_states:
                other_states = torch.stack(other_tgu_states, dim=1)
                attn_output, _ = self.cross_attention(
                    self.latent_state, other_states, other_states
                )
                self.latent_state = attn_output

            # Generate token prediction
            token_pred = self.token_head(self.latent_state[:, -1])

            # Apply temperature scaling
            token_pred = token_pred / self.temperature

            # Log generation step
            self._log_generation_step(
                -1,
                {
                    "output_shape": token_pred.shape,
                    "output_mean": token_pred.float().mean().item(),
                    "output_std": token_pred.float().std().item(),
                },
            )

            return token_pred

    def reset(self):
        """Reset the latent state."""
        self.latent_state = None


class ResponseGenerator(nn.Module):
    """Manages multiple TGUs and response synthesis."""

    def __init__(
        self,
        d_model: int,
        n_tgus: int = 3,
        max_candidates: int = 5,
        min_score: float = 15.0,
        max_retries: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_tgus = n_tgus
        self.max_candidates = max_candidates
        self.min_score = min_score
        self.max_retries = max_retries

        # Set up debug logger
        self.logger = logging.getLogger("response_generator")
        self.logger.setLevel(logging.DEBUG)

        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # Create a new log file for each run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join("logs", f"response_generator_{timestamp}.log")

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Initialize TGUs
        self.tgus = nn.ModuleList([TokenGenerationUnit(d_model) for _ in range(n_tgus)])

        # Scoring heads
        self.coherence_head = nn.Linear(d_model, 1)
        self.accuracy_head = nn.Linear(d_model, 1)
        self.relevancy_head = nn.Linear(d_model, 1)
        self.factuality_head = nn.Linear(d_model, 1)

        # Response synthesis
        self.synthesis_head = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=2,
        )

    def _log_candidate_generation(self, candidate_id: int, details: Dict):
        """Log candidate generation details."""
        self.logger.debug(f"Candidate {candidate_id}: {details}")

    async def generate_candidates(
        self, hidden_states: torch.Tensor, num_tokens: int
    ) -> List[CandidateResponse]:
        """Generate multiple response candidates using TGUs.

        Args:
            hidden_states: Hidden states from transformer trunk
            num_tokens: Number of tokens to generate

        Returns:
            List of candidate responses with scores
        """
        candidates = []

        # Log generation start
        self._log_candidate_generation(
            0, {"input_shape": hidden_states.shape, "num_tokens": num_tokens}
        )

        for _ in range(self.max_candidates):
            # Reset all TGUs
            for tgu in self.tgus:
                tgu.reset()

            # Generate response using multiple TGUs
            tokens = []
            for _ in range(num_tokens):
                # Get predictions from all TGUs
                tgu_states = []
                for tgu in self.tgus:
                    pred = await tgu(hidden_states, tgu_states)
                    tgu_states.append(pred)

                # Combine predictions (simple average for now)
                next_token = torch.stack(tgu_states).mean(dim=0)
                tokens.append(next_token)

                # Update hidden states for next iteration
                hidden_states = torch.cat(
                    [hidden_states, next_token.unsqueeze(1)], dim=1
                )

            # Score the candidate
            candidate = self._score_candidate(torch.stack(tokens, dim=1))
            candidates.append(candidate)

            # Log candidate details
            self._log_candidate_generation(
                _ + 1,
                {
                    "tgu_id": _,
                    "score": candidate.score,
                    "coherence": candidate.coherence,
                    "accuracy": candidate.accuracy,
                    "relevancy": candidate.relevancy,
                    "factuality": candidate.factuality,
                },
            )

        # Sort candidates by score
        candidates.sort(key=lambda x: x.score, reverse=True)

        # Log final candidates
        self._log_candidate_generation(
            -1,
            {
                "num_candidates": len(candidates),
                "best_score": candidates[0].score if candidates else None,
                "worst_score": candidates[-1].score if candidates else None,
            },
        )

        return candidates[: self.max_candidates]

    def _score_candidate(self, tokens: torch.Tensor) -> CandidateResponse:
        """Score a candidate response with detailed logging."""
        # Compute various metrics
        coherence = torch.mean(torch.abs(tokens[:, 1:] - tokens[:, :-1])).item()
        accuracy = torch.mean(torch.abs(tokens)).item()
        relevancy = torch.std(tokens).item()
        factuality = torch.mean(torch.abs(tokens - tokens.mean())).item()

        # Compute overall score
        score = (
            (1.0 - coherence) * 0.3
            + accuracy * 0.2
            + relevancy * 0.2
            + factuality * 0.3
        ) * 100.0

        # Log scoring details
        self.logger.debug(
            f"Score candidate: coherence={coherence:.4f}, "
            f"accuracy={accuracy:.4f}, relevancy={relevancy:.4f}, "
            f"factuality={factuality:.4f}, score={score:.4f}"
        )

        return CandidateResponse(
            tokens=tokens,
            score=score,
            coherence=coherence,
            accuracy=accuracy,
            relevancy=relevancy,
            factuality=factuality,
        )

    async def synthesize_response(
        self, candidates: List[CandidateResponse]
    ) -> Optional[torch.Tensor]:
        """Synthesize final response from candidates."""
        if not candidates:
            self.logger.warning("No candidates to synthesize from")
            return None

        # Sort by score
        candidates.sort(key=lambda x: x.score, reverse=True)
        best = candidates[0]

        # Log synthesis details
        self.logger.debug(
            f"Synthesized response: score={best.score:.4f}, "
            f"coherence={best.coherence:.4f}, accuracy={best.accuracy:.4f}, "
            f"relevancy={best.relevancy:.4f}, factuality={best.factuality:.4f}"
        )

        return best.tokens
