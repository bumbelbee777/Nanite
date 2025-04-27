import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import fft

from .complex.linear import ComplexLinear

class InfiniToeplitz(nn.Module):
    def __init__(self, embed_dim, num_heads, factorized=False, chunk_size=512, mem_gamma=0.9):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.chunk_size = chunk_size
        self.mem_gamma = mem_gamma
        
        # Complex projections with proper initialization
        self.q_proj = ComplexLinear(embed_dim, embed_dim, factorized=factorized, kronecker_rank=16 if factorized else None)
        self.k_proj = ComplexLinear(embed_dim, embed_dim, factorized=factorized, kronecker_rank=16 if factorized else None)
        self.v_proj = ComplexLinear(embed_dim, embed_dim, factorized=factorized, kronecker_rank=16 if factorized else None)
        
        # Memory parameters with Xavier initialization
        self.mem_k = nn.Parameter(torch.empty(1, num_heads, self.head_dim, 2))
        self.mem_v = nn.Parameter(torch.empty(1, num_heads, self.head_dim, 2))
        nn.init.xavier_uniform_(self.mem_k)
        nn.init.xavier_uniform_(self.mem_v)
        
        # Toeplitz optimization parameters
        self.register_buffer('toeplitz_mask', self._create_toeplitz_mask(chunk_size))
        
    def _create_toeplitz_mask(self, size):
        """Create lower-triangular Toeplitz mask for causal attention"""
        mask = torch.tril(torch.ones(size, size))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def _toeplitz_fft_matmul(self, x, W):
        """Improved Toeplitz matrix multiplication using FFT convolution"""
        # x: [B, H, L, D, 2], W: [H, L, D, 2]
        batch_size, num_heads, seq_len, head_dim, _ = x.shape
        
        # Pad sequences to next power of 2 for efficient FFT
        pad_len = 2 ** math.ceil(math.log2(seq_len * 2 - 1))
        x_pad = F.pad(x, (0, 0, 0, 0, 0, pad_len - seq_len))
        W_pad = F.pad(W, (0, 0, 0, 0, 0, pad_len - seq_len))
        
        # Convert to complex tensors
        x_complex = torch.complex(x_pad[..., 0], x_pad[..., 1])
        W_complex = torch.complex(W_pad[..., 0], W_pad[..., 1])
        
        # FFT-based convolution
        x_f = fft.fft(x_complex, dim=-2)
        W_f = fft.fft(W_complex, dim=-2)
        
        # Complex multiplication in frequency domain
        out_f = x_f * W_f
        
        # Inverse FFT and extract original sequence length
        out_complex = fft.ifft(out_f, dim=-2)[..., :seq_len]
        
        # Convert back to real tensor
        return torch.stack([out_complex.real, out_complex.imag], dim=-1)
    
    def _update_memory(self, k, v):
        """Update compressed memory using exponential moving average with stability check"""
        with torch.no_grad():
            # Compute new memory values with stability check
            new_k = self.mem_gamma * self.mem_k + (1 - self.mem_gamma) * k.mean(0, keepdim=True)
            new_v = self.mem_gamma * self.mem_v + (1 - self.mem_gamma) * v.mean(0, keepdim=True)
            
            # Apply normalization to prevent exploding values
            k_norm = torch.norm(new_k)
            v_norm = torch.norm(new_v)
            if k_norm > 1.0:
                new_k = new_k / k_norm
            if v_norm > 1.0:
                new_v = new_v / v_norm
                
            self.mem_k.copy_(new_k)
            self.mem_v.copy_(new_v)
    
    def forward(self, x, padding_mask=None):
        """
        x: [B, L, D, 2] complex input tensor
        Returns: [B, L, D, 2] complex output tensor
        """
        batch_size, seq_len, _, _ = x.shape
        
        # Project to query/key/value with proper reshaping
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to attention format: [B, H, L, D/H, 2]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim, 2).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim, 2).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim, 2).transpose(1, 2)
        
        # Handle chunked processing for long sequences
        if seq_len > self.chunk_size:
            outputs = []
            chunk_size = self.chunk_size
            
            # Process in overlapping chunks for better context
            for i in range(0, seq_len, chunk_size):
                # Define chunk with overlap
                start_idx = max(0, i - chunk_size//4)
                end_idx = min(seq_len, i + chunk_size + chunk_size//4)
                chunk_slice = slice(start_idx, end_idx)
                
                # Get chunk tensors
                q_chunk = q[:, :, chunk_slice]
                k_chunk = k[:, :, chunk_slice]
                v_chunk = v[:, :, chunk_slice]
                
                # Include memory in attention computation
                mem_k = self.mem_k.expand(batch_size, -1, -1, -1)
                mem_v = self.mem_v.expand(batch_size, -1, -1, -1)
                
                # Compute attention scores with memory
                chunk_size = end_idx - start_idx
                scores = self._toeplitz_fft_matmul(q_chunk, k_chunk)
                mask = self.toeplitz_mask[:, :, :chunk_size, :chunk_size]
                if padding_mask is not None:
                    mask = mask * padding_mask[:, None, chunk_slice, chunk_slice]
                scores = scores * mask
                
                # Add memory scores
                mem_scores = torch.einsum('bhld,bhd->bhl', 
                    torch.complex(q_chunk[..., 0], q_chunk[..., 1]),
                    torch.complex(mem_k[..., 0], mem_k[..., 1])
                ).unsqueeze(-1)
                mem_scores = torch.stack([mem_scores.real, mem_scores.imag], dim=-1)
                
                # Combine attention scores
                scores = torch.cat([mem_scores, scores], dim=-1)
                
                # Apply softmax to magnitudes
                attn_weights = torch.softmax(scores.norm(dim=-1), dim=-1)
                
                # Weighted sum with memory
                mem_weight = attn_weights[..., 0:1]
                context = mem_weight.unsqueeze(-1) * mem_v
                context = context + torch.einsum('bhlt,bhtd->bhld', 
                    attn_weights[..., 1:],
                    torch.complex(v_chunk[..., 0], v_chunk[..., 1]).real
                ).unsqueeze(-1)
                
                # Update memory with current chunk
                self._update_memory(k_chunk, v_chunk)
                
                # Extract central part if using overlap
                if i > 0:
                    context = context[:, :, chunk_size//4:]
                if i + chunk_size < seq_len:
                    context = context[:, :, :chunk_size]
                    
                outputs.append(context)
            
            # Concatenate chunks
            out = torch.cat(outputs, dim=2)
            
        else:
            # Full sequence processing for short sequences
            scores = self._toeplitz_fft_matmul(q, k)
            if padding_mask is not None:
                scores = scores * padding_mask[:, None, :, :].unsqueeze(-1)
            scores = scores * self.toeplitz_mask[:, :, :seq_len, :seq_len].unsqueeze(-1)
            
            # Apply complex-aware softmax
            attn_weights = torch.softmax(scores.norm(dim=-1), dim=-1)
            
            # Weighted sum
            out = torch.einsum('bhlt,bhtd->bhld',
                attn_weights,
                torch.complex(v[..., 0], v[..., 1]).real
            ).unsqueeze(-1)
        
        # Reshape output
        out = out.transpose(1, 2).contiguous()
        return out.view(batch_size, seq_len, self.embed_dim, 2)