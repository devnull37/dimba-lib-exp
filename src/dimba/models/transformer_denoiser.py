"""Transformer-based denoiser for DIMBA (CPU-friendly alternative to Mamba-2).

This provides a simple transformer denoiser that works on CPU and doesn't require
Mamba-SSM compilation. It's suitable for testing and CPU-only development.
"""

import torch
import torch.nn as nn
from typing import Literal, Optional

from .embeddings import FiLMConditioning, AdditiveConditioning


class TransformerBlock(nn.Module):
    """Single transformer block with self-attention and feedforward.

    Simplified transformer block without positional encoding (handled separately).

    Args:
        d_model: Hidden dimension
        num_heads: Number of attention heads
        d_ff: Feedforward dimension (usually 4 * d_model)
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)

        # Feedforward with pre-norm
        x_norm = self.norm2(x)
        ff_out = self.ff(x_norm)
        x = x + self.dropout(ff_out)

        return x


class TransformerDenoiser(nn.Module):
    """Transformer-based denoiser for DIMBA (CPU-friendly alternative).

    Uses transformer blocks instead of Mamba-2 for denoising. Works on CPU
    and doesn't require CUDA compilation.

    Args:
        d_model: Model hidden dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        d_ff: Feedforward dimension
        conditioning_type: Type of conditioning ('film' or 'additive')
        cond_dim: Dimension of conditioning vectors
        time_embed_dim: Dimension of timestep embeddings
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        conditioning_type: Literal["film", "additive"] = "film",
        cond_dim: int = 512,
        time_embed_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.conditioning_type = conditioning_type

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Conditioning layers for each block
        if conditioning_type == "film":
            self.conditioning = nn.ModuleList([
                FiLMConditioning(cond_dim, d_model)
                for _ in range(num_layers)
            ])
        elif conditioning_type == "additive":
            self.conditioning = nn.ModuleList([
                AdditiveConditioning(cond_dim, d_model)
                for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"Unknown conditioning type: {conditioning_type}")

        # Timestep embedding projection to conditioning dimension
        self.time_proj = nn.Linear(time_embed_dim, cond_dim)

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        timestep_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through denoiser.

        Args:
            x: Noisy embeddings [batch_size, seq_len, d_model]
            cond: Conditioning vectors from prompt [batch_size, seq_len, cond_dim]
            timestep_emb: Timestep embeddings [batch_size, time_embed_dim]

        Returns:
            output: Denoised embeddings [batch_size, seq_len, d_model]
        """
        # Project timestep embedding to conditioning dimension
        time_cond = self.time_proj(timestep_emb)  # [batch_size, cond_dim]
        time_cond = time_cond.unsqueeze(1)  # [batch_size, 1, cond_dim]
        time_cond = time_cond.expand(-1, cond.size(1), -1)  # [batch_size, seq_len, cond_dim]

        # Combine temporal and prompt conditioning
        combined_cond = cond + time_cond  # [batch_size, seq_len, cond_dim]

        # Pass through transformer blocks with conditioning
        output = x
        for block, cond_layer in zip(self.blocks, self.conditioning):
            # Apply conditioning
            conditioned = cond_layer(output, combined_cond)

            # Pass through transformer block
            output = block(conditioned)

            # Optional dropout
            if self.dropout is not None:
                output = self.dropout(output)

        return output
