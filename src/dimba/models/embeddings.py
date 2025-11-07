"""Embedding layers for DIMBA model."""

import math
import torch
import torch.nn as nn
from typing import Optional


class TokenEmbedding(nn.Module):
    """Token embedding layer with optional weight tying.

    Args:
        vocab_size: Size of vocabulary
        embed_dim: Embedding dimension
        padding_idx: Optional padding index
    """

    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, std=0.02)
        if padding_idx is not None:
            nn.init.constant_(self.embedding.weight[padding_idx], 0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get embeddings for token IDs.

        Args:
            input_ids: [batch_size, seq_len]

        Returns:
            embeddings: [batch_size, seq_len, embed_dim]
        """
        return self.embedding(input_ids)

    def get_weight(self) -> torch.Tensor:
        """Get embedding matrix for weight tying."""
        return self.embedding.weight


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embeddings with MLP projection.

    Converts discrete timesteps to continuous embeddings to inform the denoiser
    about the current noise level.

    Args:
        time_embed_dim: Output dimension of timestep embedding
        out_dim: Output dimension after MLP projection
    """

    def __init__(self, time_embed_dim: int = 128, out_dim: int = 512):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        # Sinusoidal position encoding
        self.position_encoding = self._create_position_encoding()

        # MLP to project to desired dimension
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def _create_position_encoding(self) -> torch.Tensor:
        """Create sinusoidal position encoding matrix.

        Uses standard transformer-style positional encoding.
        """
        dim = self.time_embed_dim
        position = torch.arange(10000, dtype=torch.float32)  # Support up to 10k timesteps

        # Dimension indices for even/odd positions
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * -(math.log(10000.0) / dim))

        pe = torch.zeros(10000, dim)
        pe[:, 0::2] = torch.sin(position.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(position.unsqueeze(1) * div_term)

        return pe

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Get timestep embeddings.

        Args:
            t: Timestep indices [batch_size], values in [0, num_steps-1]

        Returns:
            embeddings: [batch_size, out_dim]
        """
        # Get sinusoidal encodings
        pe = self.position_encoding[t].to(t.device)  # [batch_size, time_embed_dim]

        # Project through MLP
        embeddings = self.mlp(pe)  # [batch_size, out_dim]

        return embeddings


class PromptEncoder(nn.Module):
    """Lightweight MLP-based prompt encoder.

    Encodes prompt token embeddings to produce conditioning information
    for the denoiser.

    Args:
        input_dim: Input dimension (token embedding dimension)
        hidden_dim: Hidden dimension of MLP
        output_dim: Output conditioning dimension
        num_layers: Number of MLP layers (default: 2)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        for i in range(num_layers):
            in_features = input_dim if i == 0 else hidden_dim
            out_features = output_dim if i == num_layers - 1 else hidden_dim

            layers.append(nn.Linear(in_features, out_features))
            if i < num_layers - 1:  # No activation after last layer
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode prompt embeddings to conditioning vector.

        Args:
            x: Prompt token embeddings [batch_size, seq_len, input_dim]

        Returns:
            conditioning: [batch_size, seq_len, output_dim]
        """
        return self.mlp(x)


class FiLMConditioning(nn.Module):
    """Feature-wise Linear Modulation (FiLM) for conditioning.

    Learns modulation parameters (γ, β) from conditioning information
    to apply to noisy embeddings: γ(c) * x + β(c)

    Args:
        cond_dim: Dimension of conditioning vector
        target_dim: Dimension of embeddings to be modulated
    """

    def __init__(self, cond_dim: int, target_dim: int):
        super().__init__()
        self.cond_dim = cond_dim
        self.target_dim = target_dim

        # Separate linear layers for gamma and beta
        self.gamma_proj = nn.Linear(cond_dim, target_dim)
        self.beta_proj = nn.Linear(cond_dim, target_dim)

        # Initialize to identity transformation: γ=1, β=0
        nn.init.ones_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning.

        Args:
            x: Input embeddings [batch_size, seq_len, target_dim]
            cond: Conditioning vector [batch_size, seq_len, cond_dim]

        Returns:
            modulated: [batch_size, seq_len, target_dim]
        """
        gamma = self.gamma_proj(cond)  # [batch_size, seq_len, target_dim]
        beta = self.beta_proj(cond)    # [batch_size, seq_len, target_dim]

        return gamma * x + beta


class AdditiveConditioning(nn.Module):
    """Simple additive conditioning.

    Concatenates or adds conditioning information to embeddings.

    Args:
        cond_dim: Dimension of conditioning vector
        target_dim: Dimension of embeddings
    """

    def __init__(self, cond_dim: int, target_dim: int):
        super().__init__()
        # Projection to match dimensions if needed
        self.proj = nn.Linear(cond_dim, target_dim) if cond_dim != target_dim else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Add conditioning to embeddings.

        Args:
            x: Input embeddings [batch_size, seq_len, target_dim]
            cond: Conditioning vector [batch_size, seq_len, cond_dim]

        Returns:
            conditioned: [batch_size, seq_len, target_dim]
        """
        cond_proj = self.proj(cond)
        return x + cond_proj
