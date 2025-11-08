"""Mamba-2 based denoiser for DIMBA."""

import torch
import torch.nn as nn
from typing import Literal, Optional

try:
    from mamba_ssm import Mamba
    HAS_MAMBA_SSM = True
except ImportError:
    HAS_MAMBA_SSM = False

from .embeddings import FiLMConditioning, AdditiveConditioning


class Mamba2Block(nn.Module):
    """Single Mamba-2 block with normalization and conditioning.

    Wraps a Mamba SSM layer with layer normalization and optional conditioning.

    Args:
        d_model: Hidden dimension
        d_state: State size for SSM
        d_conv: Convolution kernel size
        expand: Expansion factor for inner dimension
        dt_rank: Rank for delta projection
        dt_min: Minimum delta value
        dt_max: Maximum delta value
        dt_init: Delta initialization strategy
        dt_scale: Delta scale factor
        bias: Whether to use bias in linear layers
        conv_bias: Whether to use bias in convolution
    """

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = True,
        conv_bias: bool = True,
        use_simple_mamba: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)

        # Use simple Mamba if requested or if mamba_ssm not available
        if use_simple_mamba or not HAS_MAMBA_SSM:
            from .simple_mamba import SimpleMamba2
            self.mamba = SimpleMamba2(
                d_model=d_model,
                d_state=d_state,
                d_expand=expand,
            )
        else:
            # Use optimized mamba-ssm
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init=dt_init,
                dt_scale=dt_scale,
                bias=bias,
                conv_bias=conv_bias,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Pre-norm + residual
        return x + self.mamba(self.norm(x))


class Mamba2Denoiser(nn.Module):
    """Mamba-2 based denoiser for DIMBA diffusion model.

    Stacks multiple Mamba-2 blocks with conditioning support.

    Args:
        d_model: Model hidden dimension
        num_layers: Number of Mamba-2 blocks
        d_state: SSM state size
        d_conv: Convolution kernel size
        expand: Expansion factor for inner dimension
        conditioning_type: Type of conditioning ('film' or 'additive')
        cond_dim: Dimension of conditioning vectors
        time_embed_dim: Dimension of timestep embeddings
        dropout: Dropout rate
    """

    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        conditioning_type: Literal["film", "additive"] = "film",
        cond_dim: int = 512,
        time_embed_dim: int = 512,
        dropout: float = 0.1,
        use_simple_mamba: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.conditioning_type = conditioning_type

        # Mamba blocks
        self.blocks = nn.ModuleList([
            Mamba2Block(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                use_simple_mamba=use_simple_mamba,
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
        # Expand to match sequence length
        time_cond = self.time_proj(timestep_emb)  # [batch_size, cond_dim]
        time_cond = time_cond.unsqueeze(1)  # [batch_size, 1, cond_dim]
        time_cond = time_cond.expand(-1, cond.size(1), -1)  # [batch_size, seq_len, cond_dim]

        # Combine temporal and prompt conditioning
        combined_cond = cond + time_cond  # [batch_size, seq_len, cond_dim]

        # Pass through Mamba blocks with conditioning
        output = x
        for block, cond_layer in zip(self.blocks, self.conditioning):
            # Apply conditioning
            conditioned = cond_layer(output, combined_cond)

            # Pass through Mamba block
            output = block(conditioned)

            # Optional dropout
            if self.dropout is not None:
                output = self.dropout(output)

        return output


class DenoisingHead(nn.Module):
    """Output head for converting denoised embeddings back to token logits.

    Args:
        d_model: Model hidden dimension
        vocab_size: Vocabulary size
        use_weight_tying: Whether to tie weights with embedding matrix
        embedding_weight: Embedding weight for weight tying (optional)
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        use_weight_tying: bool = False,
        embedding_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_weight_tying = use_weight_tying

        if use_weight_tying and embedding_weight is not None:
            # Weight tying: share with embedding matrix
            self.projection = nn.Identity()
            self.register_buffer("embedding_weight", embedding_weight, persistent=False)
        else:
            # Independent projection layer
            self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, embedding_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Project denoised embeddings to token logits.

        Args:
            x: Denoised embeddings [batch_size, seq_len, d_model]
            embedding_weight: Optional embedding weight for weight tying

        Returns:
            logits: Token logits [batch_size, seq_len, vocab_size]
        """
        if self.use_weight_tying:
            # Use tied embedding weight
            if embedding_weight is None:
                embedding_weight = self.embedding_weight
            # x @ W^T where W is embedding matrix transposed
            logits = torch.matmul(x, embedding_weight.t())
        else:
            # Use independent projection
            logits = self.projection(x)

        return logits
