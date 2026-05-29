"""Mamba-based denoiser for DIMBA.

Two correctness-relevant changes vs. the original implementation:

1. **Mamba-2 first.** We now prefer the genuine Mamba-2 (SSD) kernel from
   ``mamba_ssm`` (``Mamba2``), falling back to Mamba-1 (``Mamba``) and then to the
   pure-PyTorch :class:`~dimba.models.simple_mamba.SimpleMamba2`. The original
   code imported ``Mamba`` (the Mamba-1 API) while naming everything "Mamba-2".

2. **Bidirectional scans.** Vanilla Mamba is *causal* (position ``t`` only sees
   ``<= t``). For non-autoregressive diffusion denoising every position should see
   the entire (noisy) sequence, so each block optionally runs a forward and a
   backward scan with *separate* SSM parameters and sums them (the Vision-Mamba /
   Vim recipe, arXiv:2401.09417). This is enabled by default.
"""

import warnings
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _checkpoint
from typing import Literal, Optional

# Resolve the best available Mamba implementation once, at import time.
_MAMBA_CLS = None
_MAMBA_KIND = "simple"
HAS_MAMBA_SSM = False
try:  # Mamba-2 (SSD) — the intended backbone.
    from mamba_ssm import Mamba2 as _MAMBA_CLS  # type: ignore

    HAS_MAMBA_SSM = True
    _MAMBA_KIND = "mamba2"
except ImportError:
    try:  # Mamba-1 fallback.
        from mamba_ssm import Mamba as _MAMBA_CLS  # type: ignore

        HAS_MAMBA_SSM = True
        _MAMBA_KIND = "mamba1"
    except ImportError:
        _MAMBA_CLS = None

from .embeddings import FiLMConditioning, AdditiveConditioning

_FALLBACK_WARNED = False


def _make_mixer(
    d_model: int,
    d_state: int,
    d_conv: int,
    expand: int,
    use_simple_mamba: bool,
) -> nn.Module:
    """Construct a single (causal) SSM mixer using the best available backend.

    The returned module maps ``[B, L, d_model] -> [B, L, d_model]`` and contains
    no normalization or residual connection (the enclosing block owns those).
    """
    global _FALLBACK_WARNED
    if use_simple_mamba or not HAS_MAMBA_SSM:
        from .simple_mamba import SimpleMamba2

        return SimpleMamba2(d_model=d_model, d_state=d_state, d_expand=expand)

    # mamba_ssm kernels (CUDA). Mamba2 and Mamba take slightly different kwargs;
    # fall back gracefully rather than crash, and warn once if we can't use them.
    try:
        return _MAMBA_CLS(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    except (TypeError, ValueError, AssertionError, RuntimeError) as exc:  # pragma: no cover - CUDA only
        if not _FALLBACK_WARNED:
            warnings.warn(
                f"Could not construct {_MAMBA_KIND} mixer ({exc}); falling back to "
                f"pure-PyTorch SimpleMamba2.",
                RuntimeWarning,
            )
            _FALLBACK_WARNED = True
        from .simple_mamba import SimpleMamba2

        return SimpleMamba2(d_model=d_model, d_state=d_state, d_expand=expand)


class Mamba2Block(nn.Module):
    """Pre-norm Mamba block with an optional bidirectional scan and residual.

    Args:
        d_model: Hidden dimension.
        d_state: SSM state size.
        d_conv: Short convolution kernel size (Mamba kernels only).
        expand: Inner expansion factor.
        bidirectional: If True (default), run a forward + backward scan with
            separate parameters and sum them.
        use_simple_mamba: Force the pure-PyTorch fallback mixer.
    """

    def __init__(
        self,
        d_model: int = 512,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bidirectional: bool = True,
        use_simple_mamba: bool = False,
        # Accepted for backward compatibility; only used by the Mamba-1 kernel.
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        bias: bool = True,
        conv_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.bidirectional = bidirectional
        self.norm = nn.LayerNorm(d_model)

        self.mamba_fwd = _make_mixer(d_model, d_state, d_conv, expand, use_simple_mamba)
        self.mamba_bwd = (
            _make_mixer(d_model, d_state, d_conv, expand, use_simple_mamba)
            if bidirectional
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm + (bi)directional mix + residual.

        Args:
            x: Input ``[B, L, d_model]``.

        Returns:
            Output ``[B, L, d_model]``.
        """
        h = self.norm(x)
        y = self.mamba_fwd(h)
        if self.bidirectional and self.mamba_bwd is not None:
            h_rev = torch.flip(h, dims=[1])
            y_bwd = self.mamba_bwd(h_rev)
            y = y + torch.flip(y_bwd, dims=[1])
        return x + y


class Mamba2Denoiser(nn.Module):
    """Stack of (bidirectional) Mamba blocks with prompt + timestep conditioning.

    Args:
        d_model: Model hidden dimension (the diffusion latent dim).
        num_layers: Number of blocks.
        d_state: SSM state size.
        d_conv: Conv kernel size.
        expand: Inner expansion factor.
        conditioning_type: 'film' or 'additive'.
        cond_dim: Dimension of the conditioning vectors.
        time_embed_dim: Dimension of the incoming timestep embedding.
        dropout: Dropout rate between blocks.
        bidirectional: Enable bidirectional scans (default True).
        use_simple_mamba: Force the pure-PyTorch fallback mixer.
        use_gradient_checkpointing: Recompute each block in the backward pass
            instead of storing its activations. Trades ~25% extra compute for a
            large drop in peak memory (the SSM scan materializes a
            ``[B, L, d_inner, d_state]`` tensor per layer). Only active in
            training mode; ignored during eval/sampling.
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
        bidirectional: bool = True,
        use_simple_mamba: bool = False,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.conditioning_type = conditioning_type
        self.bidirectional = bidirectional
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.blocks = nn.ModuleList(
            [
                Mamba2Block(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    bidirectional=bidirectional,
                    use_simple_mamba=use_simple_mamba,
                )
                for _ in range(num_layers)
            ]
        )

        if conditioning_type == "film":
            self.conditioning = nn.ModuleList(
                [FiLMConditioning(cond_dim, d_model) for _ in range(num_layers)]
            )
        elif conditioning_type == "additive":
            self.conditioning = nn.ModuleList(
                [AdditiveConditioning(cond_dim, d_model) for _ in range(num_layers)]
            )
        else:
            raise ValueError(f"Unknown conditioning type: {conditioning_type}")

        # Project the timestep embedding into the conditioning dimension.
        self.time_proj = nn.Linear(time_embed_dim, cond_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        timestep_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Denoise ``x`` conditioned on ``cond`` (prompt) and ``timestep_emb``.

        Args:
            x: Noisy latents ``[B, L, d_model]``.
            cond: Conditioning ``[B, L, cond_dim]`` (broadcast over L is fine).
            timestep_emb: Timestep embedding ``[B, time_embed_dim]``.

        Returns:
            Denoised latents ``[B, L, d_model]``.
        """
        # Broadcast the timestep embedding across the sequence and add to cond.
        time_cond = self.time_proj(timestep_emb).unsqueeze(1)  # [B, 1, cond_dim]
        time_cond = time_cond.expand(-1, cond.size(1), -1)
        combined_cond = cond + time_cond

        output = x
        for block, cond_layer in zip(self.blocks, self.conditioning):
            if self.use_gradient_checkpointing and self.training:
                # use_reentrant=False is the DDP-safe variant and correctly
                # restores RNG state for the dropout inside _block_forward.
                output = _checkpoint(
                    self._block_forward,
                    block,
                    cond_layer,
                    output,
                    combined_cond,
                    use_reentrant=False,
                )
            else:
                output = self._block_forward(block, cond_layer, output, combined_cond)
        return output

    def _block_forward(self, block, cond_layer, x, combined_cond):
        """Condition + mix + dropout for a single denoiser block."""
        conditioned = cond_layer(x, combined_cond)
        out = block(conditioned)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class DenoisingHead(nn.Module):
    """Project denoised embeddings to token logits, with optional weight tying.

    Args:
        d_model: Model hidden dimension.
        vocab_size: Vocabulary size.
        use_weight_tying: Tie the projection with the embedding matrix.
        embedding_weight: Embedding weight for tying (optional).
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
            self.projection = nn.Identity()
            self.register_buffer("embedding_weight", embedding_weight, persistent=False)
        else:
            self.projection = nn.Linear(d_model, vocab_size)

    def forward(
        self, x: torch.Tensor, embedding_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Project denoised embeddings ``[B, L, d_model]`` to logits ``[B, L, vocab]``."""
        if self.use_weight_tying:
            if embedding_weight is None:
                embedding_weight = self.embedding_weight
            return torch.matmul(x, embedding_weight.t())
        return self.projection(x)
