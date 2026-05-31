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

import math
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

from .embeddings import FiLMConditioning, AdditiveConditioning, AdaLNZeroConditioning

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
    if use_simple_mamba:
        from .simple_mamba import SimpleMamba2

        return SimpleMamba2(d_model=d_model, d_state=d_state, d_expand=expand)

    if not HAS_MAMBA_SSM:
        # No CUDA mamba_ssm: use the pure-PyTorch SSD mixer that is weight-compatible
        # with mamba_ssm.Mamba2, so a checkpoint trained with the CUDA kernel loads
        # and runs as-is on CPU/MPS (Apple Silicon, etc.).
        from .torch_mamba2 import TorchMamba2

        return TorchMamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    # mamba_ssm kernels (CUDA). Mamba2 and Mamba take slightly different kwargs;
    # fall back gracefully rather than crash, and warn once if we can't use them.
    try:
        return _MAMBA_CLS(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    except (TypeError, ValueError, AssertionError, RuntimeError) as exc:  # pragma: no cover - CUDA only
        if not _FALLBACK_WARNED:
            warnings.warn(
                f"Could not construct {_MAMBA_KIND} mixer ({exc}); falling back to "
                f"pure-PyTorch TorchMamba2.",
                RuntimeWarning,
            )
            _FALLBACK_WARNED = True
        from .torch_mamba2 import TorchMamba2

        return TorchMamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


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
        bidir_merge: str = "sum",
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
        self.bidir_merge = bidir_merge
        self.norm = nn.LayerNorm(d_model)

        self.mamba_fwd = _make_mixer(d_model, d_state, d_conv, expand, use_simple_mamba)
        self.mamba_bwd = (
            _make_mixer(d_model, d_state, d_conv, expand, use_simple_mamba)
            if bidirectional
            else None
        )
        # Optional learned merge of the two scan directions (default "sum" preserves
        # the original behavior). Init to [I | I] so concat starts == sum, then
        # specializes -- gives the two directions independent routing capacity.
        self.merge = None
        if bidirectional and bidir_merge == "concat":
            self.merge = nn.Linear(2 * d_model, d_model, bias=False)
            with torch.no_grad():
                self.merge.weight.copy_(
                    torch.cat([torch.eye(d_model), torch.eye(d_model)], dim=1)
                )

    def _mix(self, h: torch.Tensor) -> torch.Tensor:
        """(Bi)directional SSM mix of an already-normalized input (no residual)."""
        y = self.mamba_fwd(h)
        if self.bidirectional and self.mamba_bwd is not None:
            y_bwd = torch.flip(self.mamba_bwd(torch.flip(h, dims=[1])), dims=[1])
            if self.merge is not None:
                y = self.merge(torch.cat([y, y_bwd], dim=-1))
            else:
                y = y + y_bwd
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm + (bi)directional mix + residual (FiLM / additive path)."""
        return x + self._mix(self.norm(x))

    def forward_adaln(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        """AdaLN-Zero variant: norm -> (1+scale)*h+shift -> mix -> x + gate*y."""
        h = self.norm(x)
        h = h * (1 + scale) + shift
        return x + gate * self._mix(h)


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
        conditioning_type: Literal["film", "additive", "adaln"] = "film",
        cond_dim: int = 512,
        time_embed_dim: int = 512,
        dropout: float = 0.1,
        bidirectional: bool = True,
        use_simple_mamba: bool = False,
        use_gradient_checkpointing: bool = False,
        bidir_merge: str = "sum",
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
                    bidir_merge=bidir_merge,
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
        elif conditioning_type == "adaln":
            self.conditioning = nn.ModuleList(
                [AdaLNZeroConditioning(cond_dim, d_model) for _ in range(num_layers)]
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
        if self.conditioning_type == "adaln":
            scale, shift, gate = cond_layer(combined_cond)
            out = block.forward_adaln(x, scale, shift, gate)
        else:
            out = block(cond_layer(x, combined_cond))
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
        use_norm: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_weight_tying = use_weight_tying

        # Optional pre-head LayerNorm + learnable logit scale. A bare weight-tied head
        # is a dot product with embeddings of std ~0.02, which produces near-uniform
        # logits (poor token discrimination -> garbled word formation in generation).
        # Normalizing the input and learning a temperature restores the logit dynamic
        # range. Off by default to preserve existing-checkpoint behavior.
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(d_model)
            # exp(init) == sqrt(d_model): a sensible starting temperature for a tied head.
            self.logit_scale = nn.Parameter(torch.tensor(0.5 * math.log(d_model)))

        if use_weight_tying and embedding_weight is not None:
            self.projection = nn.Identity()
            self.register_buffer("embedding_weight", embedding_weight, persistent=False)
        else:
            self.projection = nn.Linear(d_model, vocab_size)

    def forward(
        self, x: torch.Tensor, embedding_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Project denoised embeddings ``[B, L, d_model]`` to logits ``[B, L, vocab]``."""
        if self.use_norm:
            x = self.norm(x)
        if self.use_weight_tying:
            if embedding_weight is None:
                embedding_weight = self.embedding_weight
            logits = torch.matmul(x, embedding_weight.t())
        else:
            logits = self.projection(x)
        if self.use_norm:
            logits = logits * self.logit_scale.exp()
        return logits
