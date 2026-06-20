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
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint
from typing import Literal, Optional


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation (Zhang & Sennrich, 2019).

    Faster than LayerNorm: no mean subtraction, no bias.  Used by Llama,
    Qwen, Gemma, and most modern LLMs.  Drop-in replacement for
    ``nn.LayerNorm`` on the last dimension.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to fp32 for variance: bf16 max is ~65504, so x²>256 overflows to inf.
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).sqrt().to(x.dtype)
        return (x / rms) * self.weight

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
    headdim: int = 64,
    force_torch_mixer: bool = False,
) -> nn.Module:
    """Construct a single (causal) SSM mixer using the best available backend.

    The returned module maps ``[B, L, d_model] -> [B, L, d_model]`` and contains
    no normalization or residual connection (the enclosing block owns those).

    Args:
        force_torch_mixer: Select the pure-PyTorch :class:`TorchMamba2` even when
            the CUDA ``mamba_ssm`` kernels are installed. TorchMamba2 is the only
            backend that exposes ``materialize_mixing_matrix`` (needed for MOHAWK
            Stage-1 distillation) and is state_dict-compatible with
            ``mamba_ssm.Mamba2`` (identical 8-key param tree at the default config),
            so weights can be transferred to the CUDA kernel afterwards.
    """
    global _FALLBACK_WARNED
    if use_simple_mamba:
        from .simple_mamba import SimpleMamba2

        return SimpleMamba2(d_model=d_model, d_state=d_state, d_expand=expand)

    if force_torch_mixer:
        # Matrix-capable backend, regardless of whether mamba_ssm is installed.
        from .torch_mamba2 import TorchMamba2

        return TorchMamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)

    if not HAS_MAMBA_SSM:
        # No CUDA mamba_ssm: use the pure-PyTorch SSD mixer that is weight-compatible
        # with mamba_ssm.Mamba2, so a checkpoint trained with the CUDA kernel loads
        # and runs as-is on CPU/MPS (Apple Silicon, etc.).
        from .torch_mamba2 import TorchMamba2

        return TorchMamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)

    # mamba_ssm kernels (CUDA). Mamba2 and Mamba take slightly different kwargs;
    # fall back gracefully rather than crash, and warn once if we can't use them.
    try:
        return _MAMBA_CLS(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)
    except (TypeError, ValueError, AssertionError, RuntimeError) as exc:  # pragma: no cover - CUDA only
        if not _FALLBACK_WARNED:
            warnings.warn(
                f"Could not construct {_MAMBA_KIND} mixer ({exc}); falling back to "
                f"pure-PyTorch TorchMamba2.",
                RuntimeWarning,
            )
            _FALLBACK_WARNED = True
        from .torch_mamba2 import TorchMamba2

        return TorchMamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, headdim=headdim)


class _BlockFFN(nn.Module):
    """Position-wise feed-forward sub-layer for a Mamba block (channel-mixing).

    A Mamba block mixes *tokens* but has no dedicated *channel*-mixing / memory module
    the way a Transformer block's MLP does. This optional sub-layer adds one, in either
    of the two shapes used by real LLMs so a teacher's FFN weights can be inherited
    directly (MOHAWK-style cross-architecture distillation):

      * ``"mlp"``    -> ``ff2(gelu(ff1(x)))``  (GPT-2 / Pythia / BERT; ``hidden = mult*d``).
      * ``"swiglu"`` -> ``down(silu(gate(x)) * up(x))``  (Llama / Qwen / Mistral).

    Submodule names (``ff1``/``ff2``; ``gate_proj``/``up_proj``/``down_proj``) match the
    corresponding HuggingFace modules so weight transfer is a plain ``copy_``.
    """

    def __init__(
        self,
        d_model: int,
        mult: int = 4,
        ffn_type: str = "mlp",
        hidden: Optional[int] = None,
    ):
        super().__init__()
        self.ffn_type = ffn_type
        self.hidden = int(hidden) if hidden is not None else mult * d_model
        if ffn_type == "mlp":
            self.ff1 = nn.Linear(d_model, self.hidden)
            self.ff2 = nn.Linear(self.hidden, d_model)
        elif ffn_type == "swiglu":
            self.gate_proj = nn.Linear(d_model, self.hidden, bias=False)
            self.up_proj = nn.Linear(d_model, self.hidden, bias=False)
            self.down_proj = nn.Linear(self.hidden, d_model, bias=False)
        else:
            raise ValueError(f"Unknown ffn_type: {ffn_type!r} (expected 'mlp' or 'swiglu')")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ffn_type == "swiglu":
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return self.ff2(F.gelu(self.ff1(x)))


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
        headdim: int = 64,
        bidirectional: bool = True,
        use_simple_mamba: bool = False,
        force_torch_mixer: bool = False,
        bidir_merge: str = "sum",
        block_ffn: bool = False,
        ffn_mult: int = 4,
        ffn_type: str = "mlp",
        ffn_hidden: Optional[int] = None,
        dropout: float = 0.0,
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
        self.norm = RMSNorm(d_model)
        # Dropout is applied on each sub-layer branch *before* the residual add,
        # NOT on the residual stream itself, so the AdaLN-Zero identity-at-init
        # guarantee is preserved (gate=0 at init -> branch is 0 -> dropout(0)=0).
        self.dropout: nn.Module = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        self.mamba_fwd = _make_mixer(
            d_model, d_state, d_conv, expand, use_simple_mamba,
            headdim=headdim, force_torch_mixer=force_torch_mixer,
        )
        self.mamba_bwd = (
            _make_mixer(
                d_model, d_state, d_conv, expand, use_simple_mamba,
                headdim=headdim, force_torch_mixer=force_torch_mixer,
            )
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

        # Optional position-wise FFN sub-layer (channel-mixing). Off by default so the
        # mixer-only block and existing checkpoints are byte-for-byte unchanged. When
        # enabled it adds the channel-mixing/memory capacity a mixer lacks and a slot
        # to inherit a Transformer teacher's MLP weights (cross-architecture distill).
        self.norm2 = None
        self.ffn = None
        if block_ffn:
            self.norm2 = RMSNorm(d_model)
            self.ffn = _BlockFFN(d_model, mult=ffn_mult, ffn_type=ffn_type, hidden=ffn_hidden)

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
        """Pre-norm + (bi)directional mix + residual, then optional FFN sub-layer."""
        x = x + self.dropout(self._mix(self.norm(x)))
        if self.ffn is not None:
            x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

    def forward_adaln(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        gate: torch.Tensor,
        scale2: Optional[torch.Tensor] = None,
        shift2: Optional[torch.Tensor] = None,
        gate2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """AdaLN-Zero: norm→modulate→mix→residual, then optionally modulated FFN.

        When ``scale2/shift2/gate2`` are provided (DiT full-block, 6-param path)
        the FFN sub-layer is also adaptively conditioned.  When they are None the
        FFN falls back to a plain pre-norm residual (3-param path).
        """
        h = self.norm(x)
        h = h * (1 + scale) + shift
        x = x + self.dropout(gate * self._mix(h))
        if self.ffn is not None:
            if scale2 is not None:
                h2 = self.norm2(x) * (1 + scale2) + shift2
                x = x + self.dropout(gate2 * self.ffn(h2))
            else:
                x = x + self.dropout(self.ffn(self.norm2(x)))
        return x

    def materialize_matrices(self, h_normed: torch.Tensor):
        """Return ``(M_fwd, M_bwd)`` per-head token-mixing matrices for a mixer input.

        ``h_normed`` is the *already-normalized, conditioned* tensor the block feeds to
        its mixer (see :meth:`Mamba2Denoiser._mixer_input`). ``M_fwd`` ``[B, H, L, L]``
        is causal (lower-triangular); ``M_bwd`` ``[B, H, L, L]`` is the backward scan's
        matrix re-expressed in the original position basis (upper-triangular), or
        ``None`` when the block is unidirectional. These are the MOHAWK Stage-1 student
        matrices, directly comparable to a causal teacher's ``tril(A)`` / ``triu(A)``.
        """
        if not hasattr(self.mamba_fwd, "materialize_mixing_matrix"):
            raise NotImplementedError(
                f"Matrix-orientation distillation requires a mixer exposing "
                f"materialize_mixing_matrix. Got {type(self.mamba_fwd).__name__}. "
                f"This needs the pure-PyTorch TorchMamba2 backend, which is only "
                f"selected when use_simple_mamba=False AND mamba_ssm is absent "
                f"(HAS_MAMBA_SSM={HAS_MAMBA_SSM}). On a CUDA box with mamba_ssm "
                f"installed, build with a force-torch-mixer flag to enable alignment."
            )
        m_fwd = self.mamba_fwd.materialize_mixing_matrix(h_normed)
        m_bwd = None
        if self.bidirectional and self.mamba_bwd is not None:
            m_bwd_flipped = self.mamba_bwd.materialize_mixing_matrix(
                torch.flip(h_normed, dims=[1])
            )
            # The backward mixer ran on the flipped sequence; flip both sequence axes
            # of its matrix back to the original position basis (-> upper-triangular).
            m_bwd = torch.flip(m_bwd_flipped, dims=[2, 3])
        return m_fwd, m_bwd


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
        headdim: int = 64,
        conditioning_type: Literal["film", "additive", "adaln"] = "adaln",
        cond_dim: int = 512,
        time_embed_dim: int = 512,
        dropout: float = 0.1,
        bidirectional: bool = True,
        use_simple_mamba: bool = False,
        force_torch_mixer: bool = False,
        use_gradient_checkpointing: bool = False,
        bidir_merge: str = "sum",
        block_ffn: bool = False,
        ffn_mult: int = 4,
        ffn_type: str = "mlp",
        ffn_hidden: Optional[int] = None,
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
                    headdim=headdim,
                    bidirectional=bidirectional,
                    use_simple_mamba=use_simple_mamba,
                    force_torch_mixer=force_torch_mixer,
                    bidir_merge=bidir_merge,
                    block_ffn=block_ffn,
                    ffn_mult=ffn_mult,
                    ffn_type=ffn_type,
                    ffn_hidden=ffn_hidden,
                    dropout=dropout,
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
                [AdaLNZeroConditioning(cond_dim, d_model, has_ffn=block_ffn)
                 for _ in range(num_layers)]
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
        *,
        return_hidden_states: bool = False,
        return_matrices: bool = False,
    ):
        """Denoise ``x`` conditioned on ``cond`` (prompt) and ``timestep_emb``.

        Args:
            x: Noisy latents ``[B, L, d_model]``.
            cond: Conditioning ``[B, L, cond_dim]`` (broadcast over L is fine).
            timestep_emb: Timestep embedding ``[B, time_embed_dim]``.
            return_hidden_states: Also collect the residual-stream input to each block.
            return_matrices: Also materialize each block's ``(fwd, bwd)`` mixing matrices.

        Returns:
            Denoised latents ``[B, L, d_model]`` by default. When ``return_hidden_states``
            or ``return_matrices`` is set, returns ``(out, info)`` where ``info`` carries
            ``hidden_states`` / ``block_outputs`` (lists of ``[B, L, d_model]``) and
            ``matrices_fwd`` / ``matrices_bwd`` (lists of ``[B, H, L, L]`` or ``None``).
        """
        # Broadcast the timestep embedding across the sequence and add to cond.
        time_cond = self.time_proj(timestep_emb).unsqueeze(1)  # [B, 1, cond_dim]
        time_cond = time_cond.expand(-1, cond.size(1), -1)
        combined_cond = cond + time_cond

        output = x
        if not (return_hidden_states or return_matrices):
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

        # Distillation-alignment collection path: capture per-block intermediates.
        # Gradient checkpointing is bypassed here so the captured tensors are real
        # graph nodes the alignment losses can backprop through.
        #
        # WARNING (return_matrices=True): materialize_mixing_matrix builds a full
        # [B, nheads, L, L] matrix per block WITHOUT chunking, costing
        # O(num_layers * B * nheads * L^2) memory in fp32 plus the autograd graph.
        # This is intentional for distillation at modest L, but will OOM at long
        # sequences where the normal chunked forward() succeeds.  Raise early with a
        # clear message rather than an opaque CUDA OOM.
        if return_matrices and x.shape[1] > 0:
            B, L, _ = x.shape
            # Use the real per-mixer head count when available (d_model//64 under-counts
            # for expand>1, e.g. 9 vs the true 18 at d_model=576), and double it for the
            # bidirectional fwd+bwd materialisation so the budget isn't ~4x optimistic.
            _nheads_est = getattr(self.blocks[0].mamba_fwd, "nheads", max(1, self.d_model // 64))
            if self.bidirectional:
                _nheads_est *= 2
            _budget = B * _nheads_est * L * L * self.num_layers
            # Warn at >256 M elements (~1 GB fp32); hard-fail at >2 B elements (~8 GB).
            if _budget > 2_000_000_000:
                raise RuntimeError(
                    f"return_matrices=True would materialize ~{_budget/1e9:.1f}G fp32 "
                    f"elements across {self.num_layers} layers (B={B}, L={L}). "
                    "Use shorter sequences or disable return_matrices to avoid OOM."
                )
            if _budget > 256_000_000:
                warnings.warn(
                    f"return_matrices=True: estimated {_budget/1e6:.0f}M fp32 elements "
                    f"across {self.num_layers} layers (B={B}, L={L}). "
                    "Peak memory is O(num_layers * B * nheads * L^2); "
                    "consider shorter sequences for Stage-1 distillation.",
                    ResourceWarning,
                    stacklevel=2,
                )
        hidden_states, block_outputs, mats_fwd, mats_bwd = [], [], [], []
        for block, cond_layer in zip(self.blocks, self.conditioning):
            if return_hidden_states:
                hidden_states.append(output)
            if return_matrices:
                h_normed = self._mixer_input(block, cond_layer, output, combined_cond)
                m_fwd, m_bwd = block.materialize_matrices(h_normed)
                mats_fwd.append(m_fwd)
                mats_bwd.append(m_bwd)
            output = self._block_forward(block, cond_layer, output, combined_cond)
            block_outputs.append(output)
        info = {
            "hidden_states": hidden_states,
            "block_outputs": block_outputs,
            "matrices_fwd": mats_fwd,
            "matrices_bwd": mats_bwd,
        }
        return output, info

    def _mixer_input(self, block, cond_layer, x, combined_cond):
        """Reconstruct the normalized, conditioned tensor a block feeds to its mixer.

        Mirrors :meth:`_block_forward` up to (but excluding) the SSM mix, so the
        matrices from :meth:`Mamba2Block.materialize_matrices` correspond exactly to
        what that block computes for this input.
        """
        if self.conditioning_type == "adaln":
            params = cond_layer(combined_cond)
            scale, shift = params[0], params[1]
            h = block.norm(x)
            return h * (1 + scale) + shift
        return block.norm(cond_layer(x, combined_cond))

    def _block_forward(self, block, cond_layer, x, combined_cond):
        """Condition + mix for a single denoiser block.

        Dropout is applied inside each block on the sub-layer branches (not here
        on the residual stream), preserving the AdaLN-Zero identity-at-init guarantee.
        """
        if self.conditioning_type == "adaln":
            params = cond_layer(combined_cond)
            if len(params) not in (3, 6):
                raise ValueError(
                    f"AdaLN conditioning produced {len(params)} params; expected 3 or 6"
                )
            out = block.forward_adaln(x, *params)
        else:
            out = block(cond_layer(x, combined_cond))
        return out


class _AttnBlock(nn.Module):
    """Hand-rolled pre-norm bidirectional transformer encoder block.

    Each block applies:
        h = norm1(x)
        multi-head self-attention (NO causal mask) -> x = x + o_proj(a)
        h = norm2(x)
        x = x + ff2(gelu(ff1(h)))

    Separate q, k, v, o Linear projections are used (not fused) so the MLX
    backend can mirror the block weight-for-weight without rewriting shapes.

    Args:
        d_model: Input/output hidden dimension.
        nhead: Number of attention heads.  Must divide ``d_model`` evenly.
    """

    def __init__(self, d_model: int, nhead: int):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_model * 4)
        self.ff2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm bidirectional self-attention + FFN with residuals.

        Args:
            x: Input tensor ``[B, L, d_model]``.

        Returns:
            Output tensor ``[B, L, d_model]``.
        """
        B, L, D = x.shape
        # --- self-attention branch ---
        h = self.norm1(x)
        q = self.q_proj(h).view(B, L, self.nhead, self.head_dim).transpose(1, 2)  # [B, H, L, hd]
        k = self.k_proj(h).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        scale = self.head_dim ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)  # [B, H, L, L]
        a = torch.matmul(attn, v)  # [B, H, L, hd]
        a = a.transpose(1, 2).contiguous().view(B, L, D)  # [B, L, D]
        x = x + self.o_proj(a)
        # --- feed-forward branch ---
        h = self.norm2(x)
        x = x + self.ff2(torch.nn.functional.gelu(self.ff1(h)))
        return x


class DenoisingHead(nn.Module):
    """Project denoised embeddings to token logits, with optional weight tying.

    Args:
        d_model: Model hidden dimension.
        vocab_size: Vocabulary size.
        use_weight_tying: Tie the projection with the embedding matrix.
        embedding_weight: Embedding weight for tying (optional).
        use_norm: Apply LayerNorm + learnable logit scale before projection.
        head_type: ``"linear"`` (default, per-position projection, current behavior)
            or ``"attn"`` (context-aware bidirectional self-attention blocks before
            the projection).  Old checkpoints without this key default to ``"linear"``.
        head_attn_layers: Number of transformer encoder blocks when
            ``head_type="attn"`` (default 2).
        head_attn_heads: Maximum number of attention heads for the attn blocks
            (default 4).  The actual number used is the largest divisor of
            ``d_model`` that is ``<= head_attn_heads`` (minimum 1).
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        use_weight_tying: bool = False,
        embedding_weight: Optional[torch.Tensor] = None,
        use_norm: bool = False,
        head_type: str = "linear",
        head_attn_layers: int = 2,
        head_attn_heads: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_weight_tying = use_weight_tying
        self.head_type = head_type

        # --- context-aware attention blocks (head_type=="attn") ---
        self.attn_blocks: Optional[nn.ModuleList] = None
        if head_type == "attn":
            # Pick the largest divisor of d_model that is <= head_attn_heads (min 1).
            nhead = 1
            for candidate in range(head_attn_heads, 0, -1):
                if d_model % candidate == 0:
                    nhead = candidate
                    break
            self.attn_blocks = nn.ModuleList(
                [_AttnBlock(d_model=d_model, nhead=nhead) for _ in range(head_attn_layers)]
            )

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
        """Project denoised embeddings ``[B, L, d_model]`` to logits ``[B, L, vocab]``.

        When ``head_type="attn"``, the input first passes through bidirectional
        self-attention blocks so each position can attend to the full sequence
        before the final vocabulary projection.  The forward signature is
        identical to the ``"linear"`` path so trainer/sampler callers are
        unchanged.

        Args:
            x: Decoded embeddings ``[B, L, d_model]``.
            embedding_weight: Optional embedding matrix for weight-tied projection.

        Returns:
            Token logits ``[B, L, vocab_size]``.
        """
        # The sampler/decoder can hand us an fp32 tensor (fp32 timestep math upcasts the
        # latent), but the head's matmuls need an exact dtype match. Cast to the head dtype.
        if self.use_weight_tying:
            _ew = embedding_weight if embedding_weight is not None else self.embedding_weight
            x = x.to(_ew.dtype)
        elif isinstance(self.projection, nn.Linear):
            x = x.to(self.projection.weight.dtype)

        # Context-aware mixing (no-op for head_type=="linear").
        if self.attn_blocks is not None:
            for blk in self.attn_blocks:
                x = blk(x)

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
