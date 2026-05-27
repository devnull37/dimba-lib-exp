"""Pure-PyTorch Mamba selective-scan mixer (CPU/MPS fallback).

A minimal, dependency-free selective state-space mixer in the spirit of Mamba
(Gu & Dao, 2023). It is a **mixer only**: the enclosing block owns normalization
and the residual connection (matching the ``mamba_ssm`` API), so this is a drop-in
replacement for the CUDA kernels.

Correctness fixes vs. the previous implementation:

* The state matrix ``A`` is now negative (``-exp(A_log)``), making the discrete
  recurrence ``h_t = exp(dt*A) * h_{t-1} + dt * B * x`` contractive/stable. The old
  code used a positive ``A = +1`` (divergent).
* Each inner channel keeps its own input (the old code summed over the inner
  dimension via ``B_x.sum(dim=1)``, collapsing it).
* No internal LayerNorm / residual (the old code applied both *again* on top of the
  enclosing block's, double-counting them).

When :mod:`dimba.models.parallel_scan` is available, the sequential Python scan is
replaced by a vectorized associative scan.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:  # Vectorized scan (built by the performance work package).
    from .parallel_scan import selective_scan as _parallel_selective_scan

    _HAS_PARALLEL_SCAN = True
except Exception:  # pragma: no cover - module may not exist yet
    _HAS_PARALLEL_SCAN = False


class SimpleMamba2(nn.Module):
    """Selective-scan SSM mixer ``[B, L, d_model] -> [B, L, d_model]``.

    Args:
        d_model: Model dimension.
        d_state: SSM state dimension.
        d_expand: Inner expansion factor.
        dt_rank: Unused (kept for signature compatibility).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_expand: int = 2,
        dt_rank: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_expand = d_expand
        self.d_inner = int(d_model * d_expand)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.dt_proj = nn.Linear(d_model, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model)

        # S4D-real initialization: A = -[1..d_state] per inner channel, stored as log.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix the (pre-normalized) input. Returns the mixer output (no residual)."""
        z, x_in = self.in_proj(x).chunk(2, dim=-1)  # [B, L, d_inner] each
        dt = F.softplus(self.dt_proj(x))  # [B, L, d_inner]
        b_mat = self.B_proj(x)  # [B, L, d_state]
        c_mat = self.C_proj(x)  # [B, L, d_state]
        a = -torch.exp(self.A_log)  # [d_inner, d_state]

        y = self._scan(dt, a, b_mat, c_mat, x_in)  # [B, L, d_inner]
        y = y + x_in * self.D  # D skip connection
        y = y * F.silu(z)  # gating
        return self.out_proj(y)

    def _scan(
        self,
        dt: torch.Tensor,
        a: torch.Tensor,
        b_mat: torch.Tensor,
        c_mat: torch.Tensor,
        x_in: torch.Tensor,
    ) -> torch.Tensor:
        """Selective scan ``h_t = exp(dt*A) h_{t-1} + dt*B*x``; ``y_t = C_t . h_t``."""
        if _HAS_PARALLEL_SCAN:
            try:
                y = _parallel_selective_scan(dt, a, b_mat, c_mat, x_in)
                # The closed-form parallel scan can underflow (cumprod -> 0) for a
                # large state-decay over a long sequence, yielding NaN/Inf. Only use
                # it when finite; otherwise fall through to the stable sequential
                # scan below (NaN is not an exception, so it must be checked).
                if torch.isfinite(y).all():
                    return y
            except Exception:  # pragma: no cover - fall back on any incompatibility
                pass

        batch, length, d_inner = x_in.shape
        d_state = a.shape[-1]
        h = x_in.new_zeros(batch, d_inner, d_state)
        d_a = torch.exp(dt.unsqueeze(-1) * a)  # [B, L, d_inner, d_state]
        d_bx = dt.unsqueeze(-1) * b_mat.unsqueeze(2) * x_in.unsqueeze(-1)  # [B, L, d_inner, d_state]
        outputs = []
        for t in range(length):
            h = d_a[:, t] * h + d_bx[:, t]
            outputs.append(torch.einsum("bds,bs->bd", h, c_mat[:, t]))
        return torch.stack(outputs, dim=1)  # [B, L, d_inner]


class SimpleMamba2Block(nn.Module):
    """Pre-norm + :class:`SimpleMamba2` mixer + residual (standalone convenience block)."""

    def __init__(self, d_model: int = 512, d_state: int = 16, d_expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.mamba = SimpleMamba2(d_model=d_model, d_state=d_state, d_expand=d_expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-norm + mix + residual."""
        return x + self.mamba(self.norm(x))
