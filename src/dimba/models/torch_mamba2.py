"""Pure-PyTorch, weight-compatible reimplementation of ``mamba_ssm.Mamba2`` (SSD).

Runs the Mamba-2 mixer on **CPU/MPS with no CUDA / ``mamba_ssm`` dependency**, while
loading a checkpoint trained with the real ``mamba_ssm.Mamba2`` CUDA kernel *as-is*
(strict ``state_dict`` match). This lets DIMBA checkpoints trained with
``--use-mamba-ssm`` run on Apple Silicon and other non-CUDA machines.

The math is a line-for-line port of the canonical implementation
(``mamba_ssm/modules/mamba2.py`` + ``modules/ssd_minimal.py`` +
``ops/triton/layernorm_gated.py``); every subtle detail was verified against that
source: the ``in_proj`` split order ``[z, xBC, dt]``, ``conv_dim`` and the depthwise
causal conv, ``dt = softplus(dt + dt_bias)``, scalar-per-head ``A = -exp(A_log)``, the
SSD discretization ``S_t = exp(dt_t*A) S_{t-1} + (dt_t*x_t) ⊗ B_t`` with the ``D*x``
skip applied *outside* the scan, and the gated RMSNorm with ``norm_before_gate=False``
(gate then normalize). Mamba-2 / SSD: Dao & Gu, "Transformers are SSMs", ICML 2024
(arXiv:2405.21060).

Two scans are provided: a clear sequential reference (:meth:`_scan_sequential`) and a
fast chunked/parallel scan (:meth:`_scan_chunked`, matmul-based, the default) that
avoids the Python loop. They are numerically equivalent (checked in tests).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNormGated(nn.Module):
    """Gated RMSNorm matching ``mamba_ssm`` with ``norm_before_gate=False``.

    Registers ``weight`` only (no bias), so it loads ``norm.weight`` from a Mamba-2
    checkpoint. With ``norm_before_gate=False`` (the Mamba-2 default) the gate is
    applied **before** the RMS normalization::

        out = ( (x * silu(z)) / sqrt(mean((x*silu(z))**2) + eps) ) * weight
    """

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        if z is not None:
            x = x * F.silu(z)
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * self.weight


class TorchMamba2(nn.Module):
    """Pure-PyTorch Mamba-2 (SSD) mixer, weight-compatible with ``mamba_ssm.Mamba2``.

    Mixer-only: ``[B, L, d_model] -> [B, L, d_model]``. The enclosing block owns the
    pre-norm and residual (matching :class:`dimba.models.denoiser.Mamba2Block`).
    Parameter *names and shapes* match ``mamba_ssm.Mamba2`` so a checkpoint trained
    with the CUDA kernel loads with ``strict=True``.

    Args:
        d_model: Mixer model dimension.
        d_state: SSM state dimension (per head/group).
        d_conv: Causal conv kernel size.
        expand: Inner expansion factor (``d_inner = expand * d_model``).
        headdim: SSD head dimension (``d_inner`` must be divisible by it).
        ngroups: Number of B/C groups (shared across heads within a group).
        chunk_size: Chunk length for the fast chunked scan.
        use_chunked: Use the matmul-based chunked scan (fast). If False, the
            sequential reference scan is used.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 128,
        use_chunked: bool = True,
        **_compat,  # tolerate SimpleMamba2-style kwargs (d_expand, dt_rank) harmlessly
    ):
        super().__init__()
        if _compat.get("d_expand"):
            expand = _compat["d_expand"]
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.ngroups = ngroups
        self.chunk_size = chunk_size
        self.use_chunked = use_chunked

        self.d_inner = expand * d_model
        assert self.d_inner % headdim == 0, "d_inner must be divisible by headdim"
        self.nheads = self.d_inner // headdim
        assert self.nheads % ngroups == 0, "nheads must be divisible by ngroups"
        self.conv_dim = self.d_inner + 2 * ngroups * d_state
        d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads

        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)
        self.conv1d = nn.Conv1d(
            self.conv_dim, self.conv_dim, kernel_size=d_conv,
            groups=self.conv_dim, padding=d_conv - 1, bias=True,
        )
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.A_log = nn.Parameter(torch.zeros(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = RMSNormGated(self.d_inner, eps=1e-5)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        B, L, _ = u.shape
        orig_dtype = u.dtype
        u = u.float()  # run the mixer in fp32 (MPS fp16 is flaky; A_log exp needs precision)

        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.conv_dim, self.nheads], dim=-1
        )

        # Causal depthwise conv + SiLU (matches causal_conv1d_fn with activation="silu").
        xBC = self.conv1d(xBC.transpose(1, 2))[..., :L].transpose(1, 2)
        xBC = F.silu(xBC)

        x, Bm, Cm = torch.split(
            xBC,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )

        dt = F.softplus(dt + self.dt_bias)        # [B, L, nheads]
        A = -torch.exp(self.A_log.float())        # [nheads], scalar per head (negative)

        x = x.view(B, L, self.nheads, self.headdim)            # [B, L, h, p]
        Bm = Bm.view(B, L, self.ngroups, self.d_state)         # [B, L, g, n]
        Cm = Cm.view(B, L, self.ngroups, self.d_state)
        rep = self.nheads // self.ngroups
        Bm = Bm.repeat_interleave(rep, dim=2)                  # [B, L, h, n]
        Cm = Cm.repeat_interleave(rep, dim=2)

        scan = self._scan_chunked if self.use_chunked else self._scan_sequential
        y = scan(x, dt, A, Bm, Cm)                             # [B, L, h, p]

        y = y + self.D.view(1, 1, self.nheads, 1) * x          # D skip (per-head scalar)
        y = y.reshape(B, L, self.d_inner)
        y = self.norm(y, z)                                    # gated RMSNorm (gate then norm)
        out = self.out_proj(y)
        return out.to(orig_dtype)

    # ── SSM scans ───────────────────────────────────────────────────────────────
    @staticmethod
    def _scan_sequential(x, dt, A, Bm, Cm):
        """Reference SSD recurrence (``ssd_minimal_discrete`` unrolled).

        ``S_t = exp(dt_t*A) * S_{t-1} + (dt_t * x_t) ⊗ B_t`` ; ``y_t = C_t · S_t``.
        Shapes: x:[B,L,h,p] dt:[B,L,h] A:[h] Bm,Cm:[B,L,h,n] -> y:[B,L,h,p].
        """
        B, L, H, P = x.shape
        dA = torch.exp(dt * A)                                # [B, L, h]
        S = x.new_zeros(B, H, P, Bm.shape[-1])               # [B, h, p, n]
        ys = []
        for t in range(L):
            dA_t = dA[:, t].view(B, H, 1, 1)
            dt_t = dt[:, t].view(B, H, 1, 1)
            xt = x[:, t].unsqueeze(-1)                        # [B, h, p, 1]
            Bt = Bm[:, t].unsqueeze(-2)                       # [B, h, 1, n]
            S = dA_t * S + dt_t * xt * Bt
            Ct = Cm[:, t].unsqueeze(-2)                       # [B, h, 1, n]
            ys.append((S * Ct).sum(-1))                       # [B, h, p]
        return torch.stack(ys, dim=1)                         # [B, L, h, p]

    def _scan_chunked(self, x, dt, A, Bm, Cm):
        """Fast chunked/parallel SSD scan (matmul-based; no Python per-step loop).

        Equivalent to :meth:`_scan_sequential` but processes the sequence in chunks of
        ``self.chunk_size`` and carries state across chunks, turning the O(L) recurrence
        into a short loop over ``ceil(L/chunk)`` chunks with batched matmuls inside.
        Mirrors ``mamba_ssm``'s ``ssd_minimal_discrete`` decomposition (intra-chunk
        diagonal + carried inter-chunk state).
        """
        B, L, H, P = x.shape
        N = Bm.shape[-1]
        cs = self.chunk_size
        # Absorb dt into x and A (the ssd_minimal convention: X*dt, A*dt).
        xdt = x * dt.unsqueeze(-1)                            # [B, L, h, p]
        Adt = A.view(1, 1, H) * dt                           # [B, L, h]

        y = x.new_zeros(B, L, H, P)
        S = x.new_zeros(B, H, P, N)                          # carried state across chunks
        for start in range(0, L, cs):
            end = min(start + cs, L)
            cl = end - start
            xc = xdt[:, start:end]                            # [B, cl, h, p]
            ac = Adt[:, start:end]                            # [B, cl, h]
            bc = Bm[:, start:end]                             # [B, cl, h, n]
            cc = Cm[:, start:end]                             # [B, cl, h, n]

            # cumulative log-decay within the chunk: a_cum[t] = sum_{s<=t} A*dt_s
            a_cum = torch.cumsum(ac, dim=1)                   # [B, cl, h]

            # (1) contribution from the state carried into this chunk:
            #     y_carry_t = (C_t · S_in) * exp(a_cum_t)
            #     S_in: [B,h,p,n]; C_t: [B,h,n] -> [B,h,p]
            y_carry = torch.einsum("bthn,bhpn->bthp", cc, S) * torch.exp(a_cum).unsqueeze(-1)

            # (2) intra-chunk (diagonal) contribution via a decay matrix:
            #     Lmat[t,s] = exp(a_cum_t - a_cum_s) for s<=t else 0
            #     y_diag_t = sum_{s<=t} (C_t·B_s) * Lmat[t,s] * x_s
            t_idx = a_cum.unsqueeze(2)                        # [B, cl, 1, h]
            s_idx = a_cum.unsqueeze(1)                        # [B, 1, cl, h]
            logdecay = t_idx - s_idx                          # [B, cl, cl, h]
            # Mask the upper triangle (s>t) to -inf BEFORE exp, so it becomes 0 —
            # otherwise exp of a large positive exponent overflows to inf and inf*0=NaN.
            mask = torch.tril(torch.ones(cl, cl, device=x.device, dtype=torch.bool))
            logdecay = logdecay.masked_fill(~mask.view(1, cl, cl, 1), float("-inf"))
            decay = torch.exp(logdecay)                       # [B, cl, cl, h]
            cb = torch.einsum("bthn,bshn->btsh", cc, bc)     # [B, cl, cl, h]  (C_t·B_s)
            scores = cb * decay                               # [B, cl, cl, h]
            y_diag = torch.einsum("btsh,bshp->bthp", scores, xc)

            y[:, start:end] = y_carry + y_diag

            # update carried state to end of chunk:
            #   S_out = exp(a_cum_last) * S_in + sum_s exp(a_cum_last - a_cum_s) (x_s ⊗ B_s)
            a_last = a_cum[:, -1]                              # [B, h]
            state_decay = torch.exp(a_last.unsqueeze(1) - a_cum)  # [B, cl, h]
            dstate = torch.einsum("bthn,bthp,bth->bhpn", bc, xc, state_decay)  # [B,h,p,n]
            S = torch.exp(a_last).view(B, H, 1, 1) * S + dstate
        return y
