"""Vectorized diagonal selective-scan (Mamba SSM) recurrence.

This module implements the *correct* diagonal Mamba selective scan and a
length-parallel (no Python loop over the sequence) vectorized variant suitable
for the CPU / MPS code paths used by :class:`SimpleMamba2`.

Recurrence
----------
Given, per batch ``B`` and sequence length ``L``:

* ``dt``   : timestep deltas, shape ``[B, L, Din]`` (positive, e.g. softplus).
* ``A``    : SSM state-decay, shape ``[Din, Dstate]`` (negative real).
* ``Bmat`` : input->state projection, shape ``[B, L, Dstate]``.
* ``C``    : state->output projection, shape ``[B, L, Dstate]``.
* ``x``    : SSM input, shape ``[B, L, Din]``.

Discretization (zero-order hold on ``A``, Euler on ``B``)::

    dA  = exp(dt[..., None] * A)                       -> [B, L, Din, Dstate]
    dBx = dt[..., None] * Bmat[:, :, None, :] * x[..., None]  -> [B, L, Din, Dstate]

First-order linear recurrence over time (``h_{-1} = 0``)::

    h_t = dA_t * h_{t-1} + dBx_t                        -> [B, L, Din, Dstate]
    y_t = sum_s C_t[s] * h_t[..., s]                    -> [B, L, Din]

Note that the inner dimension ``Din`` stays fully independent (one scalar SSM
state per ``(Din, Dstate)`` pair). This is the property the legacy
``SimpleMamba2`` forward loop violated: it summed the ``B * x`` contribution
over ``Din`` before the state update, collapsing the inner dimension. The
functions here keep ``Din`` independent and are the intended replacement.

Closed form used by the vectorized scan
----------------------------------------
The scalar recurrence ``h_t = a_t * h_{t-1} + b_t`` has the closed form::

    h_t = P_t * cumsum_{j<=t}( b_j / P_j ),   where  P_t = cumprod_{k<=t}( a_k )

Because ``A < 0`` and ``dt > 0`` we have ``a_k = exp(dt * A) in (0, 1]``, so the
running product ``P_t`` decays toward 0 and the naive ``b_j / P_j`` term can
overflow for long sequences. We therefore default to a **chunked associative
scan**: the cumprod/cumsum identity is applied independently inside fixed-size
chunks (where ``P`` does not decay far), and the per-chunk final states are
combined with a short scan across chunks. This keeps the heavy work
parallel/vectorized while bounding the dynamic range. The naive single-pass
identity is also exposed (``_scan_cumprod_trick``) for reference and for short
sequences. See :func:`selective_scan` for the ``stable`` / ``chunk_size`` knobs.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

__all__ = [
    "selective_scan_sequential",
    "selective_scan",
    "bidirectional_selective_scan",
]


def _discretize(
    dt: torch.Tensor,
    A: torch.Tensor,
    Bmat: torch.Tensor,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Discretize the continuous SSM parameters.

    Args:
        dt: Timestep deltas ``[B, L, Din]``.
        A: State-decay matrix ``[Din, Dstate]`` (negative real).
        Bmat: Input->state projection ``[B, L, Dstate]``.
        x: SSM input ``[B, L, Din]``.

    Returns:
        Tuple ``(dA, dBx)`` each of shape ``[B, L, Din, Dstate]`` where
        ``dA = exp(dt * A)`` and ``dBx = dt * Bmat * x``.
    """
    # dt: [B, L, Din] -> [B, L, Din, 1]; A: [Din, Dstate] broadcasts over B, L.
    dA = torch.exp(dt.unsqueeze(-1) * A)  # [B, L, Din, Dstate]
    # dBx_t[i, s] = dt_t[i] * Bmat_t[s] * x_t[i]
    dBx = dt.unsqueeze(-1) * Bmat.unsqueeze(2) * x.unsqueeze(-1)  # [B, L, Din, Dstate]
    return dA, dBx


def selective_scan_sequential(
    dt: torch.Tensor,
    A: torch.Tensor,
    Bmat: torch.Tensor,
    C: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Reference selective scan via an explicit Python loop over the sequence.

    This is the unambiguous ground-truth implementation used by the tests to
    validate the vectorized :func:`selective_scan`. It is O(L) sequential and
    therefore slow, but numerically the most trustworthy.

    Args:
        dt: Timestep deltas ``[B, L, Din]`` (positive).
        A: State-decay matrix ``[Din, Dstate]`` (negative real).
        Bmat: Input->state projection ``[B, L, Dstate]``.
        C: State->output projection ``[B, L, Dstate]``.
        x: SSM input ``[B, L, Din]``.

    Returns:
        Output ``y`` of shape ``[B, L, Din]``.
    """
    batch, length, d_inner = dt.shape
    d_state = A.shape[1]

    dA, dBx = _discretize(dt, A, Bmat, x)  # [B, L, Din, Dstate]

    h = torch.zeros(batch, d_inner, d_state, dtype=dt.dtype, device=dt.device)
    ys = []
    for t in range(length):
        h = dA[:, t] * h + dBx[:, t]  # [B, Din, Dstate]
        # y_t[i] = sum_s C_t[s] * h_t[i, s]
        y_t = torch.einsum("bs,bis->bi", C[:, t], h)  # [B, Din]
        ys.append(y_t)
    return torch.stack(ys, dim=1)  # [B, L, Din]


def _scan_cumprod_trick(dA: torch.Tensor, dBx: torch.Tensor) -> torch.Tensor:
    """Solve ``h_t = dA_t * h_{t-1} + dBx_t`` with the cumprod/cumsum identity.

    Implements ``h_t = P_t * cumsum(dBx / P)`` with ``P = cumprod(dA)`` along the
    time axis (no Python loop). This is exact in real arithmetic but loses
    precision / overflows once ``cumprod(dA)`` underflows, so it is best for
    short chunks. Used as the per-chunk kernel by :func:`_scan_chunked` and
    exposed directly via ``selective_scan(..., stable=False)``.

    Args:
        dA: Per-step multipliers ``[B, L, Din, Dstate]`` in ``(0, 1]``.
        dBx: Per-step additive inputs ``[B, L, Din, Dstate]``.

    Returns:
        States ``h`` of shape ``[B, L, Din, Dstate]``.
    """
    # Cumulative product P_t = prod_{k<=t} dA_k along the length axis (dim=1).
    p = torch.cumprod(dA, dim=1)  # [B, L, Din, Dstate]
    # h_t = P_t * sum_{j<=t} dBx_j / P_j
    h = p * torch.cumsum(dBx / p, dim=1)
    return h


def _scan_chunked(
    dA: torch.Tensor,
    dBx: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Numerically-stable length-parallel scan via fixed-size chunks.

    The sequence is split into chunks of ``chunk_size``. Within each chunk the
    cumprod/cumsum identity is applied *locally* (so ``cumprod(dA)`` only decays
    across at most ``chunk_size`` steps, bounding the dynamic range). Each chunk
    is then corrected by the carried-in state ``h_carry`` from all preceding
    chunks: for a chunk-local product ``Pc_t = prod`` of ``dA`` within the chunk,
    the full state is ``h_t = h_local_t + Pc_t * h_carry``. The carry is updated
    chunk-by-chunk (a short, ``L / chunk_size``-length sequential loop), which is
    cheap relative to the per-element work done in parallel inside chunks.

    Args:
        dA: Per-step multipliers ``[B, L, Din, Dstate]``.
        dBx: Per-step additive inputs ``[B, L, Din, Dstate]``.
        chunk_size: Number of timesteps per chunk.

    Returns:
        States ``h`` of shape ``[B, L, Din, Dstate]``.
    """
    batch, length, d_inner, d_state = dA.shape
    if length <= chunk_size:
        return _scan_cumprod_trick(dA, dBx)

    n_chunks = math.ceil(length / chunk_size)
    h_carry = torch.zeros(batch, d_inner, d_state, dtype=dA.dtype, device=dA.device)
    out_chunks = []
    for c in range(n_chunks):
        lo = c * chunk_size
        hi = min(lo + chunk_size, length)
        dA_c = dA[:, lo:hi]  # [B, Lc, Din, Dstate]
        dBx_c = dBx[:, lo:hi]

        # Local (carry-free) solution within the chunk.
        h_local = _scan_cumprod_trick(dA_c, dBx_c)  # [B, Lc, Din, Dstate]
        # Chunk-local cumulative product, used to propagate the incoming carry.
        pc = torch.cumprod(dA_c, dim=1)  # [B, Lc, Din, Dstate]

        # Add contribution of the carried-in state.
        h_c = h_local + pc * h_carry.unsqueeze(1)  # broadcast carry over Lc
        out_chunks.append(h_c)

        # New carry = last state of this chunk.
        h_carry = h_c[:, -1]

    return torch.cat(out_chunks, dim=1)  # [B, L, Din, Dstate]


def selective_scan(
    dt: torch.Tensor,
    A: torch.Tensor,
    Bmat: torch.Tensor,
    C: torch.Tensor,
    x: torch.Tensor,
    *,
    stable: bool = True,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Vectorized diagonal selective scan (no Python loop over the sequence).

    Computes the same result as :func:`selective_scan_sequential` but solves the
    linear recurrence in closed form using cumulative products/sums, so the
    length dimension is processed in parallel. See the module docstring for the
    exact recurrence and the closed form.

    Numerical stability: because ``dA = exp(dt * A) in (0, 1]``, a single-pass
    cumprod can underflow on long sequences. With ``stable=True`` (default) the
    scan is computed in chunks of ``chunk_size`` so the running product never
    decays across more than ``chunk_size`` steps; the per-chunk states are
    stitched together by carrying the boundary state. With ``stable=False`` a
    single-pass cumprod/cumsum is used (faster, fine for short sequences). The
    operation is fully differentiable in both modes.

    Args:
        dt: Timestep deltas ``[B, L, Din]`` (positive).
        A: State-decay matrix ``[Din, Dstate]`` (negative real).
        Bmat: Input->state projection ``[B, L, Dstate]``.
        C: State->output projection ``[B, L, Dstate]``.
        x: SSM input ``[B, L, Din]``.
        stable: If ``True`` use the chunked associative scan; otherwise use the
            single-pass cumprod/cumsum identity.
        chunk_size: Chunk length used when ``stable=True``. Must be positive.

    Returns:
        Output ``y`` of shape ``[B, L, Din]``.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    dA, dBx = _discretize(dt, A, Bmat, x)  # [B, L, Din, Dstate]

    if stable:
        h = _scan_chunked(dA, dBx, chunk_size)
    else:
        h = _scan_cumprod_trick(dA, dBx)

    # y_t[i] = sum_s C_t[s] * h_t[i, s]; C: [B, L, Dstate], h: [B, L, Din, Dstate]
    y = torch.einsum("bls,blis->bli", C, h)  # [B, L, Din]
    return y


def bidirectional_selective_scan(
    dt_fwd: torch.Tensor,
    A_fwd: torch.Tensor,
    Bmat_fwd: torch.Tensor,
    C_fwd: torch.Tensor,
    x_fwd: torch.Tensor,
    dt_bwd: torch.Tensor,
    A_bwd: torch.Tensor,
    Bmat_bwd: torch.Tensor,
    C_bwd: torch.Tensor,
    x_bwd: torch.Tensor,
    *,
    stable: bool = True,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Bidirectional selective scan: forward + reversed, recombined by sum.

    Runs :func:`selective_scan` once on the forward inputs and once on the
    *reversed* sequence using the separate backward inputs supplied by the
    caller, then re-flips the backward output and sums the two directions. The
    caller provides independent ``(dt, A, Bmat, C, x)`` for each direction so
    that the two passes may use distinct (e.g. separately-projected) parameters,
    mirroring the typical bidirectional-Mamba design.

    The reversal is performed internally with ``torch.flip`` along the length
    axis for both the inputs and the produced output, so the returned tensor is
    in forward (natural) time order.

    Args:
        dt_fwd: Forward timestep deltas ``[B, L, Din]``.
        A_fwd: Forward state-decay ``[Din, Dstate]``.
        Bmat_fwd: Forward input->state projection ``[B, L, Dstate]``.
        C_fwd: Forward state->output projection ``[B, L, Dstate]``.
        x_fwd: Forward SSM input ``[B, L, Din]``.
        dt_bwd: Backward timestep deltas ``[B, L, Din]`` (natural order).
        A_bwd: Backward state-decay ``[Din, Dstate]``.
        Bmat_bwd: Backward input->state projection ``[B, L, Dstate]``.
        C_bwd: Backward state->output projection ``[B, L, Dstate]``.
        x_bwd: Backward SSM input ``[B, L, Din]``.
        stable: Forwarded to :func:`selective_scan`.
        chunk_size: Forwarded to :func:`selective_scan`.

    Returns:
        Combined output ``[B, L, Din]`` (sum of both directions, forward order).
    """
    y_fwd = selective_scan(
        dt_fwd, A_fwd, Bmat_fwd, C_fwd, x_fwd, stable=stable, chunk_size=chunk_size
    )

    # Reverse the backward inputs along the length axis (dim=1).
    flip = lambda t: torch.flip(t, dims=[1])  # noqa: E731
    y_bwd_rev = selective_scan(
        flip(dt_bwd),
        A_bwd,
        flip(Bmat_bwd),
        flip(C_bwd),
        flip(x_bwd),
        stable=stable,
        chunk_size=chunk_size,
    )
    # Re-flip back to natural time order before combining.
    y_bwd = torch.flip(y_bwd_rev, dims=[1])

    return y_fwd + y_bwd
